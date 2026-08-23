"""Generic multi-step tool-use loop shared by the Application Investigation
Agent and the Verification Agent. Provider-agnostic: works identically
against AnthropicLLMClient (genuine model-driven tool selection) and
FakeLLMClient (deterministic offline stand-in) because both only need to
implement LLMClient.complete().
"""

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..llm.base import LLMClient, LLMMessage, text_message
from ..tools.base import Tool


class ToolCallRecord(BaseModel):
    tool_call_id: str
    tool_name: str
    query_summary: str
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


async def run_tool_loop(
    llm: LLMClient,
    tools: List[Tool],
    system: str,
    user_text: str,
    max_steps: int = 8,
) -> List[ToolCallRecord]:
    tools_by_name = {t.name: t for t in tools}
    tool_specs = [t.spec() for t in tools]
    messages: List[LLMMessage] = [text_message("user", user_text)]
    records: List[ToolCallRecord] = []

    for _ in range(max_steps):
        response = await llm.complete(system=system, messages=messages, tools=tool_specs, max_tokens=1500)
        if not response.tool_calls:
            break

        messages.append(
            LLMMessage(
                role="assistant",
                content=[{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in response.tool_calls],
            )
        )

        result_blocks = []
        for call in response.tool_calls:
            tool = tools_by_name.get(call.name)
            if tool is None:
                records.append(
                    ToolCallRecord(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        query_summary=f"unknown tool {call.name!r}",
                        ok=False,
                        error=f"unknown tool {call.name!r}",
                    )
                )
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": json.dumps({"error": "unknown tool"})}
                )
                continue

            try:
                tool_input = tool.input_schema.model_validate(call.input)
                result = await tool.execute(tool_input)
            except Exception as exc:  # malformed tool input from the model
                records.append(
                    ToolCallRecord(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        query_summary=f"{call.name}({call.input})",
                        ok=False,
                        error=str(exc),
                    )
                )
                result_blocks.append({"type": "tool_result", "tool_use_id": call.id, "content": json.dumps({"error": str(exc)})})
                continue

            records.append(
                ToolCallRecord(
                    tool_call_id=call.id,
                    tool_name=result.tool_name,
                    query_summary=result.query_summary,
                    ok=result.ok,
                    data=result.data,
                    error=result.error,
                )
            )
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(result.data if result.ok else {"error": result.error}, default=str),
                }
            )

        messages.append(LLMMessage(role="user", content=result_blocks))

    return records
