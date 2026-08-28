"""Generic multi-step tool-use loop shared by the Application Investigation
Agent and the Verification Agent. Provider-agnostic: works identically
against AnthropicLLMClient (genuine model-driven tool selection) and
FakeLLMClient (deterministic offline stand-in) because both only need to
implement LLMClient.complete().
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from ..llm.base import LLMClient, LLMMessage, text_message
from ..tools.base import Tool

logger = logging.getLogger(__name__)


class ToolCallRecord(BaseModel):
    tool_call_id: str
    tool_name: str
    query_summary: str
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


_RETRY_NUDGE = (
    "You did not call any tools. You are required to gather at least one piece of new, "
    "independent evidence per the CONTEXT in the system prompt before concluding. Call one "
    "of the available tools now."
)


def _digest_tool_result(record: ToolCallRecord) -> str:
    """A short one-line stand-in for an older tool call's full result, used
    by _trim_stale_tool_results in place of the raw JSON once a call is old
    enough to no longer be at full fidelity in the resent history."""
    if not record.ok:
        return f"[earlier] {record.tool_name} query={record.query_summary} -> error: {record.error}"
    item_count: Optional[int] = None
    if isinstance(record.data, dict):
        for value in record.data.values():
            if isinstance(value, list):
                item_count = len(value)
                break
    outcome = f"{item_count} item(s)" if item_count is not None else "ok"
    return f"[earlier] {record.tool_name} query={record.query_summary} -> {outcome}"


def _trim_stale_tool_results(
    messages: List[LLMMessage],
    records: List[ToolCallRecord],
    result_positions: Dict[str, Tuple[int, int, str]],
    trimmed: Set[str],
    trim_after_calls: int,
    keep_full_fidelity: int,
) -> None:
    """Once more tool calls have been made than trim_after_calls, replace
    the *result* content (not the call itself -- a tool_result still needs
    a matching tool_use to form a valid message for a real provider) of
    every call older than the most recent keep_full_fidelity with its
    one-line digest. Only affects what's resent to the model inside this
    loop's `messages` list -- `records` (what callers actually consume) is
    never touched, so nothing downstream loses fidelity.

    Idempotent: `trimmed` tracks tool_call_ids already rewritten so a call
    already digested is left alone on later invocations rather than
    reprocessed every step."""
    if len(records) <= trim_after_calls:
        return
    cutoff = len(records) - keep_full_fidelity
    for record in records[:cutoff]:
        if record.tool_call_id in trimmed:
            continue
        position = result_positions.get(record.tool_call_id)
        if position is None:
            continue
        message_index, block_index, digest = position
        messages[message_index].content[block_index]["content"] = digest
        trimmed.add(record.tool_call_id)


async def run_tool_loop(
    llm: LLMClient,
    tools: List[Tool],
    system: str,
    user_text: str,
    max_steps: int = 8,
    require_at_least_one_call: bool = False,
    max_tool_calls: int = 8,
    trim_after_calls: int = 5,
    keep_full_fidelity: int = 3,
) -> List[ToolCallRecord]:
    """require_at_least_one_call is a deterministic backstop for callers
    that cannot accept a same-shot refusal (e.g. a redispatch round, where
    the model must actually gather something new): if the model's very
    first response makes no tool calls, it gets exactly one nudge and one
    more chance before the loop gives up. A prompt instruction alone only
    works if the model happens to comply on the first try -- this forces a
    second try rather than trusting that first response outright. Only a
    single retry, not a coercion loop: if the model still refuses, the
    caller gets an empty result back, same as if this flag were off.

    max_tool_calls bounds worst-case cost per round: once this many tool
    calls have been made, the loop stops -- treated the same as the model
    voluntarily stopping (whatever was gathered so far is returned, this
    never raises). Confirmed live: a single investigator round made 10
    sequential tool calls with no ceiling, each one resending the full
    accumulated message history.

    trim_after_calls / keep_full_fidelity attack that same resend cost
    directly rather than just capping the worst case: once more than
    trim_after_calls tool calls have been made, every call older than the
    most recent keep_full_fidelity has its result -- in the copy of the
    conversation resent to the model -- replaced with a short one-line
    digest instead of the full raw payload. This only changes what's
    resent inside this loop; the ToolCallRecord list returned always
    carries full-fidelity data for every call, since downstream evidence
    extraction (evidence_from_tool_calls) reads from the returned records,
    never from this function's internal message history."""
    tools_by_name = {t.name: t for t in tools}
    tool_specs = [t.spec() for t in tools]
    messages: List[LLMMessage] = [text_message("user", user_text)]
    records: List[ToolCallRecord] = []
    retried = False
    # tool_call_id -> (message_index, block_index, digest), populated as
    # each tool_result block is appended, so _trim_stale_tool_results can
    # go back and rewrite older ones once the threshold is crossed.
    result_positions: Dict[str, Tuple[int, int, str]] = {}
    trimmed: Set[str] = set()

    for step in range(max_steps):
        response = await llm.complete(system=system, messages=messages, tools=tool_specs, max_tokens=1500)
        if not response.tool_calls:
            if require_at_least_one_call and not retried and step == 0:
                retried = True
                messages.append(text_message("user", _RETRY_NUDGE))
                continue
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
            result_content = json.dumps(result.data if result.ok else {"error": result.error}, default=str)
            logger.info(
                "tool call: %s ok=%s result_chars=%d query=%s",
                result.tool_name, result.ok, len(result_content), result.query_summary,
            )
            result_blocks.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": result_content}
            )

        message_index = len(messages)
        messages.append(LLMMessage(role="user", content=result_blocks))

        step_records = records[-len(response.tool_calls):]
        for block_index, (call, record) in enumerate(zip(response.tool_calls, step_records)):
            result_positions[call.id] = (message_index, block_index, _digest_tool_result(record))

        _trim_stale_tool_results(messages, records, result_positions, trimmed, trim_after_calls, keep_full_fidelity)

        if len(records) >= max_tool_calls:
            break

    return records
