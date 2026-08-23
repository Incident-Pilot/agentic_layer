"""OpenAILLMClient — the only module in this repo allowed to import the
`openai` SDK. Everything else depends on llm.base.LLMClient.

Translates between this repo's Anthropic-shaped LLMMessage content blocks
(`text` / `tool_use` / `tool_result`) and OpenAI's Chat Completions message
shape (flat per-message `role`, a separate `role="tool"` message per tool
result, `tool_calls` as a list of `{id, function: {name, arguments}}`).
This is exactly the translation-at-the-boundary the LLMClient abstraction
was built for -- no agent code changes to switch providers.
"""

import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .base import LLMClient, LLMMessage, LLMResponse, ToolCallRequest

DEFAULT_MODEL = "gpt-4o"


def _to_openai_messages(system: str, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = [{"role": "system", "content": system}]

    for message in messages:
        if message.role == "assistant":
            text_parts = [b["text"] for b in message.content if b.get("type") == "text"]
            tool_use_blocks = [b for b in message.content if b.get("type") == "tool_use"]
            entry: Dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_use_blocks:
                entry["tool_calls"] = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    for b in tool_use_blocks
                ]
            result.append(entry)
        else:  # user
            for block in message.content:
                if block.get("type") == "text":
                    result.append({"role": "user", "content": block["text"]})
                elif block.get("type") == "tool_result":
                    result.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]})

    return result


def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, base_url: Optional[str] = None):
        # base_url lets this same client target any OpenAI-Chat-Completions
        # -compatible endpoint (e.g. OpenRouter at
        # https://openrouter.ai/api/v1) without a separate client module --
        # the wire format is identical, only the host and model id differ.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {}
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(system, messages),
            max_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments or "{}"))
            for tc in (choice.message.tool_calls or [])
        ]

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
        )
