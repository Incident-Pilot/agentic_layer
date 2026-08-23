"""AnthropicLLMClient — the only module in this repo allowed to import the
`anthropic` SDK. Everything else depends on llm.base.LLMClient."""

from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from .base import LLMClient, LLMMessage, LLMResponse, ToolCallRequest

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self._client = AsyncAnthropic(api_key=api_key)
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
            kwargs["tools"] = tools

        response = await self._client.messages.create(
            model=self._model,
            system=system,
            max_tokens=max_tokens,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )

        content_text: Optional[str] = None
        tool_calls: List[ToolCallRequest] = []
        for block in response.content:
            if block.type == "text":
                content_text = (content_text or "") + block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))

        return LLMResponse(content=content_text, tool_calls=tool_calls, stop_reason=response.stop_reason)
