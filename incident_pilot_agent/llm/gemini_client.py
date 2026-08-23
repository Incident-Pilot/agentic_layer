"""GeminiLLMClient — the only module in this repo allowed to import
`google.genai`. Everything else depends on llm.base.LLMClient.

Translates between this repo's Anthropic-shaped LLMMessage content blocks
(`text` / `tool_use` / `tool_result`) and Gemini's Content/Part structure
(role "user"/"model", `function_call` / `function_response` parts).

One structural mismatch worth calling out: Gemini correlates a
function_response to its function_call by *name*, not by an id the way
Anthropic/OpenAI do. This repo's ToolCallRequest always carries a
synthetic id (Gemini's API doesn't hand back one), so `_to_gemini_contents`
rebuilds an id -> function name map by scanning prior assistant/model
turns in the same message history it's converting -- tool_use blocks
always precede their matching tool_result block in this repo's tool_loop,
so the lookup is always populated by the time it's needed.

A second mismatch: Gemini 3 attaches a `thought_signature` to each
function_call part and rejects a later turn that replays that function
call without echoing the same signature back. ToolCallRequest.id is an
opaque str with no room for a byte string, and it's provider-agnostic
(llm/base.py, tool_loop.py) so it can't grow a Gemini-specific field.
Instead the signature is base64-encoded directly into the synthetic call
id (`_encode_call_id` / `_decode_thought_signature`) -- it rides along
for free through tool_loop's id -> tool_use_id round trip without any
other module needing to know it's there.
"""

import base64
import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .base import LLMClient, LLMMessage, LLMResponse, ToolCallRequest

DEFAULT_MODEL = "gemini-3.6-flash"

# gemini-3.6-flash mandates thinking -- thinking_budget=0 is rejected
# outright (400 INVALID_ARGUMENT), unlike older Gemini models where it
# disables thinking. MINIMAL keeps the (still-hidden, include_thoughts
# stays unset/False) thinking pass as short as the model allows. Thinking
# tokens are drawn from the same max_output_tokens budget the caller
# passed in for the visible answer, so pad the budget sent to the API by
# this reserve -- otherwise a caller's max_tokens (tuned against
# Anthropic/OpenAI, which don't have this behavior) can starve the actual
# JSON answer mid-string.
_THINKING_TOKEN_RESERVE = 1024

_SIG_SEP = "::ts::"


def _encode_call_id(index: int, thought_signature: Optional[bytes]) -> str:
    call_id = f"gemini-call-{index}"
    if thought_signature:
        call_id += _SIG_SEP + base64.urlsafe_b64encode(thought_signature).decode("ascii")
    return call_id


def _decode_thought_signature(call_id: str) -> Optional[bytes]:
    if _SIG_SEP not in call_id:
        return None
    _, encoded = call_id.split(_SIG_SEP, 1)
    return base64.urlsafe_b64decode(encoded.encode("ascii"))


def _to_gemini_contents(messages: List[LLMMessage]) -> List[types.Content]:
    contents: List[types.Content] = []
    call_id_to_name: Dict[str, str] = {}

    for message in messages:
        if message.role == "assistant":
            parts = []
            for block in message.content:
                if block.get("type") == "text":
                    parts.append(types.Part(text=block["text"]))
                elif block.get("type") == "tool_use":
                    call_id_to_name[block["id"]] = block["name"]
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(name=block["name"], args=block["input"]),
                            thought_signature=_decode_thought_signature(block["id"]),
                        )
                    )
            contents.append(types.Content(role="model", parts=parts))
        else:  # user
            parts = []
            for block in message.content:
                if block.get("type") == "text":
                    parts.append(types.Part(text=block["text"]))
                elif block.get("type") == "tool_result":
                    name = call_id_to_name.get(block["tool_use_id"], block["tool_use_id"])
                    try:
                        response_payload = json.loads(block["content"])
                    except (TypeError, ValueError):
                        response_payload = {"result": block["content"]}
                    if not isinstance(response_payload, dict):
                        response_payload = {"result": response_payload}
                    parts.append(types.Part(function_response=types.FunctionResponse(name=name, response=response_payload)))
            contents.append(types.Content(role="user", parts=parts))

    return contents


def _to_gemini_tools(tools: List[Dict[str, Any]]) -> List[types.Tool]:
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=t["input_schema"])
                for t in tools
            ]
        )
    ]


class GeminiLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(
        self,
        *,
        system: str,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens + _THINKING_TOKEN_RESERVE,
            tools=_to_gemini_tools(tools) if tools else None,
            # This repo's system prompts (prompts.py's _NO_COT_RULE) forbid
            # exposed chain-of-thought outright; include_thoughts left unset
            # (defaults False) keeps thinking out of the response entirely.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=_to_gemini_contents(messages),
            config=config,
        )

        text_parts: List[str] = []
        tool_calls: List[ToolCallRequest] = []
        candidate_parts = response.candidates[0].content.parts if response.candidates and response.candidates[0].content else []
        for i, part in enumerate(candidate_parts or []):
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                tool_calls.append(
                    ToolCallRequest(
                        id=_encode_call_id(i, part.thought_signature),
                        name=part.function_call.name,
                        input=dict(part.function_call.args or {}),
                    )
                )

        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        return LLMResponse(
            content="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            stop_reason=str(finish_reason) if finish_reason else "stop",
        )
