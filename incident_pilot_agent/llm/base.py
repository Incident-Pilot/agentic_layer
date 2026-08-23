"""LLMClient — thin provider-agnostic abstraction. Agent code (agents/*.py)
imports only from this module, never `anthropic` directly, so a different
provider can be swapped in by writing one new LLMClient implementation.

Message content is a list of provider-shaped content blocks (`{"type":
"text", "text": ...}`, `{"type": "tool_use", ...}`, `{"type":
"tool_result", ...}`) rather than a single string, because a genuine
multi-step tool-use loop needs to replay tool_use/tool_result blocks back
to the model. This shape matches Anthropic's Messages API directly; a
future OpenAI-style client would translate to/from it at its own boundary
rather than forcing this interface down to the lowest common denominator.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ToolCallRequest(BaseModel):
    id: str
    name: str
    input: Dict[str, Any]


class LLMMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: List[Dict[str, Any]]


def text_message(role: Literal["user", "assistant"], text: str) -> LLMMessage:
    return LLMMessage(role=role, content=[{"type": "text", "text": text}])


class LLMResponse(BaseModel):
    # Concise final text (expected to be JSON matching a caller-specified
    # schema once tool_calls is empty). Must never contain the model's raw
    # internal reasoning trace -- callers instruct this via system prompts.
    content: Optional[str] = None
    tool_calls: List[ToolCallRequest] = []
    stop_reason: str = "end_turn"


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...
