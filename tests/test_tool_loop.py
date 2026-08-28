"""Regression tests for run_tool_loop's require_at_least_one_call retry
backstop.

Live bug (INC-B0A77F30, INC-D07A4E58): the investigator's redispatch
prompt correctly told the model to gather new evidence, but the loop
trusted the model's very first response outright -- if that response made
no tool calls, the loop ended immediately with nothing. A prompt
instruction alone only works if the model happens to comply on the first
try. These tests exercise the deterministic one-shot retry added to
close that gap, independent of any specific agent.
"""

from typing import ClassVar, List, Optional, Type

from pydantic import BaseModel

from incident_pilot_agent.agents.prompts import json_block, system_header
from incident_pilot_agent.agents.tool_loop import run_tool_loop
from incident_pilot_agent.llm.base import LLMClient, LLMMessage, LLMResponse, ToolCallRequest
from incident_pilot_agent.llm.fake_client import FakeLLMClient
from incident_pilot_agent.tools.base import Tool, ToolResult


class _AnyInput(BaseModel):
    """Accepts whatever the fake client's deterministic plan sends
    (promql/start/end/step, logql/limit, etc.) -- pydantic v2 ignores
    unrecognized fields by default, so one permissive stub covers every
    tool name the plan might request."""


class _StubPrometheusTool(Tool):
    name: ClassVar[str] = "query_prometheus"
    description: ClassVar[str] = "stub prometheus tool for tests"
    input_schema: ClassVar[Type[BaseModel]] = _AnyInput

    async def execute(self, tool_input: BaseModel) -> ToolResult:
        return ToolResult(tool_name=self.name, ok=True, data={"series": []}, query_summary="stub prometheus query")


def _investigate_system(metric_names=("cpu_usage_seconds",), affected_services=()) -> str:
    ctx = {
        "namespace": "default",
        "affected_services": list(affected_services),
        "window_start": "2026-01-01T00:00:00",
        "window_end": "2026-01-01T00:30:00",
        "metric_names": list(metric_names),
    }
    return system_header("investigate-tools", "You are the investigator.") + "\n\n" + json_block("CONTEXT", ctx)


async def test_retry_fires_and_produces_a_real_tool_call():
    """A model that refuses on its first response but complies once
    nudged should end up with a non-empty result -- the retry must
    actually re-invoke the model with a follow-up message, not just give
    up after the first refusal."""
    llm = FakeLLMClient(refuse_tool_calls_once=True)
    records = await run_tool_loop(
        llm,
        [_StubPrometheusTool()],
        system=_investigate_system(),
        user_text="Investigate.",
        max_steps=10,
        require_at_least_one_call=True,
    )

    assert records, "retry should have produced at least one tool call record"
    assert any(r.ok for r in records), "the retried call should have actually executed a real tool"


async def test_round_one_behavior_unchanged_without_require_flag():
    """require_at_least_one_call defaults to False -- a first-round caller
    that doesn't pass it keeps today's behavior: the model's first
    response is trusted outright, no retry forced, even if that response
    made no tool calls."""
    llm = FakeLLMClient(refuse_tool_calls_once=True)
    records = await run_tool_loop(
        llm,
        [_StubPrometheusTool()],
        system=_investigate_system(),
        user_text="Investigate.",
        max_steps=10,
        # require_at_least_one_call omitted -> defaults to False
    )

    assert records == [], "without the flag, a same-shot refusal should end the loop immediately"


async def test_retry_exhausted_returns_empty_cleanly():
    """If the model still refuses after the explicit nudge, the loop must
    give up cleanly -- an empty result, not an error and not a further
    retry loop."""
    llm = FakeLLMClient(refuse_all_tool_calls=True)
    records = await run_tool_loop(
        llm,
        [_StubPrometheusTool()],
        system=_investigate_system(),
        user_text="Investigate.",
        max_steps=10,
        require_at_least_one_call=True,
    )

    assert records == []


class _InfiniteToolRequestingClient(LLMClient):
    """Always requests exactly one more tool call and never stops on its
    own -- used to prove run_tool_loop's max_tool_calls cap actually halts
    the loop even when the model keeps asking for more (a real model has
    no such ceiling on its own; the loop must enforce one), and to inspect
    the final resent message history for trimming behavior. `messages` is
    the same list object across every call (run_tool_loop only ever
    .append()s to it, never rebinds it), so capturing a reference here
    reflects the true final state once the loop finishes -- including
    trimming done after this stub's last response."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_messages: List[LLMMessage] = []

    async def complete(
        self, *, system: str, messages: List[LLMMessage], tools=None, max_tokens: int = 1024
    ) -> LLMResponse:
        self.last_messages = messages
        call_id = f"call-{self.call_count}"
        self.call_count += 1
        return LLMResponse(
            content=None,
            tool_calls=[ToolCallRequest(id=call_id, name="query_prometheus", input={})],
            stop_reason="tool_use",
        )


class _IndexTaggedPrometheusTool(Tool):
    """Returns distinguishable, non-trivial data per call so trimmed vs.
    full-fidelity tool results can be told apart by content."""

    name: ClassVar[str] = "query_prometheus"
    description: ClassVar[str] = "stub prometheus tool for tests"
    input_schema: ClassVar[Type[BaseModel]] = _AnyInput

    def __init__(self) -> None:
        self._n = 0

    async def execute(self, tool_input: BaseModel) -> ToolResult:
        self._n += 1
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={"series": [{"idx": self._n, "padding": "x" * 50}]},
            query_summary=f"call-{self._n}",
        )


async def test_max_tool_calls_caps_loop_even_when_model_keeps_requesting():
    """Confirmed live: a single investigator round made 10 sequential tool
    calls with no ceiling. max_tool_calls must stop the loop once reached,
    treated as a clean stop (whatever was gathered), not an error."""
    llm = _InfiniteToolRequestingClient()
    records = await run_tool_loop(
        llm,
        [_IndexTaggedPrometheusTool()],
        system="system prompt",
        user_text="Investigate.",
        max_steps=100,
        max_tool_calls=5,
    )

    assert len(records) == 5
    assert all(r.ok for r in records)


def _sent_tool_result_contents(messages: List[LLMMessage]) -> List[str]:
    return [
        block["content"]
        for msg in messages
        if msg.role == "user"
        for block in msg.content
        if block.get("type") == "tool_result"
    ]


async def test_older_tool_results_are_trimmed_but_returned_records_stay_full():
    """After enough tool calls, older entries in the message list actually
    resent to the LLM should be the short digest form -- while the
    returned records (what evidence_from_tool_calls and downstream callers
    consume) must still carry full-fidelity data for every call,
    regardless of whether that call's conversation entry was trimmed."""
    llm = _InfiniteToolRequestingClient()
    records = await run_tool_loop(
        llm,
        [_IndexTaggedPrometheusTool()],
        system="system prompt",
        user_text="Investigate.",
        max_steps=100,
        max_tool_calls=8,
        trim_after_calls=5,
        keep_full_fidelity=3,
    )

    assert len(records) == 8
    # Returned records are always full-fidelity, trimmed or not.
    assert [r.data["series"][0]["idx"] for r in records] == list(range(1, 9))

    contents = _sent_tool_result_contents(llm.last_messages)
    assert len(contents) == 8
    trimmed = [c for c in contents if c.startswith("[earlier]")]
    full = [c for c in contents if not c.startswith("[earlier]")]
    assert len(trimmed) == 5
    assert len(full) == 3
    # The still-full entries actually contain the raw series data.
    for c in full:
        assert "padding" in c
