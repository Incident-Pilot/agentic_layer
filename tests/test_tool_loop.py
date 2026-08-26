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

from typing import ClassVar, Type

from pydantic import BaseModel

from incident_pilot_agent.agents.prompts import json_block, system_header
from incident_pilot_agent.agents.tool_loop import run_tool_loop
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
