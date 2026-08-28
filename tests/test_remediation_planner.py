"""Graph-level proof that the remediation planner is wired into the real,
compiled LangGraph conditional edge -- not just exercised as an isolated
node function. Each test calls the actual `build_graph(...).ainvoke(...)`
used by cli.py and asserts on the final returned state, per the lesson
from the previous (reverted) attempt: a unit test on remediation_planner()
in isolation would never have caught that the conditional edge itself
never routed there in a real run.
"""

import json
from pathlib import Path

import pytest

from incident_pilot_agent.context_provider.fixture_provider import FixtureContextProvider
from incident_pilot_agent.graph.build import build_graph, finalize_status, initial_state
from incident_pilot_agent.graph.state import PHASE_ESCALATED, PHASE_REMEDIATION_PROPOSED, PHASE_ROOT_CAUSE_CONFIRMED
from incident_pilot_agent.llm.base import LLMClient, LLMResponse
from incident_pilot_agent.llm.fake_client import FakeLLMClient, _extract_json, _last_user_text
from incident_pilot_agent.telemetry.fixture_backends import FixtureLokiBackend, FixturePrometheusBackend, FixtureTempoBackend
from incident_pilot_agent.tools.loki_tool import LokiTool
from incident_pilot_agent.tools.prometheus_tool import PrometheusTool
from incident_pilot_agent.tools.tempo_tool import TempoTool
from incident_pilot_agent.trajectory.logger import TrajectoryLogger

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "incidents"


def _task(system: str) -> str:
    return system.splitlines()[0].removeprefix("Task: ").strip()


class _AlwaysRejectLLMClient(LLMClient):
    """Test-only wrapper around FakeLLMClient that forces every
    verify-decide call to REJECTED regardless of payload content, so a run
    can be driven deterministically to ESCALATED without depending on
    FakeLLMClient's own content-sensitive verify-decide heuristic (which
    would otherwise eventually CONFIRM once a hypothesis has resolvable
    supporting evidence). Every other task delegates unchanged to a real
    FakeLLMClient, so tool-gathering and hypothesis synthesis behave
    exactly as in any other fake-backed run."""

    def __init__(self):
        self._base = FakeLLMClient()

    async def complete(self, *, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        if _task(system) == "verify-decide":
            return LLMResponse(
                content=json.dumps(
                    {"verdict": "REJECTED", "reasoning_summary": "forced rejection for test", "counter_evidence_ids": []}
                )
            )
        return await self._base.complete(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


class _NullFindingLLMClient(LLMClient):
    """Test-only wrapper mirroring a real LLM's genuine judgment on a
    null-finding incident -- exactly the live INC-4DD97821 case that
    motivated Hypothesis.actionable: synthesize-hypothesis always proposes
    a non-actionable "no anomaly" hypothesis grounded in whatever evidence
    the (real, fixture-backed) gather step actually found, and verify-decide
    confirms it (supporting evidence shows stability, nothing to
    contradict). Tool-gathering itself is untouched, delegated to a real
    FakeLLMClient."""

    def __init__(self):
        self._base = FakeLLMClient()

    async def complete(self, *, system, messages, tools=None, max_tokens=1024) -> LLMResponse:
        task = _task(system)
        if task == "synthesize-hypothesis":
            payload = _extract_json(_last_user_text(messages))
            evidence_ids = [e["evidence_id"] for e in payload.get("evidence", [])]
            return LLMResponse(
                content=json.dumps(
                    {
                        "root_cause": "Insufficient evidence - no incident detected",
                        "causal_chain": [],
                        "affected_services": [],
                        "confidence": 0.1,
                        "supporting_evidence_ids": evidence_ids,
                        "actionable": False,
                    }
                )
            )
        if task == "verify-decide":
            return LLMResponse(
                content=json.dumps(
                    {
                        "verdict": "CONFIRMED",
                        "reasoning_summary": "Supporting evidence shows stability; no anomaly found; null finding confirmed.",
                        "counter_evidence_ids": [],
                    }
                )
            )
        return await self._base.complete(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


async def _run(incident_id: str, llm: LLMClient, tmp_path: Path, max_iterations: int = 4):
    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context(incident_id)

    tools = [
        PrometheusTool(FixturePrometheusBackend(provider.incident_dir(incident_id))),
        LokiTool(FixtureLokiBackend(provider.incident_dir(incident_id))),
        TempoTool(FixtureTempoBackend(provider.incident_dir(incident_id))),
    ]
    trajectory = TrajectoryLogger(incident_id, tmp_path)
    graph = build_graph(llm, tools, trajectory)

    result = await graph.ainvoke(initial_state(context, max_iterations=max_iterations))
    return finalize_status(result), trajectory


async def test_confirmed_actionable_hypothesis_reaches_remediation_proposed(tmp_path):
    """The real compiled graph, invoked exactly as cli.py does, must route
    a genuinely CONFIRMED + actionable hypothesis through the remediation
    planner node -- not just a direct call to RemediationPlanner()."""
    result, trajectory = await _run("inc-002-db-pool-exhaustion", FakeLLMClient(), tmp_path)

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_REMEDIATION_PROPOSED
    assert result["remediation_plan"] is not None
    assert result["remediation_plan"].actions

    confirmed_hypothesis = next(h for h in result["hypotheses"] if h.hypothesis_id == result["current_hypothesis_id"])
    assert confirmed_hypothesis.actionable is True

    assert any(entry.agent == "remediation_planner" for entry in trajectory.entries)


async def test_escalated_path_never_reaches_remediation(tmp_path):
    """A run that never confirms (forced REJECTED every round) must
    exhaust max_iterations and end ESCALATED, with the remediation node
    never invoked -- proving the new conditional-edge branch is additive
    and doesn't leak into the untouched escalation path."""
    result, trajectory = await _run("inc-001-redis-cascade", _AlwaysRejectLLMClient(), tmp_path, max_iterations=2)

    assert result["final_status"] == "ESCALATED"
    assert result["phase"] == PHASE_ESCALATED
    assert result["remediation_plan"] is None
    assert not any(entry.agent == "remediation_planner" for entry in trajectory.entries)


async def test_confirmed_non_actionable_hypothesis_skips_remediation(tmp_path):
    """Mirrors the real INC-4DD97821 result: a CONFIRMED verdict on a
    genuine null finding (actionable=False) must terminate at
    ROOT_CAUSE_CONFIRMED -- a correct terminal state, not a case needing
    special handling -- and must never reach the remediation planner."""
    result, trajectory = await _run("inc-002-db-pool-exhaustion", _NullFindingLLMClient(), tmp_path)

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_ROOT_CAUSE_CONFIRMED
    assert result["remediation_plan"] is None

    confirmed_hypothesis = next(h for h in result["hypotheses"] if h.hypothesis_id == result["current_hypothesis_id"])
    assert confirmed_hypothesis.actionable is False

    assert not any(entry.agent == "remediation_planner" for entry in trajectory.entries)
