"""Graph-level proof that the post-mortem agent is wired into the real,
compiled LangGraph after the remediation planner -- not just exercised as
an isolated node function. Same rationale as test_remediation_planner.py:
a unit test on PostMortemAgent() in isolation wouldn't catch a broken edge.
"""

from pathlib import Path

from incident_pilot_agent.context_provider.fixture_provider import FixtureContextProvider
from incident_pilot_agent.graph.build import build_graph, finalize_status, initial_state
from incident_pilot_agent.graph.state import PHASE_ESCALATED, PHASE_POSTMORTEM_GENERATED, PHASE_ROOT_CAUSE_CONFIRMED
from incident_pilot_agent.llm.fake_client import FakeLLMClient
from incident_pilot_agent.telemetry.fixture_backends import FixtureLokiBackend, FixturePrometheusBackend, FixtureTempoBackend
from incident_pilot_agent.tools.loki_tool import LokiTool
from incident_pilot_agent.tools.prometheus_tool import PrometheusTool
from incident_pilot_agent.tools.tempo_tool import TempoTool
from incident_pilot_agent.trajectory.logger import TrajectoryLogger

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "incidents"


async def _run(incident_id: str, tmp_path: Path, max_iterations: int = 4):
    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context(incident_id)

    tools = [
        PrometheusTool(FixturePrometheusBackend(provider.incident_dir(incident_id))),
        LokiTool(FixtureLokiBackend(provider.incident_dir(incident_id))),
        TempoTool(FixtureTempoBackend(provider.incident_dir(incident_id))),
    ]
    trajectory = TrajectoryLogger(incident_id, tmp_path)
    graph = build_graph(FakeLLMClient(), tools, trajectory)

    result = await graph.ainvoke(initial_state(context, max_iterations=max_iterations))
    return finalize_status(result), trajectory


async def test_confirmed_actionable_hypothesis_reaches_postmortem_generated(tmp_path):
    """A genuinely CONFIRMED + actionable hypothesis must route all the way
    through remediation_planner into postmortem_agent, in the real compiled
    graph -- the terminal phase is POSTMORTEM_GENERATED, and the report is
    grounded in the same hypothesis and remediation plan the earlier nodes
    produced."""
    result, trajectory = await _run("inc-002-db-pool-exhaustion", tmp_path)

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_POSTMORTEM_GENERATED
    assert result["remediation_plan"] is not None

    report = result["postmortem_report"]
    assert report is not None
    assert report.incident_id == "inc-002-db-pool-exhaustion"
    assert report.hypothesis_id == result["current_hypothesis_id"]
    assert report.summary
    assert report.impact

    confirmed_hypothesis = next(h for h in result["hypotheses"] if h.hypothesis_id == result["current_hypothesis_id"])
    assert report.root_cause == confirmed_hypothesis.root_cause
    assert report.action_items
    for item in report.action_items:
        assert item.category in ("prevent", "detect", "process")
        assert item.priority in ("low", "medium", "high")

    assert any(entry.agent == "postmortem" for entry in trajectory.entries)
    postmortem_entry = next(entry for entry in trajectory.entries if entry.agent == "postmortem")
    assert postmortem_entry.postmortem_summary == report.summary
    assert postmortem_entry.postmortem_action_items


async def test_escalated_path_never_reaches_postmortem(tmp_path):
    """Mirrors test_remediation_planner.py's escalation test: a run that
    never confirms must exhaust max_iterations and end ESCALATED, with
    neither the remediation planner nor the post-mortem agent invoked."""
    from tests.test_remediation_planner import _AlwaysRejectLLMClient

    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context("inc-001-redis-cascade")
    tools = [
        PrometheusTool(FixturePrometheusBackend(provider.incident_dir("inc-001-redis-cascade"))),
        LokiTool(FixtureLokiBackend(provider.incident_dir("inc-001-redis-cascade"))),
        TempoTool(FixtureTempoBackend(provider.incident_dir("inc-001-redis-cascade"))),
    ]
    trajectory = TrajectoryLogger("inc-001-redis-cascade", tmp_path)
    graph = build_graph(_AlwaysRejectLLMClient(), tools, trajectory)
    result = await graph.ainvoke(initial_state(context, max_iterations=2))
    result = finalize_status(result)

    assert result["final_status"] == "ESCALATED"
    assert result["phase"] == PHASE_ESCALATED
    assert result["postmortem_report"] is None
    assert not any(entry.agent == "postmortem" for entry in trajectory.entries)


async def test_confirmed_non_actionable_hypothesis_skips_postmortem(tmp_path):
    """Mirrors test_remediation_planner.py's null-finding test: a CONFIRMED
    verdict on a genuine null finding (actionable=False) terminates at
    ROOT_CAUSE_CONFIRMED and never reaches the post-mortem agent, same gate
    as the remediation planner."""
    from tests.test_remediation_planner import _NullFindingLLMClient

    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context("inc-002-db-pool-exhaustion")
    tools = [
        PrometheusTool(FixturePrometheusBackend(provider.incident_dir("inc-002-db-pool-exhaustion"))),
        LokiTool(FixtureLokiBackend(provider.incident_dir("inc-002-db-pool-exhaustion"))),
        TempoTool(FixtureTempoBackend(provider.incident_dir("inc-002-db-pool-exhaustion"))),
    ]
    trajectory = TrajectoryLogger("inc-002-db-pool-exhaustion", tmp_path)
    graph = build_graph(_NullFindingLLMClient(), tools, trajectory)
    result = await graph.ainvoke(initial_state(context, max_iterations=4))
    result = finalize_status(result)

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_ROOT_CAUSE_CONFIRMED
    assert result["postmortem_report"] is None
    assert not any(entry.agent == "postmortem" for entry in trajectory.entries)
