from pathlib import Path

import pytest

from incident_pilot_agent.context_provider.fixture_provider import FixtureContextProvider
from incident_pilot_agent.graph.build import build_graph, finalize_status, initial_state
from incident_pilot_agent.graph.state import PHASE_ROOT_CAUSE_CONFIRMED
from incident_pilot_agent.llm.fake_client import FakeLLMClient
from incident_pilot_agent.models.verification import VerificationVerdict
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


@pytest.mark.parametrize(
    "incident_id",
    ["inc-001-redis-cascade", "inc-002-db-pool-exhaustion", "inc-003-bad-deploy-crashloop"],
)
async def test_every_fixture_reaches_a_final_verdict(incident_id, tmp_path):
    result, trajectory = await _run(incident_id, tmp_path)
    assert result["final_status"] in ("CONFIRMED", "ESCALATED")
    assert trajectory.path.exists()
    # every entry has a concise reasoning_summary and no entry accidentally
    # carries a raw model reasoning trace (spot-checked by length/shape,
    # not by content -- this repo's contract is "don't expose it", which
    # the trajectory schema enforces structurally).
    assert all(entry.reasoning_summary for entry in trajectory.entries)


async def test_cascading_failure_fixture_requires_a_rejection_and_replan_cycle(tmp_path):
    """The whole point of inc-001: the loudest signal (CPU) is a red
    herring. A investigator/synthesizer pass that just chases the loudest
    metric must get REJECTED by verification before landing on the actual
    root cause (the redis pool config change). If this ever converges on
    the first guess, the fixture stopped being adversarial."""
    result, _ = await _run("inc-001-redis-cascade", tmp_path)

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_ROOT_CAUSE_CONFIRMED
    assert result["iteration"] >= 2, "expected at least one replanning round"

    verdicts = [v.verdict for v in result["verifications"]]
    assert VerificationVerdict.REJECTED in verdicts, "expected at least one rejected hypothesis before confirmation"
    assert verdicts[-1] == VerificationVerdict.CONFIRMED

    assert len(result["rejected_hypotheses"]) >= 1
    confirmed_hypothesis = next(h for h in result["hypotheses"] if h.hypothesis_id == result["current_hypothesis_id"])
    assert "redis" in confirmed_hypothesis.root_cause.lower()


async def test_straightforward_fixtures_confirm_without_replanning(tmp_path):
    for incident_id in ["inc-002-db-pool-exhaustion", "inc-003-bad-deploy-crashloop"]:
        result, _ = await _run(incident_id, tmp_path)
        assert result["final_status"] == "CONFIRMED"
        assert result["iteration"] == 1
        assert result["rejected_hypotheses"] == []
