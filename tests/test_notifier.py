"""Graph-level proof that the notifier is wired into the real, compiled
LangGraph after the remediation planner -- not just exercised as an
isolated node function. Same rationale as test_remediation_planner.py and
test_postmortem_agent.py.

tests/conftest.py's autouse fixture unsets BREVO_API_KEY/BREVO_SENDER_EMAIL/
NOTIFICATION_EMAIL_TO for every test by default. These tests explicitly
re-enable notification config (fake-looking values) and monkeypatch
BrevoEmailClient itself, so nothing here ever makes a real network call.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List

from incident_pilot_agent import config
from incident_pilot_agent.context_provider.fixture_provider import FixtureContextProvider
from incident_pilot_agent.graph.build import build_graph, finalize_status, initial_state
from incident_pilot_agent.graph.state import PHASE_POSTMORTEM_GENERATED, PHASE_ROOT_CAUSE_CONFIRMED
from incident_pilot_agent.llm.fake_client import FakeLLMClient
from incident_pilot_agent.notifications.brevo_client import BrevoAPIError
from incident_pilot_agent.telemetry.fixture_backends import FixtureLokiBackend, FixturePrometheusBackend, FixtureTempoBackend
from incident_pilot_agent.tools.loki_tool import LokiTool
from incident_pilot_agent.tools.prometheus_tool import PrometheusTool
from incident_pilot_agent.tools.tempo_tool import TempoTool
from incident_pilot_agent.trajectory.logger import TrajectoryLogger

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "incidents"


class _FakeBrevoEmailClient:
    """Records every call instead of making a real HTTP request. Class-level
    state reset by _enable_fake_notifications() at the start of each test
    that uses it -- pytest tears down monkeypatch after every test, so
    there's no real cross-test leakage risk despite the shared class attr."""

    sent: ClassVar[List[Dict[str, Any]]] = []
    should_fail: ClassVar[bool] = False

    def __init__(self, api_key, sender_email, **kwargs):
        self.api_key = api_key
        self.sender_email = sender_email

    async def send_text_email(self, *, to_email, subject, text_content):
        if _FakeBrevoEmailClient.should_fail:
            raise BrevoAPIError("simulated failure")
        _FakeBrevoEmailClient.sent.append({"to_email": to_email, "subject": subject, "text_content": text_content})
        return "fake-message-id"


def _enable_fake_notifications(monkeypatch):
    monkeypatch.setattr(config, "BREVO_API_KEY", "fake-api-key")
    monkeypatch.setattr(config, "BREVO_SENDER_EMAIL", "sender@example.com")
    monkeypatch.setattr(config, "NOTIFICATION_EMAIL_TO", "oncall@example.com")
    monkeypatch.setattr("incident_pilot_agent.agents.notifier.BrevoEmailClient", _FakeBrevoEmailClient)
    _FakeBrevoEmailClient.sent = []
    _FakeBrevoEmailClient.should_fail = False


async def _run(incident_id: str, tmp_path: Path, llm=None, max_iterations: int = 4):
    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context(incident_id)
    tools = [
        PrometheusTool(FixturePrometheusBackend(provider.incident_dir(incident_id))),
        LokiTool(FixtureLokiBackend(provider.incident_dir(incident_id))),
        TempoTool(FixtureTempoBackend(provider.incident_dir(incident_id))),
    ]
    trajectory = TrajectoryLogger(incident_id, tmp_path)
    graph = build_graph(llm or FakeLLMClient(), tools, trajectory)
    result = await graph.ainvoke(initial_state(context, max_iterations=max_iterations))
    return finalize_status(result), trajectory


async def test_confirmed_actionable_hypothesis_sends_a_notification_email(tmp_path, monkeypatch):
    _enable_fake_notifications(monkeypatch)

    result, trajectory = await _run("inc-002-db-pool-exhaustion", tmp_path)

    assert result["final_status"] == "CONFIRMED"
    assert len(_FakeBrevoEmailClient.sent) == 1
    sent = _FakeBrevoEmailClient.sent[0]
    assert sent["to_email"] == "oncall@example.com"
    assert "inc-002-db-pool-exhaustion" in sent["subject"]

    confirmed_hypothesis = next(h for h in result["hypotheses"] if h.hypothesis_id == result["current_hypothesis_id"])
    assert confirmed_hypothesis.root_cause in sent["text_content"]
    assert result["remediation_plan"].actions[0].description in sent["text_content"]
    assert result["remediation_plan"].disclaimer in sent["text_content"]

    assert any(entry.agent == "notifier" for entry in trajectory.entries)
    notifier_entry = next(entry for entry in trajectory.entries if entry.agent == "notifier")
    assert "Sent incident notification email" in notifier_entry.reasoning_summary


async def test_notification_not_configured_skips_without_sending(tmp_path):
    # No monkeypatching here -- relies on conftest.py's autouse fixture
    # leaving BREVO_API_KEY/etc unset, same as every other test in the suite.
    result, trajectory = await _run("inc-002-db-pool-exhaustion", tmp_path)

    assert result["final_status"] == "CONFIRMED"
    notifier_entry = next(entry for entry in trajectory.entries if entry.agent == "notifier")
    assert "skipped" in notifier_entry.reasoning_summary.lower()


async def test_send_failure_is_logged_but_does_not_crash_the_graph(tmp_path, monkeypatch):
    _enable_fake_notifications(monkeypatch)
    _FakeBrevoEmailClient.should_fail = True

    result, trajectory = await _run("inc-002-db-pool-exhaustion", tmp_path)

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_POSTMORTEM_GENERATED  # graph continued past the failed notification
    assert result["postmortem_report"] is not None
    notifier_entry = next(entry for entry in trajectory.entries if entry.agent == "notifier")
    assert "failed to send" in notifier_entry.reasoning_summary.lower()


async def test_escalated_path_never_sends_a_notification(tmp_path, monkeypatch):
    from tests.test_remediation_planner import _AlwaysRejectLLMClient

    _enable_fake_notifications(monkeypatch)
    result, trajectory = await _run("inc-001-redis-cascade", tmp_path, llm=_AlwaysRejectLLMClient(), max_iterations=2)

    assert result["final_status"] == "ESCALATED"
    assert _FakeBrevoEmailClient.sent == []
    assert not any(entry.agent == "notifier" for entry in trajectory.entries)


async def test_confirmed_non_actionable_hypothesis_skips_notification(tmp_path, monkeypatch):
    from tests.test_remediation_planner import _NullFindingLLMClient

    _enable_fake_notifications(monkeypatch)
    result, trajectory = await _run("inc-002-db-pool-exhaustion", tmp_path, llm=_NullFindingLLMClient())

    assert result["final_status"] == "CONFIRMED"
    assert result["phase"] == PHASE_ROOT_CAUSE_CONFIRMED
    assert _FakeBrevoEmailClient.sent == []
    assert not any(entry.agent == "notifier" for entry in trajectory.entries)
