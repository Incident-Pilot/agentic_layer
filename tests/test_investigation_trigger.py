"""POST /investigations/{incident_id} -- see api/app.py's route docstring.

Exercised against real fixture incidents (fixtures/incidents/) with
FakeLLMClient, the same pattern test_graph_end_to_end.py uses, so these
run a genuine graph investigation end-to-end rather than mocking it away.
No live Gateway involved -- source="auto" resolves these incident_ids to
FixtureContextProvider, which has no current_phase concept, so the
readiness check (only meaningful for GatewayContextProvider, see
create_app's docstring) is exercised separately, by monkeypatching
GatewayContextProvider itself rather than a real Gateway.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from incident_pilot_agent.api import reader
from incident_pilot_agent.api.app import create_app
from incident_pilot_agent.context_provider.gateway_provider import GatewayContextProvider

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "incidents"
API_KEY = "test-api-key"


def _client(tmp_path: Path, **overrides) -> TestClient:
    kwargs = dict(
        llm_name="fake",
        fixtures_root=FIXTURES_ROOT,
        max_iterations=4,
        state_file=tmp_path / "processed_incidents.json",
    )
    kwargs.update(overrides)
    return TestClient(create_app(tmp_path / "trajectories", API_KEY, **kwargs))


def _auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    pytest.fail(f"condition not met within {timeout}s")


def test_trigger_valid_ready_incident_returns_202_and_runs_a_real_investigation(tmp_path):
    trajectory_dir = tmp_path / "trajectories"

    # The background task runs on the TestClient's own portal event loop,
    # which only stays alive for the duration of this `with` block --
    # everything that needs the investigation to actually finish running
    # (the _wait_until below) has to happen inside it.
    with _client(tmp_path) as client:
        response = client.post("/investigations/inc-001-redis-cascade", headers=_auth())
        assert response.status_code == 202
        assert response.json() == {"incident_id": "inc-001-redis-cascade", "status": "investigation_started"}

        # reader.get_investigation reads the trajectory file fresh off
        # disk each call, so polling it doesn't need to touch the
        # portal's event loop / asyncio.Task at all.
        detail = _wait_until(lambda: reader.get_investigation(trajectory_dir, "inc-001-redis-cascade"))
        assert detail.phase in (
            "ROOT_CAUSE_CONFIRMED",
            "REMEDIATION_PROPOSED",
            "POSTMORTEM_GENERATED",
            "ESCALATED",
            "VERIFICATION_FAILED",
        )

        # And it's marked processed, so watch's own poll loop (sharing
        # this state file) wouldn't redundantly re-dispatch it.
        assert "inc-001-redis-cascade" in client.app.state.processed_incidents


def test_trigger_missing_bearer_token_returns_401(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/investigations/inc-001-redis-cascade")
        assert response.status_code == 401


def test_trigger_incident_not_ready_returns_409_and_starts_nothing(tmp_path, monkeypatch):
    trajectory_dir = tmp_path / "trajectories"

    async def _fake_get_current_phase(self, incident_id, client=None):
        return "investigating"  # anything other than ready_for_investigation

    monkeypatch.setattr(GatewayContextProvider, "get_current_phase", _fake_get_current_phase)

    # source="auto" only resolves to GatewayContextProvider when the
    # incident_id doesn't match a local fixture dir -- use one that isn't
    # a fixture, with fake Gateway env vars so _build_context_provider
    # actually constructs a GatewayContextProvider to check against.
    monkeypatch.setattr("incident_pilot_agent.pipeline.config.INCIDENT_GATEWAY_URL", "http://fake-gateway")
    monkeypatch.setattr("incident_pilot_agent.pipeline.config.INCIDENT_GATEWAY_API_KEY", "fake-key")

    with _client(tmp_path) as client:
        response = client.post("/investigations/INC-NOT-READY", headers=_auth())

        assert response.status_code == 409
        assert response.json()["detail"]["reason"] == "not_ready"
        assert not (trajectory_dir / "INC-NOT-READY.trajectory.json").exists()
        assert "INC-NOT-READY" not in client.app.state.investigations_in_progress
        assert "INC-NOT-READY" not in client.app.state.processed_incidents


def test_trigger_same_incident_twice_in_quick_succession_second_call_gets_409(tmp_path):
    trajectory_dir = tmp_path / "trajectories"

    with _client(tmp_path) as client:
        first = client.post("/investigations/inc-002-db-pool-exhaustion", headers=_auth())
        assert first.status_code == 202

        second = client.post("/investigations/inc-002-db-pool-exhaustion", headers=_auth())
        assert second.status_code == 409
        assert second.json()["detail"]["reason"] == "already_in_progress"

        # Only one investigation actually ran: the trajectory reflects a
        # single, complete run (iteration == 1 for this straightforward
        # fixture, per test_graph_end_to_end.py's own assertion on it)
        # rather than two interleaved/duplicated sets of entries.
        detail = _wait_until(lambda: reader.get_investigation(trajectory_dir, "inc-002-db-pool-exhaustion"))
        assert detail.iteration == 1


def test_in_progress_set_is_cleared_even_when_the_investigation_raises(tmp_path, monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated investigation failure")

    monkeypatch.setattr("incident_pilot_agent.api.app.run_incident", _boom)

    with _client(tmp_path) as client:
        response = client.post("/investigations/inc-003-bad-deploy-crashloop", headers=_auth())
        assert response.status_code == 202

        _wait_until(lambda: "inc-003-bad-deploy-crashloop" not in client.app.state.investigations_in_progress)

        # Cleared -- a retry is possible, not permanently "stuck" in-progress.
        retry = client.post("/investigations/inc-003-bad-deploy-crashloop", headers=_auth())
        assert retry.status_code == 202
