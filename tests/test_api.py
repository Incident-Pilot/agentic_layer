from pathlib import Path

from fastapi.testclient import TestClient

from incident_pilot_agent.api.app import create_app

TRAJECTORY_DIR = Path(__file__).resolve().parent / "fixtures" / "trajectories"
API_KEY = "test-api-key"


def _client() -> TestClient:
    return TestClient(create_app(TRAJECTORY_DIR, API_KEY))


def _auth() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def test_health_requires_no_auth():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_investigation_against_real_trajectory_fixture():
    """inc-001-redis-cascade.trajectory.json is a real trajectory produced
    by actually running the graph (see tests/fixtures/trajectories/README.md),
    round 1 rejected -> round 2 confirmed -> remediation proposed -- not a
    hand-built shape."""
    response = _client().get("/investigations/inc-001-redis-cascade", headers=_auth())
    assert response.status_code == 200

    body = response.json()
    assert body["incident_id"] == "inc-001-redis-cascade"
    assert body["phase"] == "REMEDIATION_PROPOSED"
    assert body["iteration"] == 2
    assert body["verification_verdict"] == "CONFIRMED"
    assert body["rejected_hypotheses_count"] == 1

    hypothesis = body["hypothesis"]
    assert hypothesis["id"].startswith("hyp-")
    assert "redis connection pool size reduced" in hypothesis["description"]
    assert hypothesis["confidence"] == 0.88
    assert hypothesis["supporting_evidence"]
    assert hypothesis["contradicting_evidence"] == []

    plan = body["remediation_plan"]
    assert plan["hypothesis_id"] == hypothesis["id"]
    assert plan["actions"]
    assert plan["actions"][0]["action_type"] == "rollback_deployment"
    assert plan["disclaimer"]


def test_get_investigation_includes_causal_chain_affected_services_actionable():
    """inc-001-redis-cascade-actionable-fields.trajectory.json is a real
    trajectory produced the same way as inc-001-redis-cascade.trajectory.json
    (see tests/fixtures/trajectories/README.md), captured after
    causal_chain/affected_services/actionable were wired into
    TrajectoryEntry -- so its synthesizer/verifier entries carry real,
    non-empty values for the three fields, not hand-built ones."""
    response = _client().get(
        "/investigations/inc-001-redis-cascade-actionable-fields", headers=_auth()
    )
    assert response.status_code == 200

    hypothesis = response.json()["hypothesis"]
    assert hypothesis["causal_chain"] == [
        "checkout-service deployed at 2026-08-22 14:02:00+00:00: redis connection pool size reduced from 50 to 5 in checkout-service config",
        "500 Internal Server Error handling POST /checkout",
        "Exception: upstream timeout while processing order total",
        "cpu_usage_seconds for checkout-service: rising (0.42 -> 3.1 cores)",
        "cpu_usage_seconds for checkout-service: rising (0.42 -> 3.1)",
    ]
    assert hypothesis["affected_services"] == ["checkout-service", "order-service"]
    assert hypothesis["actionable"] is True


def test_get_investigation_missing_incident_returns_404():
    response = _client().get("/investigations/does-not-exist", headers=_auth())
    assert response.status_code == 404
    assert response.json() == {"detail": "No investigation found for this incident"}


def test_get_investigation_missing_bearer_token_returns_401():
    response = _client().get("/investigations/inc-001-redis-cascade")
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_get_investigation_wrong_bearer_token_returns_401():
    response = _client().get(
        "/investigations/inc-001-redis-cascade", headers={"Authorization": "Bearer wrong-key"}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_list_investigations_is_a_compact_subset_of_the_detail_shape():
    response = _client().get("/investigations", headers=_auth())
    assert response.status_code == 200

    items = response.json()
    item = next(i for i in items if i["incident_id"] == "inc-001-redis-cascade")
    assert item["phase"] == "REMEDIATION_PROPOSED"
    assert item["confidence"] == 0.88
    assert item["updated_at"]


def test_list_investigations_requires_auth():
    response = _client().get("/investigations")
    assert response.status_code == 401


def test_docs_are_not_exposed_unauthenticated():
    client = _client()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
