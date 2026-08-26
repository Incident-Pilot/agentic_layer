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
    round 1 rejected -> round 2 confirmed -- not a hand-built shape."""
    response = _client().get("/investigations/inc-001-redis-cascade", headers=_auth())
    assert response.status_code == 200

    body = response.json()
    assert body["incident_id"] == "inc-001-redis-cascade"
    assert body["phase"] == "ROOT_CAUSE_CONFIRMED"
    assert body["iteration"] == 2
    assert body["verification_verdict"] == "CONFIRMED"
    assert body["rejected_hypotheses_count"] == 1

    hypothesis = body["hypothesis"]
    assert hypothesis["id"].startswith("hyp-")
    assert "redis connection pool size reduced" in hypothesis["description"]
    assert hypothesis["confidence"] == 0.88
    assert hypothesis["supporting_evidence"]
    assert hypothesis["contradicting_evidence"] == []


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
    assert item["phase"] == "ROOT_CAUSE_CONFIRMED"
    assert item["confidence"] == 0.88
    assert item["updated_at"]


def test_list_investigations_requires_auth():
    response = _client().get("/investigations")
    assert response.status_code == 401


def test_docs_are_not_exposed_unauthenticated():
    client = _client()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
