import json
from pathlib import Path

import httpx
import pytest

from incident_pilot_agent.agents.evidence_extraction import evidence_from_context
from incident_pilot_agent.context_provider.fixture_provider import FixtureContextProvider
from incident_pilot_agent.context_provider.gateway_provider import GatewayContextProvider
from incident_pilot_agent.models.context import Provenance
from incident_pilot_agent.models.evidence import EvidenceType

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "incidents"
GATEWAY_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "gateway"

ALL_INCIDENTS = [
    "inc-001-redis-cascade",
    "inc-002-db-pool-exhaustion",
    "inc-003-bad-deploy-crashloop",
]


@pytest.mark.parametrize("incident_id", ALL_INCIDENTS)
async def test_fixture_context_provider_loads_valid_context(incident_id):
    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context(incident_id)
    assert context.incident_id == incident_id
    assert context.affected_services


async def test_fixture_context_provider_missing_incident_raises():
    provider = FixtureContextProvider(FIXTURES_ROOT)
    with pytest.raises(FileNotFoundError):
        await provider.get_context("does-not-exist")


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _mock_gateway_client(incident_id: str, fixtures_dir: Path) -> httpx.AsyncClient:
    """A real httpx.AsyncClient wired to a MockTransport instead of the
    network -- GatewayContextProvider's own httpx calls run unmodified
    against these canned responses, only the transport is replaced. See
    tests/fixtures/gateway/README.md for how these response bodies were
    produced (this dev sandbox has no route to the live Gateway)."""
    responses = {
        f"/incidents/{incident_id}": _load(fixtures_dir / "incident.json"),
        f"/incidents/{incident_id}/evidence": _load(fixtures_dir / "evidence.json"),
        f"/incidents/{incident_id}/source-status": _load(fixtures_dir / "source_status.json"),
        f"/incidents/{incident_id}/timeline": _load(fixtures_dir / "timeline.json"),
        "/topology": _load(fixtures_dir / "topology.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer test-api-key"
        path = request.url.path
        assert path in responses, f"unexpected path requested: {path}"
        return httpx.Response(200, json=responses[path])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_gateway_context_provider_maps_real_incident_shape():
    incident_id = "INC-FD9FA255"
    fixtures_dir = GATEWAY_FIXTURES_ROOT / incident_id
    client = _mock_gateway_client(incident_id, fixtures_dir)

    provider = GatewayContextProvider(
        base_url="http://gateway.internal:8000", api_key="test-api-key", client=client
    )
    context = await provider.get_context(incident_id)
    await client.aclose()

    assert context.incident_id == incident_id
    assert context.title == "HighHTTPErrorRate"
    assert context.severity == "critical"
    assert context.affected_services == ["order-service"]
    assert context.affected_namespace == "cloudmart-prod"

    # metric evidence -> MetricSummary, parsed from the rendered summary text
    assert len(context.metrics_summary) == 3
    assert all(m.provenance == Provenance.TRUSTED for m in context.metrics_summary)
    restarts = next(m for m in context.metrics_summary if m.metric_name == "pod_restarts")
    assert restarts.baseline == 0.0
    assert restarts.current == 0.0
    assert restarts.trend == "stable"
    assert restarts.window_start < restarts.window_end
    cpu = next(m for m in context.metrics_summary if m.metric_name == "cpu_usage_seconds")
    assert cpu.baseline == pytest.approx(0.02304)
    assert cpu.current == pytest.approx(0.02212)

    # kubernetes_event evidence -> K8sEventItem (both are pod-status entries
    # for this incident, not discrete Warning-type Events)
    assert len(context.k8s_events) == 2
    assert all(e.provenance == Provenance.TRUSTED for e in context.k8s_events)
    assert all(e.reason == "PodStatus" for e in context.k8s_events)
    assert all(e.type == "Normal" for e in context.k8s_events)
    pod_status = next(e for e in context.k8s_events if e.object_ref == "order-service-5cc5c4d4d6-4xwsh")
    assert "phase=Running" in pod_status.message

    # deployment evidence -> DeploymentRecord
    assert len(context.recent_deployments) == 1
    deployment = context.recent_deployments[0]
    assert deployment.service == "order-service"
    assert deployment.commit_sha == "137548f"
    assert deployment.provenance == Provenance.TRUSTED

    # No log/trace evidence for this incident -- expected, not a mapping
    # failure (Loki's error-pattern filter and Tempo's error-span
    # normalization can both legitimately return nothing).
    assert context.log_excerpts == []
    assert context.trace_excerpts == []

    # /timeline mapped directly (its shape matched TimelineEvent), confirmed
    # for real this time -- no fallback-triggering warning logged, and the
    # 14 entries here are 7 observation-kind + 7 evidence-kind, one pair per
    # real evidence item (including the alert, which /timeline carries even
    # though it has no IncidentContext field of its own -- see below).
    assert len(context.timeline) == 14
    assert all(event.source for event in context.timeline)

    # /topology's adjacency-list shape translated into edges
    assert any(
        edge.from_service == "order-service" and edge.to_service == "product-service"
        for edge in context.service_topology
    )

    # The real Gateway /evidence for this incident returns 7 items, but only
    # 6 are seeded as investigator evidence (3 metrics + 2 k8s events + 1
    # deployment) -- the 7th, an "alert"-type item, has no dedicated
    # IncidentContext field by design (see _bucket_evidence in
    # gateway_provider.py and the comment on evidence_from_context), since
    # the triggering alert is already captured at the incident level via
    # initial_alerts. This mirrors the real CLI run's trajectory log line
    # ("plus 6 seeded from incident context") for this exact incident.
    seeded = evidence_from_context(context)
    assert len(seeded) == 6
    assert {e.type for e in seeded} == {
        EvidenceType.METRIC,
        EvidenceType.KUBERNETES_EVENT,
        EvidenceType.DEPLOYMENT,
    }


async def test_gateway_context_provider_falls_back_to_synthetic_timeline_on_bad_shape():
    incident_id = "INC-FD9FA255"
    fixtures_dir = GATEWAY_FIXTURES_ROOT / incident_id
    responses = {
        f"/incidents/{incident_id}": _load(fixtures_dir / "incident.json"),
        f"/incidents/{incident_id}/evidence": _load(fixtures_dir / "evidence.json"),
        f"/incidents/{incident_id}/source-status": _load(fixtures_dir / "source_status.json"),
        # Deliberately the wrong shape (missing "timeline") to force the fallback.
        f"/incidents/{incident_id}/timeline": {"incident_id": incident_id},
        "/topology": _load(fixtures_dir / "topology.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GatewayContextProvider(base_url="http://gateway.internal:8000", api_key="k", client=client)
    context = await provider.get_context(incident_id)
    await client.aclose()

    # Synthetic fallback is built straight from the 7 evidence items -- this
    # is a manufactured bad-shape response to exercise the fallback path
    # itself; the real /timeline (used above) already matches TimelineEvent
    # directly and never hits this branch.
    assert len(context.timeline) == 7
    assert all(event.description for event in context.timeline)


async def test_gateway_context_provider_maps_log_and_trace_evidence_as_untrusted():
    """The recorded INC-FD9FA255 fixture has no log/trace evidence (a real,
    expected outcome for that incident) -- this test exercises those two
    mapping paths directly with synthetic evidence so they're covered too."""
    incident_id = "inc-with-logs-and-traces"
    incident_payload = {
        "incident_id": incident_id,
        "title": "Synthetic log/trace coverage incident",
        "severity": "warning",
        "created_at": "2026-08-24T09:00:00Z",
        "affected_services": ["checkout-service"],
        "affected_namespace": "cloudmart-prod",
    }
    evidence_payload = [
        {
            "evidence_id": "ev-log-1",
            "incident_id": incident_id,
            "type": "log",
            "source": "loki",
            "timestamp": "2026-08-24T09:01:00Z",
            "service": "checkout-service",
            "resource": "checkout-service-abc123",
            "summary": "connection timeout talking to redis",
            "observation_id": "obs-loki-1",
            "raw_reference": {
                "query": '{namespace="cloudmart-prod"} |~ "(?i)error|exception|timeout|fail"',
                "extra": {
                    "message": "connection timeout talking to redis",
                    "pod": "checkout-service-abc123",
                    "container": "checkout-service",
                    "level": "error",
                },
            },
        },
        {
            "evidence_id": "ev-trace-1",
            "incident_id": incident_id,
            "type": "trace",
            "source": "tempo",
            "timestamp": "2026-08-24T09:01:30Z",
            "service": "checkout-service",
            "resource": None,
            "summary": "Error span in trace abc123trace (GET /checkout)",
            "observation_id": "obs-tempo-1",
            "raw_reference": {
                "query": "trace_id=abc123trace",
                "extra": {
                    "trace_id": "abc123trace",
                    "span_id": "span1",
                    "operation": "GET /checkout",
                    "duration_ms": 4231.5,
                    "status": "error",
                    "start_time": "2026-08-24T09:01:30Z",
                    "tags": {"http.status_code": 500},
                },
            },
        },
    ]
    source_status_payload = {"incident_id": incident_id, "source_status": []}
    timeline_payload = {"incident_id": incident_id, "timeline": []}
    topology_payload = {"namespace": "cloudmart-prod", "topology": {}}

    responses = {
        f"/incidents/{incident_id}": incident_payload,
        f"/incidents/{incident_id}/evidence": evidence_payload,
        f"/incidents/{incident_id}/source-status": source_status_payload,
        f"/incidents/{incident_id}/timeline": timeline_payload,
        "/topology": topology_payload,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GatewayContextProvider(base_url="http://gateway.internal:8000", api_key="k", client=client)
    context = await provider.get_context(incident_id)
    await client.aclose()

    assert len(context.log_excerpts) == 1
    log = context.log_excerpts[0]
    assert log.provenance == Provenance.UNTRUSTED
    assert log.message == "connection timeout talking to redis"
    assert log.level == "error"
    assert log.pod == "checkout-service-abc123"

    assert len(context.trace_excerpts) == 1
    trace = context.trace_excerpts[0]
    assert trace.provenance == Provenance.UNTRUSTED
    assert trace.trace_id == "abc123trace"
    assert trace.status == "error"
    assert trace.duration_ms == 4231.5


def test_gateway_context_provider_requires_base_url_and_api_key():
    with pytest.raises(ValueError):
        GatewayContextProvider(base_url="", api_key="k")
    with pytest.raises(ValueError):
        GatewayContextProvider(base_url="http://gateway.internal:8000", api_key="")
