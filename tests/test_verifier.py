"""Regression tests for the verifier reasoning-quality fixes.

Uses the real INC-FD9FA255 gateway fixture (tests/fixtures/gateway/
INC-FD9FA255/) -- the incident that live-exposed the bug: a hypothesis
built on essentially no direct signal for the alert it's meant to explain,
confirmed anyway because the verifier's decide step treated silence (no
new evidence, no resolved supporting evidence) as grounds for CONFIRMED.
"""

import json
from pathlib import Path

import httpx
import pytest

from incident_pilot_agent.agents.verifier import VerificationAgent, _alert_name_keywords, _find_uncited_deployment
from incident_pilot_agent.context_provider.gateway_provider import GatewayContextProvider
from incident_pilot_agent.graph.state import PHASE_VERIFICATION_FAILED
from incident_pilot_agent.llm.fake_client import FakeLLMClient
from incident_pilot_agent.models.hypothesis import Hypothesis
from incident_pilot_agent.models.verification import VerificationVerdict
from incident_pilot_agent.trajectory.logger import TrajectoryLogger

GATEWAY_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "gateway"
INCIDENT_ID = "INC-FD9FA255"


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _mock_gateway_client(incident_id: str, fixtures_dir: Path) -> httpx.AsyncClient:
    """Same wiring as test_context_provider.py's helper: a real
    httpx.AsyncClient pointed at a MockTransport serving the canned
    fixture responses instead of the network."""
    responses = {
        f"/incidents/{incident_id}": _load(fixtures_dir / "incident.json"),
        f"/incidents/{incident_id}/evidence": _load(fixtures_dir / "evidence.json"),
        f"/incidents/{incident_id}/source-status": _load(fixtures_dir / "source_status.json"),
        f"/incidents/{incident_id}/timeline": _load(fixtures_dir / "timeline.json"),
        "/topology": _load(fixtures_dir / "topology.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert path in responses, f"unexpected path requested: {path}"
        return httpx.Response(200, json=responses[path])

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _load_incident_context():
    fixtures_dir = GATEWAY_FIXTURES_ROOT / INCIDENT_ID
    client = _mock_gateway_client(INCIDENT_ID, fixtures_dir)
    provider = GatewayContextProvider(base_url="http://gateway.internal:8000", api_key="test-api-key", client=client)
    return await provider.get_context(INCIDENT_ID)


def _make_state(context, hypothesis, evidence=None):
    return {
        "incident_context": context,
        "phase": "VERIFYING",
        "evidence": evidence or [],
        "hypotheses": [hypothesis],
        "verifications": [],
        "rejected_hypotheses": [],
        "dispatch_targets": [],
        "current_hypothesis_id": hypothesis.hypothesis_id,
        "iteration": 1,
        "max_iterations": 3,
        "final_status": None,
    }


async def test_hypothesis_with_no_grounding_cannot_be_confirmed(tmp_path):
    """Gap 1 + Gap 2 regression: a hypothesis with empty
    supporting_evidence_ids, verified when the gather step turns up zero
    new evidence (forced here by giving the agent no tools to call), must
    not reach CONFIRMED. This is exactly the INC-FD9FA255 live bug: the
    old prompt treated "new evidence is silent" as sufficient for
    confirmation, and the decide step never even saw whether the
    hypothesis's original supporting evidence existed at all."""
    context = await _load_incident_context()

    hypothesis = Hypothesis(
        hypothesis_id="hyp-test-ungrounded",
        incident_id=context.incident_id,
        root_cause="order-service is returning elevated HTTP error rates due to an unspecified issue",
        causal_chain=["order-service started returning more 5xx responses"],
        affected_services=["order-service"],
        supporting_evidence_ids=[],  # nothing backs this claim
        confidence=0.4,
        round=1,
    )
    state = _make_state(context, hypothesis, evidence=[])

    trajectory = TrajectoryLogger(context.incident_id, tmp_path)
    agent = VerificationAgent(FakeLLMClient(), tools=[], trajectory=trajectory)

    result = await agent(state)

    verification = result["verifications"][0]
    assert result["evidence"] == [], "gather step should have produced no new evidence (no tools available)"
    assert verification.verdict == VerificationVerdict.REJECTED
    assert result["phase"] == PHASE_VERIFICATION_FAILED
    assert result.get("final_status") is None


async def test_alert_name_fallback_keywords_fire_when_no_uncited_deployment():
    """Gap 3 regression: INC-FD9FA255 has exactly one deployment, and once
    a hypothesis's root_cause/causal_chain already mentions it,
    _find_uncited_deployment correctly returns None -- this used to leave
    `keywords` empty and starve the gather step of anything to probe.
    The alert-name fallback must produce concrete keywords instead."""
    context = await _load_incident_context()
    assert context.title == "HighHTTPErrorRate"
    assert len(context.recent_deployments) == 1

    dep = context.recent_deployments[0]
    hypothesis = Hypothesis(
        hypothesis_id="hyp-test-cited-deploy",
        incident_id=context.incident_id,
        root_cause=f"{dep.service} was deployed 27293 minutes prior (commit {dep.commit_sha}), causing errors",
        causal_chain=[],
        affected_services=["order-service"],
        supporting_evidence_ids=[],
        confidence=0.5,
        round=1,
    )

    assert _find_uncited_deployment(hypothesis, context) is None, "the only deployment is already cited by the hypothesis"

    keywords = _alert_name_keywords(context.title)
    assert keywords, "alert-name fallback must yield concrete probe keywords, not an empty list"
    assert "error" in keywords
