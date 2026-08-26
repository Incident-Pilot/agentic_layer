"""Regression tests for the investigator redispatch fixes.

Uses the real INC-FD9FA255 gateway fixture (tests/fixtures/gateway/
INC-FD9FA255/) -- same fixture and wiring as test_verifier.py. Live bug
(seen on INC-B0A77F30): every redispatch round showed "ran 0 tool
call(s)" because the investigator (1) only ever saw the bare
`root_cause` string of the last rejected hypothesis, never the
verifier's actual `rejection_reason` critique, and (2) had no prompt
distinction between round 1 (might legitimately have enough seeded
evidence) and a redispatch round (which by definition needs new
evidence, since the previous round's was just judged insufficient).
"""

import json
from pathlib import Path

import httpx
import pytest

from incident_pilot_agent.agents.investigator import ApplicationInvestigationAgent, _REDISPATCH_ADDENDUM
from incident_pilot_agent.context_provider.gateway_provider import GatewayContextProvider
from incident_pilot_agent.llm.fake_client import FakeLLMClient
from incident_pilot_agent.models.hypothesis import Hypothesis, HypothesisStatus
from incident_pilot_agent.trajectory.logger import TrajectoryLogger

GATEWAY_FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures" / "gateway"
INCIDENT_ID = "INC-FD9FA255"


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _mock_gateway_client(incident_id: str, fixtures_dir: Path) -> httpx.AsyncClient:
    """Same wiring as test_context_provider.py's / test_verifier.py's
    helper: a real httpx.AsyncClient pointed at a MockTransport serving
    the canned fixture responses instead of the network."""
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


class _RecordingLLMClient(FakeLLMClient):
    """Wraps FakeLLMClient, recording every system prompt handed to
    complete() -- lets a test assert on prompt construction (what
    CONTEXT/instructions the investigator built) without needing a real,
    network-calling LLM."""

    def __init__(self):
        super().__init__()
        self.systems = []

    async def complete(self, *, system, messages, tools=None, max_tokens=1024):
        self.systems.append(system)
        return await super().complete(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


def _make_state(context, rejected=None, evidence=None, round_num=1):
    return {
        "incident_context": context,
        "phase": "INVESTIGATING",
        "evidence": evidence or [],
        "hypotheses": [],
        "verifications": [],
        "rejected_hypotheses": rejected or [],
        "dispatch_targets": [],
        "current_hypothesis_id": None,
        "iteration": round_num,
        "max_iterations": 3,
        "final_status": None,
    }


async def test_ctx_payload_carries_rejection_reason_not_root_cause(tmp_path):
    """Gap 1 regression: the investigator must see the verifier's actual
    critique (rejection_reason), not just the bare root_cause string of
    the rejected hypothesis -- the real round-2 rejection on
    INC-B0A77F30-style incidents reads like the text below, and is far
    more actionable than "don't say this root cause again"."""
    context = await _load_incident_context()
    rejection_reason = (
        "lacks specific details linking increased response times to the order-service "
        "deployment -- no newly gathered evidence corroborates this causal link"
    )
    rejected_hyp = Hypothesis(
        hypothesis_id="hyp-test-rejected",
        incident_id=context.incident_id,
        root_cause="order-service deployment caused elevated error rates",
        causal_chain=[],
        affected_services=["order-service"],
        supporting_evidence_ids=[],
        confidence=0.5,
        status=HypothesisStatus.REJECTED,
        rejection_reason=rejection_reason,
        round=1,
    )
    state = _make_state(context, rejected=[rejected_hyp], round_num=2)

    llm = _RecordingLLMClient()
    trajectory = TrajectoryLogger(context.incident_id, tmp_path)
    agent = ApplicationInvestigationAgent(llm, tools=[], trajectory=trajectory)

    await agent(state)

    assert llm.systems, "expected at least one LLM call"
    system = llm.systems[0]
    assert rejection_reason in system
    assert rejected_hyp.root_cause not in system


async def test_ctx_payload_falls_back_to_root_cause_when_no_rejection_reason(tmp_path):
    """When rejection_reason is somehow empty, fall back to root_cause
    rather than sending None -- the investigator should still get
    *something* to avoid repeating."""
    context = await _load_incident_context()
    rejected_hyp = Hypothesis(
        hypothesis_id="hyp-test-rejected-no-reason",
        incident_id=context.incident_id,
        root_cause="order-service deployment caused elevated error rates",
        causal_chain=[],
        affected_services=["order-service"],
        supporting_evidence_ids=[],
        confidence=0.5,
        status=HypothesisStatus.REJECTED,
        rejection_reason=None,
        round=1,
    )
    state = _make_state(context, rejected=[rejected_hyp], round_num=2)

    llm = _RecordingLLMClient()
    trajectory = TrajectoryLogger(context.incident_id, tmp_path)
    agent = ApplicationInvestigationAgent(llm, tools=[], trajectory=trajectory)

    await agent(state)

    assert rejected_hyp.root_cause in llm.systems[0]


async def test_redispatch_round_prompt_requires_new_evidence(tmp_path):
    """Gap 2 regression: a redispatch round (rejected non-empty) must add
    the "gather at least one new piece of evidence" instruction that a
    first round does not get."""
    context = await _load_incident_context()
    rejected_hyp = Hypothesis(
        hypothesis_id="hyp-test-rejected",
        incident_id=context.incident_id,
        root_cause="order-service deployment caused elevated error rates",
        causal_chain=[],
        affected_services=["order-service"],
        supporting_evidence_ids=[],
        confidence=0.5,
        status=HypothesisStatus.REJECTED,
        rejection_reason="no newly gathered evidence corroborates this causal link",
        round=1,
    )
    state = _make_state(context, rejected=[rejected_hyp], round_num=2)

    llm = _RecordingLLMClient()
    trajectory = TrajectoryLogger(context.incident_id, tmp_path)
    agent = ApplicationInvestigationAgent(llm, tools=[], trajectory=trajectory)

    await agent(state)

    assert _REDISPATCH_ADDENDUM in llm.systems[0]


async def test_first_round_prompt_has_no_redispatch_addendum(tmp_path):
    """A first round (no rejected hypotheses yet) should not carry the
    redispatch instruction -- it may legitimately have enough seeded
    evidence already."""
    context = await _load_incident_context()
    state = _make_state(context, rejected=[], round_num=1)

    llm = _RecordingLLMClient()
    trajectory = TrajectoryLogger(context.incident_id, tmp_path)
    agent = ApplicationInvestigationAgent(llm, tools=[], trajectory=trajectory)

    await agent(state)

    assert llm.systems, "expected at least one LLM call"
    assert _REDISPATCH_ADDENDUM not in llm.systems[0]
    assert '"previous_rejection_reason": null' in llm.systems[0]
