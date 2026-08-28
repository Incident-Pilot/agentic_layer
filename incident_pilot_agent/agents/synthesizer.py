"""Hypothesis Synthesizer — consumes accumulated evidence and produces a
ranked Hypothesis: root cause, causal chain, supporting/contradicting
evidence refs, confidence, affected services. Excludes any root_cause
text already rejected in a prior round so replanning actually explores a
different angle instead of looping on the same guess.
"""

import uuid
from typing import List

from ..llm.base import LLMClient, text_message
from ..models.evidence import Evidence
from ..models.hypothesis import Hypothesis
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_HYPOTHESIS_GENERATED, AgentState
from .prompts import json_block, parse_json_response, system_header

_ROLE_DESCRIPTION = (
    "You are the Hypothesis Synthesizer for an incident response system. Given the "
    "accumulated evidence, recent deployments, and Kubernetes events in CONTEXT below, "
    "propose the single most likely root cause. Do not just point at the loudest metric -- "
    "check whether a more recent deployment or configuration change plausibly explains it "
    "instead of being a downstream symptom. Never repeat a root_cause already listed in "
    "rejected_hypotheses; if the evidence still points that way, refine it or pick the next "
    "most likely candidate.\n\n"
    "Also judge explicitly whether this represents a real problem to remediate: set "
    "\"actionable\": true when root_cause identifies a genuine fault, and \"actionable\": "
    "false when the evidence points to no real anomaly (a null finding -- e.g. metrics "
    "stable, no errors, nothing to fix) and root_cause is only recording that absence of a "
    "problem. Do not leave this to be inferred from the wording of root_cause later -- set "
    "it as its own field based on the same judgment.\n\n"
    "Respond with ONLY a JSON object (no markdown fences, no prose) with this shape:\n"
    '{"root_cause": str, "causal_chain": [str, ...], "affected_services": [str, ...], '
    '"confidence": float between 0 and 1, "supporting_evidence_ids": [str, ...], '
    '"actionable": bool}'
)


class HypothesisSynthesizer:
    def __init__(self, llm: LLMClient, trajectory: TrajectoryLogger):
        self._llm = llm
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        context = state["incident_context"]
        evidence: List[Evidence] = state["evidence"]
        rejected = state.get("rejected_hypotheses", [])
        round_num = state["iteration"]

        payload = {
            "incident_id": context.incident_id,
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "type": e.type.value,
                    "summary": e.summary,
                    "service": e.service,
                    "timestamp": e.timestamp,
                }
                for e in evidence
            ],
            "rejected_hypotheses": [h.root_cause for h in rejected],
            "recent_deployments": [
                {
                    "service": d.service,
                    "deployed_at": d.deployed_at,
                    "change_summary": d.change_summary,
                    "commit_sha": d.commit_sha,
                }
                for d in context.recent_deployments
            ],
            "k8s_events": [
                {"reason": ev.reason, "object_ref": ev.object_ref, "timestamp": ev.timestamp, "message": ev.message}
                for ev in context.k8s_events
            ],
        }

        system = system_header("synthesize-hypothesis", _ROLE_DESCRIPTION)
        user_text = json_block("CONTEXT", payload)

        response = await self._llm.complete(system=system, messages=[text_message("user", user_text)], max_tokens=1200)
        parsed = parse_json_response(response.content)

        hypothesis = Hypothesis(
            hypothesis_id=f"hyp-{uuid.uuid4().hex[:10]}",
            incident_id=context.incident_id,
            root_cause=parsed["root_cause"],
            causal_chain=parsed.get("causal_chain", []),
            affected_services=parsed.get("affected_services", []),
            supporting_evidence_ids=parsed.get("supporting_evidence_ids", []),
            confidence=float(parsed.get("confidence", 0.5)),
            actionable=bool(parsed.get("actionable", True)),
            round=round_num,
        )

        self._trajectory.log(
            agent="synthesizer",
            phase=PHASE_HYPOTHESIS_GENERATED,
            reasoning_summary=f"Round {round_num}: proposed hypothesis '{hypothesis.root_cause}' (confidence={hypothesis.confidence:.2f}).",
            hypothesis_id=hypothesis.hypothesis_id,
            round=round_num,
            hypothesis_description=hypothesis.root_cause,
            hypothesis_confidence=hypothesis.confidence,
            hypothesis_supporting_evidence_ids=hypothesis.supporting_evidence_ids,
            hypothesis_causal_chain=hypothesis.causal_chain,
            hypothesis_affected_services=hypothesis.affected_services,
            hypothesis_actionable=hypothesis.actionable,
        )

        return {
            "hypotheses": [hypothesis],
            "current_hypothesis_id": hypothesis.hypothesis_id,
            "phase": PHASE_HYPOTHESIS_GENERATED,
        }
