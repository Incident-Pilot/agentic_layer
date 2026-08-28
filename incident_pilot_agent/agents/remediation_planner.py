"""Remediation Planner — synthesizes a structured, proposed-only remediation
plan from the already-confirmed Hypothesis and the evidence accumulated in
state. No tool loop: this node makes exactly one LLM call and never
executes anything -- no kubectl, no shell, no write call of any kind. It
produces structured text data only.

Only ever reached from graph/build.py's conditional edge when the verifier's
verdict is genuinely CONFIRMED *and* the confirmed hypothesis is
Hypothesis.actionable -- an ESCALATED incident, a REJECTED-then-replanned
round, or a CONFIRMED-but-non-actionable null finding never route here.
Execution of any proposed action requires a future, separate Safety
Gateway / human-approval workflow that does not exist yet."""

from datetime import datetime, timezone
from typing import List

from ..llm.base import LLMClient, text_message
from ..models.remediation import RemediationAction, RemediationPlan
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_REMEDIATION_PROPOSED, AgentState
from .prompts import json_block, parse_json_response, system_header

_ROLE_DESCRIPTION = (
    "You are the Remediation Planner for an incident response system. Given the CONFIRMED "
    "root-cause hypothesis and its supporting evidence in CONTEXT below, propose a concrete, "
    "structured remediation plan. You do not execute anything -- this plan is proposed only "
    "and requires explicit human approval before any action is taken. If no safe, specific "
    "action is warranted, propose a single manual_investigation_required action instead of "
    "guessing.\n\n"
    "Respond with ONLY a JSON object (no markdown fences, no prose) with this shape:\n"
    '{"actions": [{"description": str, "target": str, "action_type": '
    '"rollback_deployment" | "scale_replicas" | "restart_pod" | "config_change" | '
    '"manual_investigation_required", "risk_level": "low" | "medium" | "high", '
    '"rationale": str}, ...]}'
)


class RemediationPlanner:
    def __init__(self, llm: LLMClient, trajectory: TrajectoryLogger):
        self._llm = llm
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        context = state["incident_context"]
        hypothesis = next(h for h in state["hypotheses"] if h.hypothesis_id == state["current_hypothesis_id"])
        evidence_by_id = {e.evidence_id: e for e in state["evidence"]}

        payload = {
            "hypothesis": {
                "root_cause": hypothesis.root_cause,
                "causal_chain": hypothesis.causal_chain,
                "affected_services": hypothesis.affected_services,
                "confidence": hypothesis.confidence,
            },
            "supporting_evidence": [
                {"evidence_id": eid, "summary": evidence_by_id[eid].summary}
                for eid in hypothesis.supporting_evidence_ids
                if eid in evidence_by_id
            ],
        }

        system = system_header("propose-remediation", _ROLE_DESCRIPTION)
        user_text = json_block("CONTEXT", payload)

        response = await self._llm.complete(system=system, messages=[text_message("user", user_text)], max_tokens=1200)
        parsed = parse_json_response(response.content)
        actions: List[RemediationAction] = [RemediationAction(**a) for a in parsed.get("actions", [])]

        plan = RemediationPlan(
            incident_id=context.incident_id,
            hypothesis_id=hypothesis.hypothesis_id,
            generated_at=datetime.now(timezone.utc),
            actions=actions,
        )

        self._trajectory.log(
            agent="remediation_planner",
            phase=PHASE_REMEDIATION_PROPOSED,
            reasoning_summary=(
                f"Proposed {len(plan.actions)} remediation action(s) for hypothesis "
                f"{hypothesis.hypothesis_id} ({hypothesis.root_cause})."
            ),
            hypothesis_id=hypothesis.hypothesis_id,
            round=state["iteration"],
            hypothesis_description=hypothesis.root_cause,
            hypothesis_confidence=hypothesis.confidence,
            hypothesis_supporting_evidence_ids=hypothesis.supporting_evidence_ids,
            hypothesis_contradicting_evidence_ids=hypothesis.contradicting_evidence_ids,
            remediation_actions=[a.model_dump() for a in actions],
        )

        return {
            "phase": PHASE_REMEDIATION_PROPOSED,
            "remediation_plan": plan,
        }
