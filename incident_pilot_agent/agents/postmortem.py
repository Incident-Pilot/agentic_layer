"""Post-Mortem Agent — synthesizes a structured, blameless retrospective
from the already-CONFIRMED Hypothesis, its supporting evidence, and the
RemediationPlan already proposed for it. No tool loop: this node makes
exactly one LLM call and never executes anything, same as
RemediationPlanner.

Only ever reached from graph/build.py's edge after remediation_planner --
i.e. only for a genuinely CONFIRMED, actionable hypothesis that already has
a remediation plan. An ESCALATED incident, a REJECTED-then-replanned round,
or a CONFIRMED-but-non-actionable null finding never route here."""

from datetime import datetime, timezone
from typing import List

from ..llm.base import LLMClient, text_message
from ..models.postmortem import PostMortemActionItem, PostMortemReport
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_POSTMORTEM_GENERATED, AgentState
from .prompts import json_block, parse_json_response, system_header

_ROLE_DESCRIPTION = (
    "You are the Post-Mortem Agent for an incident response system. Given the CONFIRMED "
    "root-cause hypothesis, its supporting evidence, and the remediation plan already proposed "
    "for it in CONTEXT below, write a structured, blameless post-mortem. Focus on systemic "
    "factors -- what allowed this to happen, what let it go undetected or unmitigated longer "
    "than it should have -- never individual fault. action_items must be preventive/detection/"
    "process follow-ups aimed at reducing the chance or impact of a recurrence (e.g. better "
    "alerting, a guardrail, a runbook), distinct from the remediation plan's immediate "
    "technical fix for this specific incident.\n\n"
    "Respond with ONLY a JSON object (no markdown fences, no prose) with this shape:\n"
    '{"summary": str, "impact": str, "contributing_factors": [str, ...], '
    '"action_items": [{"description": str, "category": "prevent" | "detect" | "process", '
    '"priority": "low" | "medium" | "high"}, ...], "lessons_learned": [str, ...]}'
)


class PostMortemAgent:
    def __init__(self, llm: LLMClient, trajectory: TrajectoryLogger):
        self._llm = llm
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        context = state["incident_context"]
        hypothesis = next(h for h in state["hypotheses"] if h.hypothesis_id == state["current_hypothesis_id"])
        evidence_by_id = {e.evidence_id: e for e in state["evidence"]}
        remediation_plan = state.get("remediation_plan")

        payload = {
            "incident_id": context.incident_id,
            "title": context.title,
            "severity": context.severity,
            "detected_at": context.detected_at,
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
            "remediation_actions": [a.model_dump() for a in remediation_plan.actions] if remediation_plan else [],
        }

        system = system_header("generate-postmortem", _ROLE_DESCRIPTION)
        user_text = json_block("CONTEXT", payload)

        response = await self._llm.complete(system=system, messages=[text_message("user", user_text)], max_tokens=1200)
        parsed = parse_json_response(response.content)
        action_items: List[PostMortemActionItem] = [
            PostMortemActionItem(**item) for item in parsed.get("action_items", [])
        ]

        report = PostMortemReport(
            incident_id=context.incident_id,
            hypothesis_id=hypothesis.hypothesis_id,
            generated_at=datetime.now(timezone.utc),
            summary=parsed["summary"],
            impact=parsed.get("impact", ""),
            root_cause=hypothesis.root_cause,
            contributing_factors=parsed.get("contributing_factors", []),
            action_items=action_items,
            lessons_learned=parsed.get("lessons_learned", []),
        )

        self._trajectory.log(
            agent="postmortem",
            phase=PHASE_POSTMORTEM_GENERATED,
            reasoning_summary=(
                f"Generated post-mortem for hypothesis {hypothesis.hypothesis_id} ({hypothesis.root_cause}) "
                f"with {len(action_items)} follow-up action item(s)."
            ),
            hypothesis_id=hypothesis.hypothesis_id,
            round=state["iteration"],
            postmortem_summary=report.summary,
            postmortem_impact=report.impact,
            postmortem_contributing_factors=report.contributing_factors,
            postmortem_action_items=[a.model_dump() for a in action_items],
            postmortem_lessons_learned=report.lessons_learned,
        )

        return {
            "phase": PHASE_POSTMORTEM_GENERATED,
            "postmortem_report": report,
        }
