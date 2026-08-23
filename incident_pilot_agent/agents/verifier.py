"""Verification Agent — independently tests the top hypothesis. Actively
looks for counter-evidence by querying for information the investigation
agent did not already fetch (a targeted query derived from any deployment
that plausibly explains the incident but isn't yet cited by the
hypothesis), rather than re-reading existing evidence and rubber-stamping
it. Outputs CONFIRMED or REJECTED with reasoning.

Finding "which deployment isn't cited yet" and "what keywords to probe
for" is mechanical text matching, done deterministically in Python; the
actual judgment call -- does what was found actually contradict the
hypothesis -- is delegated to the LLM via the verify-decide call.
"""

import uuid
from datetime import timedelta
from typing import List, Optional

from ..llm.base import LLMClient, text_message
from ..models.context import DeploymentRecord, IncidentContext
from ..models.hypothesis import Hypothesis, HypothesisStatus
from ..models.verification import Verification, VerificationVerdict
from ..tools.base import Tool
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_ROOT_CAUSE_CONFIRMED, PHASE_VERIFICATION_FAILED, PHASE_VERIFYING, AgentState
from .evidence_extraction import evidence_from_tool_calls
from .prompts import json_block, parse_json_response, system_header
from .tool_loop import run_tool_loop

_STOPWORDS = {
    "reduced", "changed", "updated", "config", "connection", "increased",
    "default", "values", "value", "from", "into", "with", "settings",
}


def _extract_keywords(text: str, max_keywords: int = 3) -> List[str]:
    words = [w.strip(",.():;\"'").lower() for w in (text or "").split()]
    candidates = [w for w in words if len(w) > 4 and w not in _STOPWORDS]
    seen: List[str] = []
    for w in candidates:
        if w not in seen:
            seen.append(w)
    return seen[:max_keywords] or ["error"]


def _change_summary_keywords(dep: DeploymentRecord, max_keywords: int = 3) -> List[str]:
    """Keywords distinctive to *what changed*, excluding the deployment's
    own service name -- a change_summary like "redis pool size reduced in
    checkout-service config" would otherwise yield "checkout-service" as a
    keyword, which trivially always appears in the hypothesis text (every
    hypothesis mentions its affected service) and would make every
    deployment look "already cited" regardless of whether the change
    itself was ever considered."""
    marker = dep.change_summary or dep.commit_sha or ""
    service_tokens = set(dep.service.lower().replace("-", " ").replace("_", " ").split()) | {dep.service.lower()}
    candidates = _extract_keywords(marker, max_keywords=max_keywords + len(service_tokens) + 2)
    keywords = [kw for kw in candidates if kw not in service_tokens]
    return keywords[:max_keywords] or ["error"]


def _find_uncited_deployment(hypothesis: Hypothesis, context: IncidentContext) -> Optional[DeploymentRecord]:
    services = set(hypothesis.affected_services) or set(context.affected_services)
    haystack = (hypothesis.root_cause + " " + " ".join(hypothesis.causal_chain)).lower()

    for dep in context.recent_deployments:
        if dep.service not in services:
            continue
        if dep.deployed_at >= context.detected_at:
            continue
        if any(kw in haystack for kw in _change_summary_keywords(dep)):
            continue  # already cited by the hypothesis
        return dep
    return None


_ROLE_DESCRIPTION = (
    "You are the Verification Agent for an incident response system. Your job is to "
    "falsify the current hypothesis, not confirm it by default. CONTEXT below includes "
    "the hypothesis under test and evidence gathered from a NEW, targeted query the "
    "investigation agent had not already run. Decide CONFIRMED only if this new evidence "
    "is consistent with (or silent on) the hypothesis; decide REJECTED if it points to an "
    "uncited deployment/config change as a more likely, earlier root cause.\n\n"
    "Respond with ONLY a JSON object (no markdown fences, no prose) with this shape:\n"
    '{"verdict": "CONFIRMED" or "REJECTED", "reasoning_summary": str, "counter_evidence_ids": [str, ...]}'
)


class VerificationAgent:
    def __init__(self, llm: LLMClient, tools: List[Tool], trajectory: TrajectoryLogger):
        self._llm = llm
        self._tools = tools
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        context = state["incident_context"]
        round_num = state["iteration"]
        hypothesis = next(h for h in state["hypotheses"] if h.hypothesis_id == state["current_hypothesis_id"])

        uncited_deployment = _find_uncited_deployment(hypothesis, context)
        keywords = _change_summary_keywords(uncited_deployment) if uncited_deployment else []

        window_start = context.detected_at - timedelta(minutes=20)
        window_end = context.detected_at + timedelta(minutes=5)

        gather_ctx = {
            "namespace": context.affected_namespace or "default",
            "affected_services": hypothesis.affected_services or context.affected_services,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "uncited_deployment": (
                {
                    "service": uncited_deployment.service,
                    "change_summary": uncited_deployment.change_summary,
                    "deployed_at": uncited_deployment.deployed_at.isoformat(),
                }
                if uncited_deployment
                else None
            ),
            "keywords": keywords,
        }
        gather_system = system_header(
            "verify-tools",
            "You are gathering independent verification evidence -- a targeted query the "
            "investigation agent has not already run, per CONTEXT below.",
        ) + "\n\n" + json_block("CONTEXT", gather_ctx)

        records = await run_tool_loop(
            self._llm,
            self._tools,
            system=gather_system,
            user_text="Gather independent verification evidence per the CONTEXT in the system prompt.",
            max_steps=4,
        )
        new_evidence = evidence_from_tool_calls(records, incident_id=context.incident_id, produced_by="verifier")

        decide_payload = {
            "hypothesis": {
                "root_cause": hypothesis.root_cause,
                "causal_chain": hypothesis.causal_chain,
                "affected_services": hypothesis.affected_services,
                "confidence": hypothesis.confidence,
            },
            "uncited_deployment": gather_ctx["uncited_deployment"],
            "newly_gathered_evidence": [
                {"evidence_id": e.evidence_id, "type": e.type.value, "summary": e.summary} for e in new_evidence
            ],
        }
        decide_system = system_header("verify-decide", _ROLE_DESCRIPTION)
        decide_user = json_block("CONTEXT", decide_payload)

        response = await self._llm.complete(system=decide_system, messages=[text_message("user", decide_user)], max_tokens=800)
        parsed = parse_json_response(response.content)
        verdict = VerificationVerdict(parsed["verdict"])

        verification = Verification(
            verification_id=f"ver-{uuid.uuid4().hex[:10]}",
            hypothesis_id=hypothesis.hypothesis_id,
            incident_id=context.incident_id,
            verdict=verdict,
            reasoning_summary=parsed.get("reasoning_summary", ""),
            new_evidence_ids=[e.evidence_id for e in new_evidence],
            counter_evidence_ids=parsed.get("counter_evidence_ids", []),
            timestamp=context.detected_at,
            round=round_num,
        )

        self._trajectory.log(
            agent="verifier",
            phase=PHASE_VERIFYING,
            reasoning_summary=f"Round {round_num}: {verdict.value} -- {verification.reasoning_summary}",
            tool_calls=[r.model_dump() for r in records],
            evidence_ids=[e.evidence_id for e in new_evidence],
            hypothesis_id=hypothesis.hypothesis_id,
            verification_verdict=verdict.value,
            round=round_num,
        )

        update: dict = {
            "evidence": new_evidence,
            "verifications": [verification],
        }
        if verdict == VerificationVerdict.CONFIRMED:
            update["phase"] = PHASE_ROOT_CAUSE_CONFIRMED
            update["final_status"] = "CONFIRMED"
        else:
            rejected_copy = hypothesis.model_copy(
                update={"status": HypothesisStatus.REJECTED, "rejection_reason": verification.reasoning_summary}
            )
            update["rejected_hypotheses"] = [rejected_copy]
            update["phase"] = PHASE_VERIFICATION_FAILED

        return update
