"""Verification Agent — independently tests the top hypothesis. Actively
looks for counter-evidence by querying for information the investigation
agent did not already fetch: a targeted query derived from any deployment
that plausibly explains the incident but isn't yet cited by the
hypothesis, or -- when there's no such deployment -- from the alert name
itself, so the verifier always probes the actual symptom rather than
gathering nothing. Rather than re-reading existing evidence and
rubber-stamping it, the decide step also re-checks whether the
hypothesis's own supporting evidence genuinely substantiates its claims.
Outputs CONFIRMED or REJECTED with reasoning.

Finding "which deployment isn't cited yet" and "what keywords to probe
for" is mechanical text matching, done deterministically in Python; the
actual judgment call -- does supporting evidence substantiate the specific
claim, and does what was newly found actually contradict the hypothesis
-- is delegated to the LLM via the verify-decide call.
"""

import re
import uuid
from datetime import timedelta
from typing import List, Optional, Tuple

from ..llm.base import LLMClient, text_message
from ..models.context import DeploymentRecord, IncidentContext
from ..models.hypothesis import Hypothesis, HypothesisStatus
from ..models.verification import Verification, VerificationVerdict
from ..tools.base import Tool
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_ESCALATED, PHASE_ROOT_CAUSE_CONFIRMED, PHASE_VERIFICATION_FAILED, AgentState
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


# A small, non-exhaustive mapping from common alert-name patterns to probe
# keywords -- used when _find_uncited_deployment finds nothing to build
# keywords from, so the gather step still has something concrete to check
# (the symptom the alert itself describes) instead of an empty keyword list.
_ALERT_KEYWORD_PATTERNS: List[Tuple["re.Pattern[str]", List[str]]] = [
    (re.compile(r"http|5xx|error.?rate", re.I), ["error", "5xx", "exception"]),
    (re.compile(r"latency|slow|timeout", re.I), ["timeout", "latency", "slow"]),
    (re.compile(r"memory|oom", re.I), ["oom", "memory", "killed"]),
    (re.compile(r"cpu|throttl", re.I), ["throttl", "cpu"]),
    (re.compile(r"crash|restart|backoff|loop", re.I), ["crashloop", "backoff", "restart"]),
    (re.compile(r"disk|storage", re.I), ["disk", "storage", "space"]),
]


def _alert_name_keywords(alert_name: str, max_keywords: int = 3) -> List[str]:
    """Fallback probe keywords derived from the alert name itself -- a
    second, independent candidate path alongside _change_summary_keywords,
    used when there's no uncited deployment to build keywords from. Lets
    the verifier still independently check the symptom the alert describes
    rather than gathering nothing (see module docstring / _find_uncited_deployment)."""
    for pattern, keywords in _ALERT_KEYWORD_PATTERNS:
        if pattern.search(alert_name or ""):
            return keywords[:max_keywords]
    return ["error"]


_ROLE_DESCRIPTION = (
    "You are the Verification Agent for an incident response system. Your job is to "
    "falsify the current hypothesis, not confirm it by default. CONTEXT below includes "
    "the hypothesis under test, the ORIGINAL supporting_evidence it was built on, and "
    "newly_gathered_evidence from a targeted query the investigation agent had not "
    "already run.\n\n"
    "Weigh two independent questions:\n"
    "1. Does supporting_evidence actually substantiate the SPECIFIC claims in root_cause "
    "and causal_chain -- not merely correlate with the affected service in general? "
    "Evidence that just happens to mention the same service, without speaking to the "
    "specific mechanism claimed, does not substantiate the hypothesis.\n"
    "2. Does newly_gathered_evidence fail to contradict the hypothesis, or does it instead "
    "point to uncited_deployment (or another earlier config change) as a more likely root "
    "cause?\n\n"
    "Decide CONFIRMED only if (1) supporting_evidence genuinely substantiates a specific "
    "claim in the hypothesis AND (2) newly_gathered_evidence does not contradict it. New "
    "evidence being absent, inconclusive, or merely silent is NOT by itself grounds for "
    "confirmation -- if newly_gathered_evidence is empty, judge on supporting_evidence "
    "alone, and if that is thin, generic, or only correlational, REJECT rather than default "
    "to CONFIRMED. Decide REJECTED whenever newly_gathered_evidence points to an uncited "
    "deployment/config change as a more likely, earlier root cause, or whenever "
    "supporting_evidence does not genuinely establish the specific claim being made.\n\n"
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
        evidence_by_id = {e.evidence_id: e for e in state["evidence"]}

        uncited_deployment = _find_uncited_deployment(hypothesis, context)
        if uncited_deployment:
            keywords = _change_summary_keywords(uncited_deployment)
        else:
            keywords = _alert_name_keywords(context.title)

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
            require_at_least_one_call=True,
        )
        new_evidence = evidence_from_tool_calls(records, incident_id=context.incident_id, produced_by="verifier")

        decide_payload = {
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

        # The trajectory's own `phase` field previously always recorded
        # PHASE_VERIFYING here, even on a terminal verdict -- readers of the
        # trajectory file alone (e.g. api/reader.py) had no way to tell
        # ROOT_CAUSE_CONFIRMED/ESCALATED from an in-progress verification,
        # since that distinction otherwise only exists in the in-memory
        # AgentState (see graph/build.py's finalize_status(), which is never
        # logged). Mirrors _route_after_verifier's own
        # iteration >= max_iterations check so this always agrees with how
        # the graph will actually route.
        if verdict == VerificationVerdict.CONFIRMED:
            logged_phase = PHASE_ROOT_CAUSE_CONFIRMED
        elif round_num >= state["max_iterations"]:
            logged_phase = PHASE_ESCALATED
        else:
            logged_phase = PHASE_VERIFICATION_FAILED

        self._trajectory.log(
            agent="verifier",
            phase=logged_phase,
            reasoning_summary=f"Round {round_num}: {verdict.value} -- {verification.reasoning_summary}",
            tool_calls=[r.model_dump() for r in records],
            evidence_ids=[e.evidence_id for e in new_evidence],
            hypothesis_id=hypothesis.hypothesis_id,
            verification_verdict=verdict.value,
            round=round_num,
            hypothesis_description=hypothesis.root_cause,
            hypothesis_confidence=hypothesis.confidence,
            hypothesis_supporting_evidence_ids=hypothesis.supporting_evidence_ids,
            hypothesis_contradicting_evidence_ids=verification.counter_evidence_ids,
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
