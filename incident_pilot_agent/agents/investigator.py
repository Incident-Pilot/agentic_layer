"""Application Investigation Agent — investigates logs, traces, and metrics
for the affected services via the tool layer, producing structured
Evidence (never free text).

Baseline evidence already implied by IncidentContext (what the future
Context Builder pre-fetched) is extracted deterministically; the tool loop
is reserved for genuinely new querying beyond that starting point -- which
is also what makes the Verification Agent's "query something the
investigator didn't already fetch" requirement meaningful rather than
vacuous.
"""

from datetime import timedelta
from typing import List

from ..llm.base import LLMClient
from ..models.evidence import Evidence, EvidenceType
from ..tools.base import Tool
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_INVESTIGATING, AgentState
from .evidence_extraction import evidence_from_context, evidence_from_tool_calls
from .prompts import json_block, system_header
from .tool_loop import run_tool_loop

_ROLE_DESCRIPTION = (
    "You are the Application Investigation Agent for an incident response system. "
    "Given the CONTEXT below (affected services, time window, and metric names already "
    "known from pre-aggregated incident context), use the available read-only tools "
    "(query_prometheus, query_loki, query_tempo) to gather additional evidence about what "
    "is happening in these services. Query broadly across metrics, logs, and traces -- do "
    "not fixate on a single signal. When you have enough data, stop calling tools."
)

# Appended to _ROLE_DESCRIPTION only when rejected hypotheses exist -- i.e.
# this round is a redispatch, not the first investigation. A redispatch
# happened specifically because the previous round's evidence was judged
# insufficient by the Verification Agent, so "stop when you have enough
# data" alone (the round-1 instruction) is the wrong default here: by
# definition more should be gathered. No hard-coded minimum call count --
# that risks noise-for-noise's-sake -- just a concrete gap to go address.
_REDISPATCH_ADDENDUM = (
    "\n\nThis is a redispatch after a hypothesis was rejected. CONTEXT.previous_rejection_reason "
    "below is the Verification Agent's actual critique -- the specific evidence gap it found. You "
    "must gather at least one piece of new evidence that directly addresses that specific gap "
    "before responding. Do not conclude that the existing evidence is already sufficient: it was "
    "already judged insufficient last round, so re-reading it and stopping without querying "
    "anything new is not an acceptable outcome."
)

# Kubernetes-event evidence summaries containing any of these terms flag a
# concrete, already-known anomaly (e.g. a failed liveness probe) worth
# giving the investigator as an explicit query target, rather than only
# generic metric names -- see _notable_signals.
_ANOMALY_KEYWORDS = ("failed", "unhealthy", "restart", "crashloop", "backoff", "oomkill")


def _notable_signals(evidence: List[Evidence], max_signals: int = 5) -> List[str]:
    signals = []
    for e in evidence:
        if e.type != EvidenceType.KUBERNETES_EVENT:
            continue
        if any(kw in e.summary.lower() for kw in _ANOMALY_KEYWORDS):
            signals.append(e.summary)
    return signals[:max_signals]


class ApplicationInvestigationAgent:
    def __init__(self, llm: LLMClient, tools: List[Tool], trajectory: TrajectoryLogger):
        self._llm = llm
        self._tools = tools
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        context = state["incident_context"]
        rejected = state.get("rejected_hypotheses", [])
        round_num = state["iteration"]

        new_evidence = []
        if not state.get("evidence"):
            new_evidence.extend(evidence_from_context(context))

        window_start = context.detected_at - timedelta(minutes=20)
        window_end = context.detected_at + timedelta(minutes=5)
        metric_names = list(dict.fromkeys(m.metric_name for m in context.metrics_summary))

        previous_rejection_reason = None
        if rejected:
            previous_rejection_reason = rejected[-1].rejection_reason or rejected[-1].root_cause

        notable_signals = _notable_signals(state.get("evidence", []) + new_evidence)

        ctx_payload = {
            "namespace": context.affected_namespace or "default",
            "affected_services": context.affected_services,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "metric_names": metric_names,
            "notable_signals": notable_signals,
            "previous_rejection_reason": previous_rejection_reason,
        }
        role_description = _ROLE_DESCRIPTION + (_REDISPATCH_ADDENDUM if rejected else "")
        system = system_header("investigate-tools", role_description) + "\n\n" + json_block("CONTEXT", ctx_payload)

        is_redispatch = bool(rejected)
        records = await run_tool_loop(
            self._llm,
            self._tools,
            system=system,
            user_text="Investigate this incident using the available tools, guided by the CONTEXT in the system prompt.",
            max_steps=10,
            require_at_least_one_call=is_redispatch,
        )
        tool_evidence = evidence_from_tool_calls(records, incident_id=context.incident_id, produced_by="investigator")
        new_evidence.extend(tool_evidence)

        rejection_note = (
            f" Addressing previous rejection reason: '{previous_rejection_reason}'." if previous_rejection_reason else ""
        )
        if is_redispatch and not records:
            rejection_note += " Model did not gather new evidence even after an explicit retry."
        reasoning = (
            f"Round {round_num}: ran {len(records)} tool call(s) across metrics/logs/traces for "
            f"{context.affected_services}, producing {len(tool_evidence)} new evidence item(s) "
            f"(plus {len(new_evidence) - len(tool_evidence)} seeded from incident context)." + rejection_note
        )

        self._trajectory.log(
            agent="investigator",
            phase=PHASE_INVESTIGATING,
            reasoning_summary=reasoning,
            tool_calls=[r.model_dump() for r in records],
            evidence_ids=[e.evidence_id for e in new_evidence],
            round=round_num,
        )

        return {"evidence": new_evidence, "phase": PHASE_INVESTIGATING}
