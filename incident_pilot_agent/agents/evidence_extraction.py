"""Deterministic reshaping of IncidentContext items and tool results into
Evidence records. No LLM call: this is exactly the kind of mechanical
parsing the top-level spec says to keep deterministic ("don't ask the LLM
to parse OTLP JSON" applies just as much to reshaping already-typed tool
output into Evidence rows). The LLM's real job is deciding which tools to
call and what the resulting evidence means -- not restructuring it.
"""

import uuid
from typing import List

from ..models.context import IncidentContext
from ..models.evidence import Evidence, EvidenceType
from .tool_loop import ToolCallRecord


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _match_service(object_ref: str, known_services: List[str]) -> str:
    """K8s event/pod object_refs are pod names (service-name + a
    generated suffix), not service names -- normalize back to the known
    service so it doesn't leak a pod-ish string into Evidence.service /
    Hypothesis.affected_services. Falls back to the raw ref if no known
    service prefixes it."""
    for service in known_services:
        if object_ref == service or object_ref.startswith(f"{service}-"):
            return service
    return object_ref


def evidence_from_context(context: IncidentContext) -> List[Evidence]:
    """Baseline evidence taken directly from the pre-aggregated
    IncidentContext the (future) Context Builder already assembled --
    no tool call needed to know these facts."""
    evidence: List[Evidence] = []

    for m in context.metrics_summary:
        evidence.append(
            Evidence(
                evidence_id=_new_id("ev-metric"),
                incident_id=context.incident_id,
                type=EvidenceType.METRIC,
                summary=f"{m.metric_name} for {m.service}: {m.trend} ({m.baseline} -> {m.current} {m.unit or ''})".strip(),
                service=m.service,
                timestamp=m.window_end,
                produced_by="context",
            )
        )
    for log in context.log_excerpts:
        evidence.append(
            Evidence(
                evidence_id=_new_id("ev-log"),
                incident_id=context.incident_id,
                type=EvidenceType.LOG,
                summary=log.message[:200],
                service=log.service,
                timestamp=log.timestamp,
                produced_by="context",
            )
        )
    for trace in context.trace_excerpts:
        evidence.append(
            Evidence(
                evidence_id=_new_id("ev-trace"),
                incident_id=context.incident_id,
                type=EvidenceType.TRACE,
                summary=f"{trace.operation} on {trace.service}: status={trace.status}, duration={trace.duration_ms}ms",
                service=trace.service,
                timestamp=trace.start_time,
                produced_by="context",
            )
        )
    for ev in context.k8s_events:
        evidence.append(
            Evidence(
                evidence_id=_new_id("ev-k8s"),
                incident_id=context.incident_id,
                type=EvidenceType.KUBERNETES_EVENT,
                summary=f"{ev.reason} on {ev.object_ref}: {ev.message}"[:250],
                service=_match_service(ev.object_ref, context.affected_services),
                timestamp=ev.timestamp,
                produced_by="context",
            )
        )
    for dep in context.recent_deployments:
        evidence.append(
            Evidence(
                evidence_id=_new_id("ev-deploy"),
                incident_id=context.incident_id,
                type=EvidenceType.DEPLOYMENT,
                summary=f"{dep.service} deployed at {dep.deployed_at.isoformat()}: {dep.change_summary or dep.commit_sha or 'no summary'}",
                service=dep.service,
                timestamp=dep.deployed_at,
                produced_by="context",
            )
        )
    return evidence


def evidence_from_tool_calls(records: List[ToolCallRecord], *, incident_id: str, produced_by: str) -> List[Evidence]:
    """Reshape tool_loop.ToolCallRecord results into Evidence: one row per
    Prometheus series / capped Loki lines / Tempo trace or error span."""
    evidence: List[Evidence] = []

    for rec in records:
        if not rec.ok or not rec.data:
            continue

        if rec.tool_name == "query_prometheus":
            for series in rec.data.get("series", []) or []:
                labels = series.get("labels", {}) or {}
                metric_name = labels.get("__name__", "metric")
                service = labels.get("service") or labels.get("pod") or labels.get("job")
                first_v, last_v = series.get("first_value", 0.0), series.get("last_value", 0.0)
                evidence.append(
                    Evidence(
                        evidence_id=_new_id("ev-metric"),
                        incident_id=incident_id,
                        type=EvidenceType.METRIC,
                        summary=f"{metric_name} for {service}: {series.get('trend')} ({first_v:.3g} -> {last_v:.3g})",
                        service=service,
                        tool_call_id=rec.tool_call_id,
                        source_query=rec.query_summary,
                        produced_by=produced_by,
                    )
                )

        elif rec.tool_name == "query_loki":
            for entry in (rec.data.get("entries", []) or [])[:10]:
                evidence.append(
                    Evidence(
                        evidence_id=_new_id("ev-log"),
                        incident_id=incident_id,
                        type=EvidenceType.LOG,
                        summary=(entry.get("message") or "")[:200],
                        service=entry.get("service"),
                        timestamp=entry.get("timestamp"),
                        tool_call_id=rec.tool_call_id,
                        source_query=rec.query_summary,
                        produced_by=produced_by,
                    )
                )

        elif rec.tool_name == "query_tempo":
            for trace in (rec.data.get("traces", []) or [])[:5]:
                evidence.append(
                    Evidence(
                        evidence_id=_new_id("ev-trace"),
                        incident_id=incident_id,
                        type=EvidenceType.TRACE,
                        summary=f"trace {trace.get('trace_id')} root={trace.get('root_service')} duration={trace.get('duration_ms')}ms",
                        service=trace.get("root_service"),
                        timestamp=trace.get("start_time"),
                        tool_call_id=rec.tool_call_id,
                        source_query=rec.query_summary,
                        produced_by=produced_by,
                    )
                )
            for span in (rec.data.get("spans", []) or []):
                if span.get("status") == "error":
                    evidence.append(
                        Evidence(
                            evidence_id=_new_id("ev-span"),
                            incident_id=incident_id,
                            type=EvidenceType.TRACE,
                            summary=f"error span {span.get('operation')} on {span.get('service')}: {str(span.get('tags', {}))[:120]}",
                            service=span.get("service"),
                            timestamp=span.get("start_time"),
                            tool_call_id=rec.tool_call_id,
                            source_query=rec.query_summary,
                            produced_by=produced_by,
                        )
                    )

    return evidence
