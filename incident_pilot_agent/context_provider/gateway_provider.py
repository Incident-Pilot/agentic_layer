"""GatewayContextProvider — implements ContextProvider against the real
incident-pilot-ecommerce Gateway (the `observation-gateway` service).

Calls the Gateway's already-aggregated incident endpoints exactly once per
`get_context()` (GET /incidents/{id}, /evidence, /source-status, /timeline,
plus /topology) and maps its Evidence list onto this repo's typed
IncidentContext (models/context.py). Does not talk to Prometheus/Loki/Tempo/
Kubernetes directly, and does not re-run any of the Gateway's own
aggregation/normalization logic -- that already happened Gateway-side; this
class only consumes the result. Live, iterative telemetry queries during an
investigation still go through this repo's own PrometheusTool/LokiTool/
TempoTool (wired in cli.py), not through the Gateway.

Response shapes below are taken directly from incident-pilot-ecommerce's
services/observation-gateway/app/api/incidents.py, app/api/topology.py, and
shared/models/{incident,evidence,enums}.py, not guessed:

  GET /incidents/{id}          -> Incident fields (incident_id, title,
                                   severity, status, current_phase,
                                   created_at, updated_at, source,
                                   affected_services, affected_namespace,
                                   initial_alerts, root_cause,
                                   root_cause_confidence) plus observations
                                   (id list), evidence (summarized), topology
                                   (subgraph limited to affected_services).
  GET /incidents/{id}/evidence -> List[Evidence]: evidence_id, incident_id,
                                   type, source, timestamp, service,
                                   resource, summary, observation_id,
                                   raw_reference{query, trace_id, log_query,
                                   extra}. `summary` is the only place a
                                   metric's baseline/current/trend or a k8s
                                   event's reason/type end up -- the
                                   underlying structured fields live only on
                                   the Observation, which /evidence does not
                                   expose. Best-effort text-parsed below;
                                   see _map_metric / _map_k8s_event.
  GET /incidents/{id}/source-status -> {incident_id, source_status: [{source,
                                   status, error, observation_count}]}.
  GET /incidents/{id}/timeline -> {incident_id, timeline: [{timestamp,
                                   kind, id, source, ..., description}]}.
                                   Both "observation" and "evidence" kind
                                   entries already carry timestamp/
                                   description/source, so this lines up with
                                   TimelineEvent directly -- mapped
                                   defensively regardless, with a synthetic
                                   fallback built from evidence timestamps
                                   if a real response ever doesn't match.
  GET /topology                -> {namespace, topology: {service: [dep,
                                   ...]}}, an adjacency list, not an edge
                                   list -- translated into ServiceTopologyEdge
                                   below.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..models.context import (
    DeploymentRecord,
    IncidentContext,
    K8sEventItem,
    LogExcerpt,
    MetricSummary,
    Provenance,
    ServiceTopologyEdge,
    TimelineEvent,
    TraceExcerpt,
)
from .base import ContextProvider

logger = logging.getLogger(__name__)

# The Gateway Context Builder's default lookback window (services/
# observation-gateway/app/config/settings.py: CONTEXT_WINDOW_MINUTES,
# default 15) -- used only as a fallback to derive a metric's window_start,
# since GET /evidence exposes just the rendered summary text, not the raw
# window bounds (only GET /observations carries those, and that endpoint is
# intentionally out of scope here -- see module docstring).
_DEFAULT_METRIC_WINDOW_MINUTES = 15

# Matches summaries built by IncidentContextBuilder._collect_metrics, e.g.
# "cpu_usage_seconds for order-service: 0.42 cores -> 3.1 cores (rising)".
_METRIC_SUMMARY_RE = re.compile(
    r"^(?P<metric_name>\S+) for (?P<service>.+): "
    r"(?P<baseline>[-+0-9.eE]+)(?: (?P<unit>\S+))? -> "
    r"(?P<current>[-+0-9.eE]+)(?: \S+)? \((?P<trend>\w+)\)$"
)

# Matches summaries built by IncidentContextBuilder._collect_kubernetes'
# event branch, e.g. "BackOff on order-service-abc123: Back-off restarting
# failed container". Its pod-status branch ("Pod X status: phase=...") is
# handled separately, not by this pattern.
_K8S_EVENT_RE = re.compile(r"^(?P<reason>.+?) on (?P<object_ref>.+?): (?P<message>.*)$")

# GET /evidence never exposes the k8s Event's Normal/Warning `type` either
# (only Observation.metadata does) -- inferred from the reason/message text.
_WARNING_KEYWORDS = (
    "fail", "error", "back-off", "backoff", "kill", "unhealthy", "evict", "oom", "crash", "unavailable",
)

# Matches the short commit sha IncidentContextBuilder._deployment_summary_text
# embeds, e.g. "order-service deployed 4 minutes before this incident
# (commit a1b2c3d)".
_COMMIT_RE = re.compile(r"\(commit ([0-9a-fA-F]{4,40})\)")

_UNTRUSTED_EVIDENCE_TYPES = {"log", "trace"}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class GatewayContextProvider(ContextProvider):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 10.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._injected_client = client

    async def get_context(self, incident_id: str) -> IncidentContext:
        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(timeout=self._timeout_seconds)

        try:
            incident = await self._get_json(client, f"/incidents/{incident_id}")
            evidence = await self._get_json(client, f"/incidents/{incident_id}/evidence")
            source_status = await self._get_json(client, f"/incidents/{incident_id}/source-status")
            timeline_payload = await self._get_json(client, f"/incidents/{incident_id}/timeline")
            topology_payload = await self._get_json(client, "/topology")
        finally:
            if owns_client:
                await client.aclose()

        _log_source_status(incident_id, source_status)

        metrics_summary, log_excerpts, trace_excerpts, k8s_events, recent_deployments = _bucket_evidence(evidence)

        return IncidentContext(
            incident_id=incident["incident_id"],
            title=incident["title"],
            detected_at=incident["created_at"],
            severity=incident.get("severity", "unknown"),
            affected_services=incident.get("affected_services") or [],
            affected_namespace=incident.get("affected_namespace"),
            timeline=_build_timeline(timeline_payload, evidence),
            service_topology=_build_topology(topology_payload),
            metrics_summary=metrics_summary,
            log_excerpts=log_excerpts,
            trace_excerpts=trace_excerpts,
            k8s_events=k8s_events,
            recent_deployments=recent_deployments,
        )

    async def _get_json(self, client: httpx.AsyncClient, path: str) -> Any:
        url = f"{self._base_url}{path}"
        response = await client.get(url, headers={"Authorization": f"Bearer {self._api_key}"})
        response.raise_for_status()
        return response.json()


def _log_source_status(incident_id: str, payload: Dict[str, Any]) -> None:
    for entry in payload.get("source_status") or []:
        status = entry.get("status")
        log = logger.info if status == "available" else logger.warning
        log(
            "gateway source-status incident=%s source=%s status=%s observation_count=%s error=%s",
            incident_id,
            entry.get("source"),
            status,
            entry.get("observation_count"),
            entry.get("error"),
        )


def _build_topology(payload: Dict[str, Any]) -> List[ServiceTopologyEdge]:
    edges: List[ServiceTopologyEdge] = []
    for from_service, deps in (payload.get("topology") or {}).items():
        for to_service in deps or []:
            edges.append(ServiceTopologyEdge(from_service=from_service, to_service=to_service))
    return edges


def _build_timeline(timeline_payload: Dict[str, Any], evidence: List[Dict[str, Any]]) -> List[TimelineEvent]:
    try:
        return _map_timeline_entries(timeline_payload["timeline"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "GET /timeline response did not match the expected shape (%s); "
            "falling back to a synthetic timeline derived from evidence timestamps",
            exc,
        )
        return _synthetic_timeline(evidence)


def _map_timeline_entries(entries: List[Dict[str, Any]]) -> List[TimelineEvent]:
    events = []
    for entry in entries:
        provenance = (
            Provenance.UNTRUSTED
            if entry.get("kind") == "evidence" and entry.get("type") in _UNTRUSTED_EVIDENCE_TYPES
            else Provenance.TRUSTED
        )
        events.append(
            TimelineEvent(
                timestamp=entry["timestamp"],
                description=entry["description"],
                source=entry["source"],
                provenance=provenance,
            )
        )
    return events


def _synthetic_timeline(evidence: List[Dict[str, Any]]) -> List[TimelineEvent]:
    events = [
        TimelineEvent(
            timestamp=e["timestamp"],
            description=e.get("summary", ""),
            source=e.get("source", "unknown"),
            provenance=Provenance.UNTRUSTED if e.get("type") in _UNTRUSTED_EVIDENCE_TYPES else Provenance.TRUSTED,
        )
        for e in evidence
    ]
    events.sort(key=lambda ev: ev.timestamp)
    return events


def _bucket_evidence(
    evidence: List[Dict[str, Any]],
) -> Tuple[List[MetricSummary], List[LogExcerpt], List[TraceExcerpt], List[K8sEventItem], List[DeploymentRecord]]:
    metrics_summary: List[MetricSummary] = []
    log_excerpts: List[LogExcerpt] = []
    trace_excerpts: List[TraceExcerpt] = []
    k8s_events: List[K8sEventItem] = []
    recent_deployments: List[DeploymentRecord] = []

    for e in evidence:
        etype = e.get("type")
        if etype == "metric":
            metrics_summary.append(_map_metric(e))
        elif etype == "log":
            log_excerpts.append(_map_log(e))
        elif etype == "trace":
            trace_excerpts.append(_map_trace(e))
        elif etype == "kubernetes_event":
            k8s_events.append(_map_k8s_event(e))
        elif etype == "deployment":
            recent_deployments.append(_map_deployment(e))
        # "alert" and "security" evidence have no dedicated IncidentContext
        # field -- they flow into `timeline` instead (see _build_timeline),
        # matching how GET /timeline already includes every evidence type.

    return metrics_summary, log_excerpts, trace_excerpts, k8s_events, recent_deployments


def _map_metric(e: Dict[str, Any]) -> MetricSummary:
    summary = e.get("summary", "")
    window_end = _parse_ts(e["timestamp"])
    window_start = window_end - timedelta(minutes=_DEFAULT_METRIC_WINDOW_MINUTES)

    match = _METRIC_SUMMARY_RE.match(summary)
    if match:
        return MetricSummary(
            service=e.get("service") or match.group("service"),
            metric_name=match.group("metric_name"),
            unit=match.group("unit") or None,
            baseline=float(match.group("baseline")),
            current=float(match.group("current")),
            trend=match.group("trend"),
            window_start=window_start,
            window_end=window_end,
            provenance=Provenance.TRUSTED,
        )

    # Stopgap: GET /evidence only exposes the Context Builder's rendered
    # summary text for metric evidence, never the structured baseline/
    # current/trend (those live only in Observation.metadata). If the
    # summary text doesn't match the known format, don't drop the evidence
    # -- fall back to an unparsed-but-valid MetricSummary instead.
    logger.warning("could not parse metric evidence summary text: %r", summary)
    return MetricSummary(
        service=e.get("service") or "unknown",
        metric_name=(summary[:80] or "unknown_metric"),
        window_start=window_start,
        window_end=window_end,
        provenance=Provenance.TRUSTED,
    )


def _map_log(e: Dict[str, Any]) -> LogExcerpt:
    extra = ((e.get("raw_reference") or {}).get("extra")) or {}
    return LogExcerpt(
        timestamp=e["timestamp"],
        service=e.get("service"),
        pod=extra.get("pod") or e.get("resource"),
        level=extra.get("level"),
        message=extra.get("message") or e.get("summary") or "",
        labels={},
        provenance=Provenance.UNTRUSTED,
    )


def _map_trace(e: Dict[str, Any]) -> TraceExcerpt:
    raw_reference = e.get("raw_reference") or {}
    extra = raw_reference.get("extra") or {}
    return TraceExcerpt(
        trace_id=extra.get("trace_id") or raw_reference.get("trace_id") or "",
        span_id=extra.get("span_id") or "",
        service=e.get("service"),
        operation=extra.get("operation") or e.get("summary") or "",
        duration_ms=float(extra.get("duration_ms") or 0.0),
        status=extra.get("status") or "error",
        start_time=extra.get("start_time") or e["timestamp"],
        tags=extra.get("tags") or {},
        provenance=Provenance.UNTRUSTED,
    )


def _map_k8s_event(e: Dict[str, Any]) -> K8sEventItem:
    summary = e.get("summary", "")
    resource = e.get("resource") or ""

    if summary.startswith("Pod ") and " status: phase=" in summary:
        # IncidentContextBuilder._collect_kubernetes' pod-status branch --
        # not a discrete Event, but bucketed under the same evidence type.
        reason = "PodStatus"
        object_ref = resource
        message = summary
    else:
        match = _K8S_EVENT_RE.match(summary)
        reason = match.group("reason") if match else "unknown"
        object_ref = (match.group("object_ref") if match else None) or resource
        message = match.group("message") if match else summary

    event_type = "Warning" if any(keyword in summary.lower() for keyword in _WARNING_KEYWORDS) else "Normal"

    return K8sEventItem(
        timestamp=e["timestamp"],
        type=event_type,
        reason=reason,
        object_ref=object_ref,
        message=message,
        provenance=Provenance.TRUSTED,
    )


def _map_deployment(e: Dict[str, Any]) -> DeploymentRecord:
    summary = e.get("summary", "")
    commit_match = _COMMIT_RE.search(summary)
    return DeploymentRecord(
        service=e.get("service") or "unknown",
        deployed_at=e["timestamp"],
        commit_sha=commit_match.group(1) if commit_match else None,
        # Not present in Evidence.summary or raw_reference -- GET /evidence
        # doesn't expose it (only Observation.metadata would, out of scope).
        branch=None,
        change_summary=summary or None,
        provenance=Provenance.TRUSTED,
    )
