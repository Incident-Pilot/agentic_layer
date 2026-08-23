"""
IncidentContext — the contract this repo consumes from the (not-yet-built)
Incident Normalizer / Incident Context Builder.

This is deliberately modeled on the shape that incident-pilot-ecommerce's
IncidentContextBuilder already produces (Observation/Evidence with
provenance-bearing fields, per-service metrics/logs/traces, k8s events,
deployment history) but is NOT imported from that repo — this package must
not depend on the Gateway directly (see ContextProvider in
context_provider/base.py). It is this repo's own declaration of the
contract the real Context Builder will eventually satisfy.

Every data item carries its own `provenance`: TRUSTED for structured facts
sourced from deterministic systems (Kubernetes API, deployment metadata,
Alertmanager), UNTRUSTED for freeform content that could contain adversarial
or misleading text (raw log lines, trace attributes/tags). Agent prompts
must render untrusted fields inside a clearly delimited DATA block and never
treat their contents as instructions.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Provenance(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class TimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    description: str
    source: str
    provenance: Provenance = Provenance.TRUSTED


class ServiceTopologyEdge(BaseModel):
    """One directed edge in the service dependency graph: `from_service`
    calls/depends on `to_service`."""

    model_config = ConfigDict(extra="forbid")

    from_service: str
    to_service: str
    relationship: str = "calls"
    provenance: Provenance = Provenance.TRUSTED


class MetricSummary(BaseModel):
    """A pre-aggregated metric summary for one service over the incident
    window, as the Context Builder would produce it. This is a summary,
    not a raw series — the Application Investigation Agent can pull raw
    series via PrometheusTool if it needs finer granularity."""

    model_config = ConfigDict(extra="forbid")

    service: str
    metric_name: str
    unit: Optional[str] = None
    baseline: Optional[float] = None
    current: Optional[float] = None
    trend: Optional[str] = None  # "rising" | "falling" | "stable"
    window_start: datetime
    window_end: datetime
    provenance: Provenance = Provenance.TRUSTED


class LogExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    service: Optional[str] = None
    pod: Optional[str] = None
    level: Optional[str] = None
    message: str
    labels: Dict[str, str] = Field(default_factory=dict)
    provenance: Provenance = Provenance.UNTRUSTED


class TraceExcerpt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    service: Optional[str] = None
    operation: str
    duration_ms: float
    status: str  # "ok" | "error"
    start_time: datetime
    tags: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Provenance.UNTRUSTED


class K8sEventItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    type: str  # "Normal" | "Warning"
    reason: str
    object_ref: str
    message: str
    provenance: Provenance = Provenance.TRUSTED


class DeploymentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str
    deployed_at: datetime
    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    change_summary: Optional[str] = None
    provenance: Provenance = Provenance.TRUSTED


class IncidentContext(BaseModel):
    """The eventual output of the real Context Builder. Fetched only
    through ContextProvider.get_context() — never constructed ad hoc by
    agent code."""

    model_config = ConfigDict(extra="forbid")

    incident_id: str
    title: str
    description: Optional[str] = None
    detected_at: datetime
    severity: str = "unknown"  # "critical" | "warning" | "info" | "unknown"

    affected_services: List[str] = Field(default_factory=list)
    affected_namespace: Optional[str] = None

    timeline: List[TimelineEvent] = Field(default_factory=list)
    service_topology: List[ServiceTopologyEdge] = Field(default_factory=list)
    metrics_summary: List[MetricSummary] = Field(default_factory=list)
    log_excerpts: List[LogExcerpt] = Field(default_factory=list)
    trace_excerpts: List[TraceExcerpt] = Field(default_factory=list)
    k8s_events: List[K8sEventItem] = Field(default_factory=list)
    recent_deployments: List[DeploymentRecord] = Field(default_factory=list)
