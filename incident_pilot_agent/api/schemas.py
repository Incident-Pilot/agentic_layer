"""Response models for the read-only investigation API. Field names mirror
TrajectoryEntry (trajectory/logger.py) directly -- this module only shapes
that data for HTTP, it never derives anything TrajectoryEntry doesn't
already carry (see reader.py)."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class HypothesisSummary(BaseModel):
    id: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    contradicting_evidence: List[str]


class InvestigationDetail(BaseModel):
    incident_id: str
    phase: str
    iteration: int
    hypothesis: Optional[HypothesisSummary]
    verification_verdict: Optional[str]
    rejected_hypotheses_count: int
    updated_at: datetime
    reasoning_summary: str


class InvestigationListItem(BaseModel):
    """A strict subset of InvestigationDetail's fields (same names/types)
    so the dashboard's list and detail views share one parser. `confidence`
    is InvestigationDetail.hypothesis.confidence, flattened -- the list
    view doesn't need the rest of the hypothesis to render a status badge
    and a confidence figure per row."""

    incident_id: str
    phase: str
    confidence: Optional[float]
    updated_at: datetime
