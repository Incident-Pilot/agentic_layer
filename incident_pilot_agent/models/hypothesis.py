"""Hypothesis — a ranked, falsifiable root-cause candidate produced by the
Hypothesis Synthesizer, and later confirmed/rejected by the Verification
Agent. Never free text: every claim traces to evidence_ids."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    incident_id: str

    root_cause: str
    # Ordered narrative of cause -> effect steps, each grounded in evidence,
    # e.g. ["config change reduced redis pool size", "connection retries
    # spiked", "retry backoff exhausted CPU", "latency and 500s rose"].
    causal_chain: List[str] = Field(default_factory=list)

    affected_services: List[str] = Field(default_factory=list)
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)

    confidence: float = Field(ge=0.0, le=1.0)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    rejection_reason: Optional[str] = None

    round: int = 1
