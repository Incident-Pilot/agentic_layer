"""Verification — the independent falsification check run against the top
hypothesis. The Verification Agent must gather at least one piece of
evidence the investigation agent did not already fetch; it is not permitted
to just re-read existing evidence and rubber-stamp it (enforced by
agents/verifier.py, not by this schema, but recorded here via
new_evidence_ids so the trajectory shows it happened)."""

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class VerificationVerdict(str, Enum):
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_id: str
    hypothesis_id: str
    incident_id: str

    verdict: VerificationVerdict
    reasoning_summary: str  # concise; never the raw model reasoning trace

    new_evidence_ids: List[str] = Field(default_factory=list)
    counter_evidence_ids: List[str] = Field(default_factory=list)

    timestamp: datetime
    round: int = 1
