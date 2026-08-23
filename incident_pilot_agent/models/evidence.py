"""
Evidence — a structured, citable fact produced by an agent's tool call.

Agents never emit free-text claims about what they found; they emit
Evidence records referencing the specific tool call that produced them.
Downstream nodes (Hypothesis Synthesizer, Verification Agent) reason over
these structured records, not over an agent's prose.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class EvidenceType(str, Enum):
    METRIC = "metric"
    LOG = "log"
    TRACE = "trace"
    KUBERNETES_EVENT = "kubernetes_event"
    DEPLOYMENT = "deployment"
    TOPOLOGY = "topology"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    incident_id: str
    type: EvidenceType

    # Concise, human/agent-readable statement of the fact. Never a raw
    # payload dump and never the model's chain-of-thought — just the
    # finding, e.g. "cpu_usage_seconds for checkout-service rose from
    # 0.4 to 3.1 cores between 14:02 and 14:06".
    summary: str

    service: Optional[str] = None
    timestamp: Optional[datetime] = None

    # Which tool call (see trajectory log) produced this, and the query
    # that was issued — lets a later agent or a human re-run the exact
    # query rather than trusting the summary blindly.
    tool_call_id: Optional[str] = None
    source_query: Optional[str] = None

    produced_by: str = "unknown"  # agent name, e.g. "investigator" | "verifier"
