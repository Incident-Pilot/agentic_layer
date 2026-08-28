"""RemediationPlan — a structured, proposed-only set of remediation actions
produced by the Remediation Planner from a CONFIRMED, actionable Hypothesis.

Purely descriptive data: no field here ever triggers execution of anything.
Execution requires a future, separate Safety Gateway / approval workflow
that does not exist yet -- see agents/remediation_planner.py's module
docstring."""

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

REMEDIATION_DISCLAIMER = (
    "This is a proposed plan only. No action has been taken. "
    "Execution requires explicit human approval, which is not yet built."
)


class RemediationAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    target: str
    action_type: Literal[
        "rollback_deployment", "scale_replicas", "restart_pod",
        "config_change", "manual_investigation_required",
    ]
    risk_level: Literal["low", "medium", "high"]
    rationale: str


class RemediationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    hypothesis_id: str
    generated_at: datetime
    actions: List[RemediationAction]
    status: Literal["proposed"] = "proposed"
    disclaimer: str = REMEDIATION_DISCLAIMER
