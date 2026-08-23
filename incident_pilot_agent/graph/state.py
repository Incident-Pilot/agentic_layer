"""LangGraph state for the incident investigation graph. Fields without an
Annotated reducer are last-write-wins; evidence/hypotheses/verifications
accumulate across nodes and across replanning rounds via operator.add so
the full trajectory of a multi-round investigation is preserved in state,
not just the latest round's output."""

import operator
from typing import Annotated, List, Optional, TypedDict

from ..models.context import IncidentContext
from ..models.evidence import Evidence
from ..models.hypothesis import Hypothesis
from ..models.verification import Verification

PHASE_DETECTED = "DETECTED"
PHASE_INVESTIGATING = "INVESTIGATING"
PHASE_HYPOTHESIS_GENERATED = "HYPOTHESIS_GENERATED"
PHASE_VERIFYING = "VERIFYING"
PHASE_ROOT_CAUSE_CONFIRMED = "ROOT_CAUSE_CONFIRMED"
PHASE_VERIFICATION_FAILED = "VERIFICATION_FAILED"
PHASE_ESCALATED = "ESCALATED"


class AgentState(TypedDict):
    incident_context: IncidentContext
    phase: str

    evidence: Annotated[List[Evidence], operator.add]
    hypotheses: Annotated[List[Hypothesis], operator.add]
    verifications: Annotated[List[Verification], operator.add]
    rejected_hypotheses: Annotated[List[Hypothesis], operator.add]

    dispatch_targets: List[str]
    current_hypothesis_id: Optional[str]

    iteration: int
    max_iterations: int
    final_status: Optional[str]
