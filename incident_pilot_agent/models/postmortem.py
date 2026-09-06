"""PostMortemReport — a structured, blameless retrospective produced by the
Post-Mortem Agent from an already-CONFIRMED, actionable Hypothesis, its
supporting evidence, and the RemediationPlan already proposed for it.

Purely descriptive: distinct in kind from RemediationPlan.actions (the
immediate technical fix for *this* incident) -- action_items here are
preventive/detection/process follow-ups aimed at reducing the chance or
impact of a recurrence, not resolving the current one. No field here ever
triggers execution of anything, same as RemediationPlan."""

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class PostMortemActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    category: Literal["prevent", "detect", "process"]
    priority: Literal["low", "medium", "high"]


class PostMortemReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    hypothesis_id: str
    generated_at: datetime

    summary: str
    impact: str
    root_cause: str
    contributing_factors: List[str] = Field(default_factory=list)
    action_items: List[PostMortemActionItem] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)
