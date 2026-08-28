"""Reads trajectory/{incident_id}.trajectory.json files back out and
derives an investigation's current state from the TrajectoryEntry list.
Read-only: never writes, never touches TrajectoryLogger's write path.

Current state comes entirely from the last entry: its phase, hypothesis_id,
verification_verdict, and reasoning_summary. Rejected-hypothesis count is
derived by counting verifier entries with verdict REJECTED (exactly one
hypothesis is proposed and verified per round in this graph -- see
graph/build.py -- so this is equivalent to counting distinct rejected
hypothesis_ids without needing to track that separately)."""

import json
from pathlib import Path
from typing import List, Optional

from ..trajectory.logger import TrajectoryEntry
from .schemas import (
    HypothesisSummary,
    InvestigationDetail,
    InvestigationListItem,
    RemediationActionSummary,
    RemediationPlanSummary,
)


def _trajectory_path(trajectory_dir: Path, incident_id: str) -> Path:
    return trajectory_dir / f"{incident_id}.trajectory.json"


def _load_entries(trajectory_dir: Path, incident_id: str) -> Optional[List[TrajectoryEntry]]:
    path = _trajectory_path(trajectory_dir, incident_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return [TrajectoryEntry.model_validate(e) for e in payload["entries"]]


def _current_hypothesis(entries: List[TrajectoryEntry], hypothesis_id: str) -> Optional[HypothesisSummary]:
    # Both synthesizer.py and verifier.py log a full snapshot of the
    # hypothesis on every entry that carries its hypothesis_id, so the most
    # recent such entry is authoritative for "current" description/
    # confidence/supporting evidence; contradicting evidence only exists
    # once a verifier entry has run against it.
    for entry in reversed(entries):
        if entry.hypothesis_id == hypothesis_id and entry.hypothesis_description is not None:
            return HypothesisSummary(
                id=hypothesis_id,
                description=entry.hypothesis_description,
                confidence=entry.hypothesis_confidence or 0.0,
                supporting_evidence=entry.hypothesis_supporting_evidence_ids,
                contradicting_evidence=entry.hypothesis_contradicting_evidence_ids,
                causal_chain=entry.hypothesis_causal_chain,
                affected_services=entry.hypothesis_affected_services,
                actionable=entry.hypothesis_actionable if entry.hypothesis_actionable is not None else True,
            )
    return None


def _rejected_count(entries: List[TrajectoryEntry]) -> int:
    return sum(1 for e in entries if e.agent == "verifier" and e.verification_verdict == "REJECTED")


def _current_remediation_plan(entries: List[TrajectoryEntry], hypothesis_id: str) -> Optional[RemediationPlanSummary]:
    # Only agents/remediation_planner.py's own entries ever carry
    # remediation_actions -- absent (empty) on every other agent's entry,
    # so the most recent entry for this hypothesis_id that has any is
    # authoritative, same pattern as _current_hypothesis above.
    for entry in reversed(entries):
        if entry.hypothesis_id == hypothesis_id and entry.agent == "remediation_planner" and entry.remediation_actions:
            return RemediationPlanSummary(
                hypothesis_id=hypothesis_id,
                actions=[RemediationActionSummary(**a.model_dump()) for a in entry.remediation_actions],
            )
    return None


def _to_detail(incident_id: str, entries: List[TrajectoryEntry]) -> InvestigationDetail:
    last = entries[-1]
    hypothesis = _current_hypothesis(entries, last.hypothesis_id) if last.hypothesis_id else None
    remediation_plan = _current_remediation_plan(entries, last.hypothesis_id) if last.hypothesis_id else None
    return InvestigationDetail(
        incident_id=incident_id,
        phase=last.phase,
        iteration=last.round,
        hypothesis=hypothesis,
        verification_verdict=last.verification_verdict,
        rejected_hypotheses_count=_rejected_count(entries),
        updated_at=last.timestamp,
        reasoning_summary=last.reasoning_summary,
        remediation_plan=remediation_plan,
    )


def get_investigation(trajectory_dir: Path, incident_id: str) -> Optional[InvestigationDetail]:
    entries = _load_entries(trajectory_dir, incident_id)
    if not entries:
        return None
    return _to_detail(incident_id, entries)


def list_investigations(trajectory_dir: Path) -> List[InvestigationListItem]:
    items: List[InvestigationListItem] = []
    if not trajectory_dir.exists():
        return items
    for path in sorted(trajectory_dir.glob("*.trajectory.json")):
        incident_id = path.name[: -len(".trajectory.json")]
        entries = _load_entries(trajectory_dir, incident_id)
        if not entries:
            continue
        detail = _to_detail(incident_id, entries)
        items.append(
            InvestigationListItem(
                incident_id=detail.incident_id,
                phase=detail.phase,
                confidence=detail.hypothesis.confidence if detail.hypothesis else None,
                updated_at=detail.updated_at,
            )
        )
    return items
