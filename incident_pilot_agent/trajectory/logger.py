"""TrajectoryLogger — records every node execution, tool call, and agent
decision for one incident run to a local JSON file.

This is the seed of the future `AgentRun` / `IncidentMemory` Postgres
tables (see the top-level spec, section 7): each entry is a flat,
JSON-serializable record keyed by incident_id + monotonic sequence number,
so loading this file's `entries` list into a table later is a straight
column mapping, not a schema redesign.

Never logs chain-of-thought: `reasoning_summary` fields on entries are the
node's own concise summary (a few sentences), never a raw model reasoning
trace. Tool call records store `query_summary` (what was asked) and a
truncated preview of the result, not the full raw payload.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def _truncate(value: Any, max_len: int = 500) -> Any:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    if len(text) <= max_len:
        return value
    return text[:max_len] + f"... [truncated, {len(text)} chars total]"


class ToolCallLogEntry(BaseModel):
    tool_name: str
    query_summary: str
    ok: bool
    result_preview: Any = None
    error: Optional[str] = None


class RemediationActionLogEntry(BaseModel):
    description: str
    target: str
    action_type: str
    risk_level: str
    rationale: str


class TrajectoryEntry(BaseModel):
    sequence: int
    timestamp: datetime
    incident_id: str
    agent: str
    phase: str
    reasoning_summary: str
    tool_calls: List[ToolCallLogEntry] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    hypothesis_id: Optional[str] = None
    verification_verdict: Optional[str] = None
    round: int = 1

    # Additive snapshot of the hypothesis named by hypothesis_id above, as
    # of this entry -- root_cause/confidence/supporting evidence/causal
    # chain/affected services/actionable live only on the in-memory
    # Hypothesis during a graph run and were otherwise unrecoverable from
    # the trajectory file alone (only hypothesis_id was logged). Populated
    # by synthesizer.py (description/confidence/supporting/causal_chain/
    # affected_services/actionable) and verifier.py (contradicting, from
    # counter_evidence_ids, plus the same causal_chain/affected_services/
    # actionable) so a trajectory file is sufficient, on its own, to answer
    # "what is the current hypothesis" -- see incident_pilot_agent/api/.
    hypothesis_description: Optional[str] = None
    hypothesis_confidence: Optional[float] = None
    hypothesis_supporting_evidence_ids: List[str] = Field(default_factory=list)
    hypothesis_contradicting_evidence_ids: List[str] = Field(default_factory=list)
    hypothesis_causal_chain: List[str] = Field(default_factory=list)
    hypothesis_affected_services: List[str] = Field(default_factory=list)
    hypothesis_actionable: Optional[bool] = None

    # Additive, same pattern as the hypothesis snapshot fields above: set
    # only by agents/remediation_planner.py's own entry, so a trajectory
    # file is sufficient on its own to answer "what remediation plan was
    # proposed" -- see incident_pilot_agent/api/.
    remediation_actions: List[RemediationActionLogEntry] = Field(default_factory=list)


class TrajectoryLogger:
    def __init__(self, incident_id: str, output_dir: Path):
        self._incident_id = incident_id
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._output_dir / f"{incident_id}.trajectory.json"
        self._entries: List[TrajectoryEntry] = []
        self._sequence = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entries(self) -> List[TrajectoryEntry]:
        return list(self._entries)

    def log(
        self,
        *,
        agent: str,
        phase: str,
        reasoning_summary: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        evidence_ids: Optional[List[str]] = None,
        hypothesis_id: Optional[str] = None,
        verification_verdict: Optional[str] = None,
        round: int = 1,
        hypothesis_description: Optional[str] = None,
        hypothesis_confidence: Optional[float] = None,
        hypothesis_supporting_evidence_ids: Optional[List[str]] = None,
        hypothesis_contradicting_evidence_ids: Optional[List[str]] = None,
        hypothesis_causal_chain: Optional[List[str]] = None,
        hypothesis_affected_services: Optional[List[str]] = None,
        hypothesis_actionable: Optional[bool] = None,
        remediation_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> TrajectoryEntry:
        self._sequence += 1
        entry = TrajectoryEntry(
            sequence=self._sequence,
            timestamp=datetime.now(timezone.utc),
            incident_id=self._incident_id,
            agent=agent,
            phase=phase,
            reasoning_summary=reasoning_summary,
            tool_calls=[
                ToolCallLogEntry(
                    tool_name=tc["tool_name"],
                    query_summary=tc["query_summary"],
                    ok=tc["ok"],
                    result_preview=_truncate(tc.get("data")),
                    error=tc.get("error"),
                )
                for tc in (tool_calls or [])
            ],
            evidence_ids=evidence_ids or [],
            hypothesis_id=hypothesis_id,
            verification_verdict=verification_verdict,
            round=round,
            hypothesis_description=hypothesis_description,
            hypothesis_confidence=hypothesis_confidence,
            hypothesis_supporting_evidence_ids=hypothesis_supporting_evidence_ids or [],
            hypothesis_contradicting_evidence_ids=hypothesis_contradicting_evidence_ids or [],
            hypothesis_causal_chain=hypothesis_causal_chain or [],
            hypothesis_affected_services=hypothesis_affected_services or [],
            hypothesis_actionable=hypothesis_actionable,
            remediation_actions=[RemediationActionLogEntry(**a) for a in (remediation_actions or [])],
        )
        self._entries.append(entry)
        self._flush()
        return entry

    def _flush(self) -> None:
        payload = {
            "incident_id": self._incident_id,
            "entries": [json.loads(e.model_dump_json()) for e in self._entries],
        }
        self._path.write_text(json.dumps(payload, indent=2))
