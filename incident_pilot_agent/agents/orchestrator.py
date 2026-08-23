"""Orchestrator — decides which specialist agent(s) to invoke for the
current round. In this phase there is exactly one specialist
(Application Investigation Agent), but dispatch is a table lookup rather
than a hardcoded call so that adding an Infrastructure/Network/Security
agent later means adding a key to `_AVAILABLE_SPECIALISTS` and a graph
node, not restructuring this function or the graph's edges.

Dispatch itself is deterministic (no LLM call needed to decide "run the
one specialist that exists") -- see top-level spec section 9's "prefer
deterministic logic over LLM calls wherever a deterministic answer is
possible". What the LLM-driven agents downstream do with that dispatch is
where the actual reasoning happens.
"""

from typing import List

from ..models.context import IncidentContext
from ..trajectory.logger import TrajectoryLogger
from ..graph.state import PHASE_INVESTIGATING, AgentState

_AVAILABLE_SPECIALISTS = ["application"]  # extend here as Infrastructure/Network/Security agents land


def _select_specialists(context: IncidentContext) -> List[str]:
    return [s for s in _AVAILABLE_SPECIALISTS if s == "application" and context.affected_services]


class Orchestrator:
    def __init__(self, trajectory: TrajectoryLogger):
        self._trajectory = trajectory

    async def __call__(self, state: AgentState) -> dict:
        context = state["incident_context"]
        iteration = state["iteration"] + 1
        rejected = state.get("rejected_hypotheses", [])

        targets = _select_specialists(context)

        if rejected:
            avoid = rejected[-1].root_cause
            reasoning = (
                f"Round {iteration}: previous hypothesis rejected ('{avoid}'). "
                f"Redispatching to {targets} with instruction to explore a different angle."
            )
        else:
            reasoning = f"Round {iteration}: dispatching to {targets} for initial investigation."

        self._trajectory.log(
            agent="orchestrator",
            phase=PHASE_INVESTIGATING,
            reasoning_summary=reasoning,
            round=iteration,
        )

        return {
            "phase": PHASE_INVESTIGATING,
            "dispatch_targets": targets,
            "iteration": iteration,
        }
