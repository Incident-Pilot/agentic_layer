"""LangGraph wiring for the incident investigation state machine:

    DETECTED -> INVESTIGATING -> HYPOTHESIS_GENERATED -> VERIFYING
        -> ROOT_CAUSE_CONFIRMED
        -> (or) VERIFICATION_FAILED -> back to INVESTIGATING (replan)
        -> (or) ESCALATED (max iterations exceeded)

Nodes: orchestrator -> investigator -> synthesizer -> verifier, with a
conditional edge after verifier that either ends the run (CONFIRMED),
loops back to orchestrator (REJECTED, iteration budget remaining), or ends
as ESCALATED (REJECTED, budget exhausted). No remediation states exist
past ROOT_CAUSE_CONFIRMED -- out of scope for this phase.
"""

from langgraph.graph import END, StateGraph

from ..agents.investigator import ApplicationInvestigationAgent
from ..agents.orchestrator import Orchestrator
from ..agents.synthesizer import HypothesisSynthesizer
from ..agents.verifier import VerificationAgent
from ..llm.base import LLMClient
from ..tools.base import Tool
from ..trajectory.logger import TrajectoryLogger
from .state import (
    PHASE_DETECTED,
    PHASE_ESCALATED,
    PHASE_VERIFICATION_FAILED,
    AgentState,
)


def _route_after_verifier(state: AgentState) -> str:
    if state.get("final_status") == "CONFIRMED":
        return "confirmed"
    if state["iteration"] >= state["max_iterations"]:
        return "escalated"
    return "replan"


def build_graph(llm: LLMClient, tools: list[Tool], trajectory: TrajectoryLogger):
    orchestrator = Orchestrator(trajectory)
    investigator = ApplicationInvestigationAgent(llm, tools, trajectory)
    synthesizer = HypothesisSynthesizer(llm, trajectory)
    verifier = VerificationAgent(llm, tools, trajectory)

    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("investigator", investigator)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("verifier", verifier)

    graph.set_entry_point("orchestrator")
    # Orchestrator's dispatch_targets is a list so adding a second
    # specialist later is a registry entry + a graph node, not a rewrite
    # of this edge -- in this phase it is always exactly ["application"].
    graph.add_edge("orchestrator", "investigator")
    graph.add_edge("investigator", "synthesizer")
    graph.add_edge("synthesizer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"confirmed": END, "escalated": END, "replan": "orchestrator"},
    )

    return graph.compile()


def initial_state(incident_context, max_iterations: int = 4) -> AgentState:
    return AgentState(
        incident_context=incident_context,
        phase=PHASE_DETECTED,
        evidence=[],
        hypotheses=[],
        verifications=[],
        rejected_hypotheses=[],
        dispatch_targets=[],
        current_hypothesis_id=None,
        iteration=0,
        max_iterations=max_iterations,
        final_status=None,
    )


def finalize_status(state: AgentState) -> AgentState:
    """Called by the CLI after graph.ainvoke() returns, since the escalated
    path ends the graph via the same END target as confirmed -- this sets
    the distinguishing phase/final_status the router itself doesn't get a
    chance to write (there is no node after the conditional edge)."""
    if state.get("final_status") != "CONFIRMED":
        state["final_status"] = "ESCALATED"
        state["phase"] = PHASE_ESCALATED if state["iteration"] >= state["max_iterations"] else PHASE_VERIFICATION_FAILED
    return state
