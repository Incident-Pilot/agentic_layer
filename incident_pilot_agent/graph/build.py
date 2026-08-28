"""LangGraph wiring for the incident investigation state machine:

    DETECTED -> INVESTIGATING -> HYPOTHESIS_GENERATED -> VERIFYING
        -> ROOT_CAUSE_CONFIRMED
            -> (if hypothesis.actionable) -> REMEDIATION_PROPOSED
        -> (or) VERIFICATION_FAILED -> back to INVESTIGATING (replan)
        -> (or) ESCALATED (max iterations exceeded)

Nodes: orchestrator -> investigator -> synthesizer -> verifier, with a
conditional edge after verifier that routes to the remediation planner only
on a genuine CONFIRMED verdict for an actionable hypothesis, ends the run
at ROOT_CAUSE_CONFIRMED for a CONFIRMED-but-not-actionable null finding,
loops back to orchestrator (REJECTED, iteration budget remaining), or ends
as ESCALATED (REJECTED, budget exhausted).
"""

from typing import Optional

from langgraph.graph import END, StateGraph

from ..agents.investigator import ApplicationInvestigationAgent
from ..agents.orchestrator import Orchestrator
from ..agents.remediation_planner import RemediationPlanner
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
        hypothesis = next(h for h in state["hypotheses"] if h.hypothesis_id == state["current_hypothesis_id"])
        return "confirmed_actionable" if hypothesis.actionable else "confirmed_not_actionable"
    if state["iteration"] >= state["max_iterations"]:
        return "escalated"
    return "replan"


def build_graph(
    llm: LLMClient,
    tools: list[Tool],
    trajectory: TrajectoryLogger,
    *,
    investigator_llm: Optional[LLMClient] = None,
    synthesizer_llm: Optional[LLMClient] = None,
    verifier_llm: Optional[LLMClient] = None,
    remediation_llm: Optional[LLMClient] = None,
):
    """`llm` is the shared default; investigator_llm/synthesizer_llm/
    verifier_llm/remediation_llm each override it for that one node only, so
    a caller that wants tiered models (see cli.py's per-node
    OPENROUTER_MODEL overrides) can pass distinct clients while every
    existing call site that just passes `llm` keeps working unchanged --
    purely additive."""
    orchestrator = Orchestrator(trajectory)
    investigator = ApplicationInvestigationAgent(investigator_llm or llm, tools, trajectory)
    synthesizer = HypothesisSynthesizer(synthesizer_llm or llm, trajectory)
    verifier = VerificationAgent(verifier_llm or llm, tools, trajectory)
    remediation_planner = RemediationPlanner(remediation_llm or llm, trajectory)

    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("investigator", investigator)
    graph.add_node("synthesizer", synthesizer)
    graph.add_node("verifier", verifier)
    graph.add_node("remediation_planner", remediation_planner)

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
        {
            "confirmed_actionable": "remediation_planner",
            "confirmed_not_actionable": END,
            "escalated": END,
            "replan": "orchestrator",
        },
    )
    graph.add_edge("remediation_planner", END)

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
        remediation_plan=None,
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
