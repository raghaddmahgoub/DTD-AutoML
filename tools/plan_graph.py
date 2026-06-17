"""LangGraph subgraph for planning: identify_target → model_selection."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from tools.graph_state import TrainingGraphState
from tools.nodes.identify_target import identify_target_node
from tools.nodes.model_selection import model_selection_node


def build_plan_graph(llm):
    """Compiled LangGraph used by plan_training tool."""

    def _model_selection(state: TrainingGraphState) -> TrainingGraphState:
        return model_selection_node(state, llm)

    workflow = StateGraph(TrainingGraphState)
    workflow.add_node("identify_target", identify_target_node)
    workflow.add_node("model_selection", _model_selection)
    workflow.set_entry_point("identify_target")
    workflow.add_edge("identify_target", "model_selection")
    workflow.add_edge("model_selection", END)
    return workflow.compile()
