"""LangGraph workflow for PreprocessingAgent: execute data preprocessing pipeline."""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.dynamic.preprocessing_agent.nodes.preprocessing import make_preprocessing_node
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState


def _route_after_preprocessing(state: PreprocessingAgentState) -> Literal["end"]:
    """
    Route after preprocessing completes.
    Currently just routes to END, but can be extended for multi-step workflows.
    """
    return "end"


def build_preprocessing_graph(llm: Any, registry: Any, config: dict | None = None):
    """
    Compiled LangGraph for the preprocessing phase.

    Nodes:
      preprocessing → preprocessing_execution tool (handles all preprocessing steps)

    Args:
        llm: Language model instance
        registry: Tool registry containing 'preprocessing_execution' tool
        config: Configuration dict with preprocessing_input and other parameters

    Returns:
        Compiled LangGraph workflow
    """
    cfg = dict(config or {})

    workflow = StateGraph(PreprocessingAgentState)
    workflow.add_node(
        "preprocessing", make_preprocessing_node(llm, registry, cfg))

    workflow.set_entry_point("preprocessing")
    workflow.add_conditional_edges(
        "preprocessing",
        _route_after_preprocessing,
        {"end": END},
    )

    return workflow.compile()
