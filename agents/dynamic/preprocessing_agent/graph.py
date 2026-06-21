"""Tool-driven LangGraph workflow for the standalone preprocessing agent."""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agents.dynamic.preprocessing_agent.nodes.feature_engineering import (
    make_feature_engineering_node,
)
from agents.dynamic.preprocessing_agent.nodes.stages import make_stage_node
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState


STAGES = [
    ("split", "preprocessing_split", "Prepare data and create the train/test split"),
    ("missing", "preprocessing_missing_values", "Handle missing values"),
    ("outliers", "preprocessing_outliers", "Handle numerical outliers"),
    ("encoding", "preprocessing_encoding", "Encode categorical features"),
    ("scaling", "preprocessing_scaling", "Scale numerical features"),
    ("normalization", "preprocessing_normalization", "Normalize feature rows"),
    ("balancing", "preprocessing_balancing", "Balance the training target"),
    ("validation", "preprocessing_validation", "Validate modeling readiness"),
]


def _route(next_node: str):
    def route(state: PreprocessingAgentState) -> str:
        return "end" if state.get("error") else next_node

    return route


def build_preprocessing_graph(llm: Any, registry: Any, config: dict | None = None):
    """
    Compile preprocessing stages followed by feature engineering.

    This graph belongs only to the standalone preprocessing agent. It does not
    invoke model training, evaluation, EDA, deployment, or the main orchestrator.
    """
    cfg = dict(config or {})
    workflow = StateGraph(PreprocessingAgentState)

    for node_name, tool_name, task in STAGES:
        workflow.add_node(
            node_name,
            make_stage_node(tool_name, task, llm, registry, cfg),
        )
    workflow.add_node(
        "feature_engineering",
        make_feature_engineering_node(llm, registry, cfg),
    )
    workflow.set_entry_point("split")

    stage_names = [stage[0] for stage in STAGES]
    for current, following in zip(stage_names, stage_names[1:]):
        workflow.add_conditional_edges(
            current,
            _route(following),
            {following: following, "end": END},
        )
    workflow.add_conditional_edges(
        "validation",
        _route("feature_engineering"),
        {"feature_engineering": "feature_engineering", "end": END},
    )
    workflow.add_edge("feature_engineering", END)
    return workflow.compile()
