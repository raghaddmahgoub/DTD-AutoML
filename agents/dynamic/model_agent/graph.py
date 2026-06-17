"""LangGraph workflow for ModelAgent: plan → train → evaluate."""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.dynamic.model_agent.nodes.evaluate import make_evaluate_node
from agents.dynamic.model_agent.nodes.plan import make_plan_node
from agents.dynamic.model_agent.nodes.train import make_train_node
from agents.dynamic.model_agent.state import ModelAgentState


def _route_after_plan(state: ModelAgentState) -> Literal["train", "end"]:
    if state.get("error"):
        return "end"
    result = state.get("last_result") or {}
    if result.get("status") != "planned":
        return "end"
    plan = (state.get("pipeline_state") or {}).get("training_plan") or {}
    if not plan.get("approved"):
        return "end"
    if not plan.get("train_tool"):
        return "end"
    return "train"


def _route_after_train(state: ModelAgentState) -> Literal["evaluate", "end"]:
    if state.get("error"):
        return "end"
    result = state.get("last_result") or {}
    if result.get("status") != "success":
        return "end"
    return "evaluate"


def build_model_graph(llm: Any, registry: Any, config: dict | None = None):
    """
    Compiled LangGraph for the training phase.

    Nodes:
      plan     → plan_training tool (includes identify_target + model_selection subgraph)
      train    → train_simple | train_simple_optuna | train_autogluon (from plan)
      evaluate → evaluate tool
    """
    cfg = dict(config or {})

    workflow = StateGraph(ModelAgentState)
    workflow.add_node("plan", make_plan_node(llm, registry, cfg))
    workflow.add_node("train", make_train_node(llm, registry, cfg))
    workflow.add_node("evaluate", make_evaluate_node(llm, registry, cfg))

    workflow.set_entry_point("plan")
    workflow.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"train": "train", "end": END},
    )
    workflow.add_conditional_edges(
        "train",
        _route_after_train,
        {"evaluate": "evaluate", "end": END},
    )
    workflow.add_edge("evaluate", END)

    return workflow.compile()
