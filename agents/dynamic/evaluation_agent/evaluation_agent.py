"""Evaluation Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import logging
from langgraph.types import interrupt

from state.pipeline_state import PipelineState
from tools.shared import get_llm
from tools.training import evaluate

logger = logging.getLogger(__name__)


def _build_feedback_context(state: PipelineState, agent_name: str) -> str:
    """
    Pull this agent's own feedback entries from state["feedback_history"]
    and format them for the prompt.
    """
    history = state.get("feedback_history", []) or []
    own = [h["feedback_text"] for h in history if h.get("agent") == agent_name]
    if own:
        return f"\n\nUser Feedback History for {agent_name}:\n" + "\n".join(f"- {f}" for f in own)
    return ""


def evaluation_node(state: PipelineState) -> dict:
    logger.info("[EvaluationAgent] Starting evaluation")
    llm = get_llm()

    feedback = _build_feedback_context(state, "evaluation")
    prompt = state.get("nl_query", state.get("prompt", "")) + feedback

    result, updated_state = evaluate.invoke({
        "task": "Evaluate trained model",
        "tool_input": {},
        "prompt": prompt,
        "data_path": state.get("data_path", ""),
        "llm": llm,
        "state": state,
    })

    # Set UI agent output
    agent_output = {
        "status": result.get("status"),
        "metrics": result.get("metrics"),
        "best_model": result.get("best_model"),
        "best_score": result.get("best_score"),
        "error": result.get("error") if result.get("status") == "error" else None,
    }

    merged_outputs = dict(updated_state.get("agent_outputs", {}))
    merged_outputs["evaluation"] = agent_output
    updated_state["agent_outputs"] = merged_outputs

    if result.get("metrics"):
        updated_state["model_metrics"] = result.get("metrics")

    if result.get("status") == "error":
        updated_state["error"] = result.get("error", "evaluation failed")

    return updated_state

def route_after_evaluation(state: PipelineState) -> str:
    flags = state["intent_flags"]
    if flags.get("run_deployment"):
        return "deployment_agent"
    return "pipeline_done"
