"""Evaluation Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import logging
from langgraph.types import interrupt

from state.pipeline_state import PipelineState
from tools.shared import get_llm
from tools.training import evaluate
from graph.knowledge_graph import update_agent_progress

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

    run_id = state.get("run_id")
    sub_nodes = [
        {"name": "Load Test", "description": "Loading independent preprocessed testing splits.", "status": "pending"},
        {"name": "Prediction", "description": "Running model inference on test dataset.", "status": "pending"},
        {"name": "Compute Metrics", "description": "Calculating evaluation performance metrics.", "status": "pending"}
    ]
    agent_output = {
        "status": "running",
        "sub_nodes": sub_nodes
    }

    # Initialize progress in DB
    update_agent_progress(run_id, "evaluation", agent_output)

    def update_step(name: str, status: str, description: str = None):
        for node in sub_nodes:
            if node["name"] == name:
                node["status"] = status
                if description:
                    node["description"] = description
                break
        update_agent_progress(run_id, "evaluation", agent_output)

    feedback = _build_feedback_context(state, "evaluation")
    prompt = state.get("nl_query", state.get("prompt", "")) + feedback

    update_step("Load Test", "running")
    update_step("Prediction", "running")
    update_step("Compute Metrics", "running")

    result, updated_state = evaluate.invoke({
        "task": "Evaluate trained model",
        "tool_input": {},
        "prompt": prompt,
        "data_path": state.get("data_path", ""),
        "llm": llm,
        "state": state,
    })

    # Set UI agent output
    status_val = result.get("status", "success")
    metrics = result.get("metrics") or {}
    metrics_desc = ", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
    best_model = result.get("best_model", "Unknown Model")

    if status_val == "success":
        update_step("Load Test", "completed", "Loaded test splits successfully.")
        update_step("Prediction", "completed", f"Ran inference on test dataset with {best_model}.")
        update_step("Compute Metrics", "completed", f"Calculated performance metrics: {metrics_desc}." if metrics_desc else "Computed evaluation metrics.")
    else:
        err_msg = result.get("error", "Evaluation failed.")
        update_step("Load Test", "failed", f"Failed: {err_msg}")
        update_step("Prediction", "skipped")
        update_step("Compute Metrics", "skipped")

    agent_output = {
        "status": status_val,
        "metrics": metrics,
        "best_model": best_model,
        "best_score": result.get("best_score"),
        "error": result.get("error") if status_val == "error" else None,
        "sub_nodes": sub_nodes
    }
    update_agent_progress(run_id, "evaluation", agent_output)

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
    if flags.get("deployment"):
        return "deployment_agent"
    return "pipeline_done"
