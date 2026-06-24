"""Training Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import logging
from langgraph.types import interrupt

from state.pipeline_state import PipelineState
from tools.shared import get_llm
from tools.training import train_simple, train_simple_optuna, train_autogluon
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


def training_node(state: PipelineState) -> dict:
    plan = state.get("training_plan") or {}
    train_tool_name = plan.get("train_tool")
    run_id = state.get("run_id")

    sub_nodes = [
        {"name": "Initialization", "description": f"Initializing training configurations for {train_tool_name}.", "status": "pending"},
        {"name": "Training", "description": f"Running model fit via {train_tool_name}.", "status": "pending"},
        {"name": "Save Artifact", "description": "Saving serialized model file.", "status": "pending"}
    ]
    agent_output = {
        "status": "running",
        "sub_nodes": sub_nodes
    }

    # Initialize progress in DB
    update_agent_progress(run_id, "training", agent_output)

    def update_step(name: str, status: str, description: str = None):
        for node in sub_nodes:
            if node["name"] == name:
                node["status"] = status
                if description:
                    node["description"] = description
                break
        update_agent_progress(run_id, "training", agent_output)

    if not plan.get("approved"):
        logger.error("[TrainingAgent] Training plan not approved")
        for node in sub_nodes:
            node["status"] = "skipped"
        agent_output["status"] = "skipped"
        agent_output["sub_nodes"] = sub_nodes
        update_agent_progress(run_id, "training", agent_output)

        outputs = dict(state.get("agent_outputs", {}))
        outputs["training"] = agent_output

        return {
            **state,
            "error": "Training plan not approved",
            "step": "train_skipped",
            "agent_outputs": outputs
        }

    tool_map = {
        "train_simple": train_simple,
        "train_simple_optuna": train_simple_optuna,
        "train_autogluon": train_autogluon,
    }

    tool = tool_map.get(train_tool_name)
    if tool is None:
        logger.error("[TrainingAgent] Unknown train tool: %s", train_tool_name)
        sub_nodes[0]["status"] = "failed"
        sub_nodes[0]["description"] = f"Unknown train tool: {train_tool_name}"
        for node in sub_nodes[1:]:
            node["status"] = "skipped"
        agent_output["status"] = "failed"
        agent_output["sub_nodes"] = sub_nodes
        update_agent_progress(run_id, "training", agent_output)

        outputs = dict(state.get("agent_outputs", {}))
        outputs["training"] = agent_output

        return {
            **state,
            "error": f"Unknown train tool: {train_tool_name}",
            "step": "error",
            "agent_outputs": outputs
        }

    llm = get_llm()
    tool_input = {}
    if train_tool_name == "train_simple_optuna":
        plan_optuna = plan.get("optuna_config") or {}
        tool_input["optuna_trials"] = plan_optuna.get("n_trials", 30)

    task = f"Train with {train_tool_name}"
    feedback = _build_feedback_context(state, "training")
    prompt = state.get("nl_query", state.get("prompt", "")) + feedback

    update_step("Initialization", "completed", f"Selected tool '{train_tool_name}' and initialized model parameters.")
    update_step("Training", "running")
    update_step("Save Artifact", "running")

    result, updated_state = tool.invoke({
        "task": task,
        "tool_input": tool_input,
        "prompt": prompt,
        "data_path": state.get("data_path", ""),
        "llm": llm,
        "state": state,
    })

    # Sync fields to PipelineState compatibility
    pickle_path = updated_state.get("saved_files", {}).get("pickle")
    if pickle_path:
        updated_state["trained_model_path"] = pickle_path

    # Set UI agent output
    error_val = None
    status_val = result.get("status", "success")
    if status_val == "error":
        error_val = result.get("error", f"{train_tool_name} failed")

    best_model = result.get("best_model", "Unknown Model")
    best_score = result.get("best_score")
    score_desc = f" (score: {best_score:.4f})" if isinstance(best_score, float) else (f" (score: {best_score})" if best_score is not None else "")
    pickle_file = (result.get("saved_files") or {}).get("pickle", "")

    if status_val == "success":
        update_step("Training", "completed", f"Successfully trained best model: {best_model}{score_desc}.")
        update_step("Save Artifact", "completed", f"Saved model to: {pickle_file}." if pickle_file else "Saved model artifact.")
    else:
        update_step("Training", "failed", f"Failed: {error_val}")
        update_step("Save Artifact", "skipped")

    agent_output = {
        "status": status_val,
        "training_method": result.get("training_method"),
        "best_model": best_model,
        "best_score": best_score,
        "saved_files": result.get("saved_files"),
        "error": error_val,
        "sub_nodes": sub_nodes
    }
    update_agent_progress(run_id, "training", agent_output)

    merged_outputs = dict(updated_state.get("agent_outputs", {}))
    merged_outputs["training"] = agent_output
    updated_state["agent_outputs"] = merged_outputs

    if error_val:
        updated_state["error"] = error_val

    return updated_state

def route_after_training(state: PipelineState) -> str:
    flags = state["intent_flags"]
    if flags.get("evaluation"):
        return "evaluation_agent"
    if flags.get("deployment"):
        return "deployment_agent"
    return "pipeline_done"
