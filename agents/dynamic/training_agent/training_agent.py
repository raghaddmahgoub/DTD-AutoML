"""Training Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import logging
from langgraph.types import interrupt

from state.pipeline_state import PipelineState
from tools.llm_client import get_llm
from tools.train_simple import train_simple
from tools.train_simple_optuna import train_simple_optuna
from tools.train_autogluon import train_autogluon

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

    if not plan.get("approved"):
        logger.error("[TrainingAgent] Training plan not approved")
        return {
            **state,
            "error": "Training plan not approved",
            "step": "train_skipped",
        }

    tool_map = {
        "train_simple": train_simple,
        "train_simple_optuna": train_simple_optuna,
        "train_autogluon": train_autogluon,
    }

    tool = tool_map.get(train_tool_name)
    if tool is None:
        logger.error("[TrainingAgent] Unknown train tool: %s", train_tool_name)
        return {
            **state,
            "error": f"Unknown train tool: {train_tool_name}",
            "step": "error",
        }

    llm = get_llm()
    tool_input = {}
    if train_tool_name == "train_simple_optuna":
        plan_optuna = plan.get("optuna_config") or {}
        tool_input["optuna_trials"] = plan_optuna.get("n_trials", 30)

    task = f"Train with {train_tool_name}"
    feedback = _build_feedback_context(state, "training")
    prompt = state.get("nl_query", state.get("prompt", "")) + feedback

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
    if result.get("status") == "error":
        error_val = result.get("error", f"{train_tool_name} failed")

    agent_output = {
        "status": result.get("status"),
        "training_method": result.get("training_method"),
        "best_model": result.get("best_model"),
        "best_score": result.get("best_score"),
        "saved_files": result.get("saved_files"),
        "error": error_val,
    }

    merged_outputs = dict(updated_state.get("agent_outputs", {}))
    merged_outputs["training"] = agent_output
    updated_state["agent_outputs"] = merged_outputs

    if error_val:
        updated_state["error"] = error_val

    return updated_state


def training_checkpoint_node(state: PipelineState) -> dict:
    logger.info("[TrainingCheckpoint] Interrupting for human review")
    human_response: dict = interrupt({
        "agent":        "training",
        "agent_output": state["agent_outputs"].get("training", {}),
    })

    decision      = human_response.get("decision", "accept")
    feedback_text = human_response.get("text", "")

    updates: dict = {
        "user_decision": decision,
        "feedback_text":  feedback_text,
    }

    if decision == "feedback" and feedback_text:
        history = list(state.get("feedback_history", []))
        history.append({
            "agent":         "training",
            "feedback_text": feedback_text,
            "iteration":     len([h for h in history if h["agent"] == "training"]) + 1,
        })
        updates["feedback_history"] = history

    logger.info("[TrainingCheckpoint] decision=%s", decision)
    return updates


def route_after_training(state: PipelineState) -> str:
    flags = state["intent_flags"]
    if flags.get("run_evaluation"):
        return "evaluation_agent"
    if flags.get("run_deployment"):
        return "deployment_agent"
    return "pipeline_done"
