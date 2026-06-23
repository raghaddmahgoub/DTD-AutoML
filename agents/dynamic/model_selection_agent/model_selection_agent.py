"""Model Selection Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import logging
from langgraph.types import interrupt

from state.pipeline_state import PipelineState
from tools.shared import get_llm
from tools.training import plan_training

logger = logging.getLogger(__name__)


def _build_feedback_context(state: PipelineState, agent_names) -> str:
    """
    Pull feedback entries for the given agent name(s) from state["feedback_history"]
    and format them for the prompt.

    model_selection re-plans on feedback from TWO checkpoints:
        - its own checkpoint ("model_selection") — feedback on the plan itself
        - the training checkpoint ("training") — feedback on the trained
          result (e.g. "try XGBoost instead"), rerouted here by
          graph_builder.py's training-checkpoint router because
          plan_training already has a reliable LLM step to turn that into
          a concrete model/approach change, instead of training_agent
          guessing at it.
    """
    if isinstance(agent_names, str):
        agent_names = (agent_names,)
    history = state.get("feedback_history", []) or []
    own = [h["feedback_text"] for h in history if h.get("agent") in agent_names]
    if own:
        return "\n\nUser Feedback History:\n" + "\n".join(f"- {f}" for f in own)
    return ""


def model_selection_node(state: PipelineState) -> dict:
    logger.info("[ModelSelectionAgent] Starting model selection planning")
    llm = get_llm()

    user_prefs = dict(state.get("user_preferences") or {})
    tool_input = {
        "target_column": state.get("target_column"),
        "problem_type": state.get("task_type") or state.get("problem_type"),
        "training_approach": user_prefs.get("training_approach", ""),
        "preferred_models": user_prefs.get("preferred_models", []),
        "time_preference": user_prefs.get("time_preference", ""),
        "hw_complexity": user_prefs.get("hw_complexity", ""),
    }

    task = state.get("controller_task") or "Build training plan"
    feedback = _build_feedback_context(state, ("model_selection", "training"))
    prompt = state.get("nl_query", state.get("prompt", "")) + feedback

    result, updated_state = plan_training.invoke({
        "task": task,
        "tool_input": tool_input,
        "prompt": prompt,
        "data_path": state.get("data_path", ""),
        "llm": llm,
        "state": state,
    })

    # Sync fields to PipelineState compatibility
    plan = updated_state.get("training_plan") or {}
    updated_state["automl_config"] = plan.get("automl_config")
    updated_state["model_selection_reasoning"] = plan.get("reasoning")

    # Set UI agent output
    agent_output = {
        "status": result.get("status"),
        "message": result.get("message"),
        "plan_preview": result.get("plan_preview"),
        "train_tool": result.get("train_tool"),
        "error": result.get("error") if result.get("status") == "error" else None,
    }

    merged_outputs = dict(updated_state.get("agent_outputs", {}))
    merged_outputs["model_selection"] = agent_output
    updated_state["agent_outputs"] = merged_outputs

    if result.get("status") == "error":
        updated_state["error"] = result.get("error", "plan_training failed")

    return updated_state

def route_after_model_selection(state: PipelineState) -> str:
    flags = state["intent_flags"]
    if flags.get("run_training"):
        return "training_agent"
    if flags.get("run_evaluation"):
        return "evaluation_agent"
    if flags.get("run_deployment"):
        return "deployment_agent"
    return "pipeline_done"
