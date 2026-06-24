"""Dynamic Feature Engineering Agent — LangGraph node wrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from state.pipeline_state import PipelineState
from tools.feature_engineering import feature_engineering_execution
from tools.shared import get_llm
from src.utils.logger import Logger

logger = logging.getLogger(__name__)


def _build_feedback_context(state: dict, agent_name: str = "feature_engineering") -> str:
    """Format user feedback for this feature-engineering checkpoint."""
    history = state.get("feedback_history", []) or []
    own = [h.get("feedback_text", "") for h in history if h.get("agent") == agent_name]
    own = [text for text in own if text]
    if own:
        return "\n\nUser Feedback History for feature engineering:\n" + "\n".join(f"- {text}" for text in own)
    return ""


class FeatureEngineeringAgent:
    """Run feature engineering after preprocessing using the execution tool."""

    def __init__(self, logger_obj: Any = None, llm: Any = None, registry: Any = None):
        self.logger = logger_obj or Logger()
        self.llm = llm or get_llm()

    def run(self, pipeline_state: dict) -> dict:
        data_path = pipeline_state.get("data_path")
        base_prompt = pipeline_state.get("prompt") or pipeline_state.get("nl_query") or "feature engineer the data"
        feedback_context = _build_feedback_context(pipeline_state)
        prompt = f"{base_prompt}{feedback_context}"
        preprocessing_plan = pipeline_state.get("preprocessing_plan") or {}
        feature_plan = preprocessing_plan.get("feature_engineering") or {}

        try:
            top_k = int(feature_plan.get("top_k", 4))
        except (TypeError, ValueError):
            top_k = 4
        top_k = max(0, min(20, top_k))
        enabled = bool(feature_plan.get("enabled", top_k > 0)) and top_k > 0

        outputs = dict(pipeline_state.get("agent_outputs", {}))
        if not enabled:
            output = {
                "status": "skipped",
                "message": "Feature engineering disabled by preprocessing plan.",
                "top_k": top_k,
            }
            outputs["feature_engineering"] = output
            pipeline_state["agent_outputs"] = outputs
            pipeline_state["step"] = "feature_engineering_skipped"
            pipeline_state["status"] = "success"
            return pipeline_state

        required_paths = {
            "X_train_path": pipeline_state.get("X_train_path"),
            "X_test_path": pipeline_state.get("X_test_path"),
            "y_train_path": pipeline_state.get("y_train_path"),
        }
        missing = [name for name, value in required_paths.items() if not value or not Path(value).exists()]
        if missing:
            message = "Feature engineering requires completed preprocessing splits; missing: " + ", ".join(missing)
            output = {"status": "error", "error": message}
            outputs["feature_engineering"] = output
            pipeline_state["agent_outputs"] = outputs
            pipeline_state["step"] = "feature_engineering_failed"
            pipeline_state["status"] = "error"
            pipeline_state["error"] = message
            return pipeline_state

        try:
            max_candidates = int(feature_plan.get("max_candidates", max(12, top_k * 3)))
        except (TypeError, ValueError):
            max_candidates = max(12, top_k * 3)

        output_folder = (
            (pipeline_state.get("preprocessing_output") or {}).get("output_folder")
            or str(Path(required_paths["X_train_path"]).parent)
        )

        self.logger.info("[FeatureEngineeringAgent] Running feature engineering...")
        result, new_state = feature_engineering_execution.invoke({
            "task": "Generate and select feature-engineering columns after preprocessing",
            "tool_input": {
                **required_paths,
                "top_k": top_k,
                "max_candidates": max_candidates,
                "use_llm": True,
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })

        outputs = dict(new_state.get("agent_outputs", {}))
        if result.get("status") == "success":
            feature_output = result.get("feature_engineering_output") or new_state.get("feature_engineering_output") or {}
            outputs["feature_engineering"] = {
                "status": "success",
                "message": result.get("message"),
                "feedback_applied": bool(feedback_context),
                **feature_output,
            }
            new_state["agent_outputs"] = outputs
            new_state["status"] = "success"
            return new_state

        outputs["feature_engineering"] = {
            "status": "error",
            "error": result.get("error", "Feature engineering failed."),
        }
        new_state["agent_outputs"] = outputs
        return new_state


def feature_engineering_node(state: PipelineState) -> dict:
    """LangGraph node representing the Feature Engineering Agent."""
    logger.info("[FeatureEngineeringAgent] Executing LangGraph node")
    return FeatureEngineeringAgent().run(state)


def route_after_feature_engineering(state: PipelineState) -> str:
    """Route to the next active agent after feature engineering."""
    flags = state.get("intent_flags", {})
    if flags.get("model_selection"):
        return "model_selection_agent"
    if flags.get("training"):
        return "training_agent"
    if flags.get("evaluation"):
        return "evaluation_agent"
    if flags.get("deployment"):
        return "deployment_agent"
    return "pipeline_done"
