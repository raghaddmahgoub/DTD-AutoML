"""Backward compatible ModelAgent wrapper importing from separate agents."""
from __future__ import annotations

import logging
from typing import Any

from state.pipeline_state import PipelineState
from tools.shared import get_llm

# Import nodes from their new locations to preserve backward compatibility
from agents.dynamic.model_selection_agent import (
    model_selection_node,
    route_after_model_selection,
)
from agents.dynamic.training_agent import (
    training_node,
    route_after_training,
)
from agents.dynamic.evaluation_agent import (
    evaluation_node,
    route_after_evaluation,
)

logger = logging.getLogger(__name__)


class ModelAgent:
    """
    Backward-compatible wrapper executing selection, training, and evaluation
    nodes sequentially without creating a standalone graph or state.
    """

    def __init__(self, logger: Any, llm: Any, registry: Any):
        self.logger = logger
        self.llm = llm
        self.registry = registry

    def run(
        self,
        data_path: str,
        prompt: str,
        pipeline_state: dict | None = None,
        *,
        task: str = "Train model",
        training_approach: str = "",
        target_column: str = "",
        problem_type: str = "",
        optuna_trials: str | int | None = None,
        plan_input: dict | None = None,
        train_input: dict | None = None,
        evaluate_input: dict | None = None,
    ) -> dict:
        from tools.shared import ensure_state

        state = ensure_state(pipeline_state, data_path, prompt)

        # Merge configuration inputs
        user_prefs = dict(state.get("user_preferences") or {})
        if training_approach:
            user_prefs["training_approach"] = training_approach
        if target_column:
            state["target_column"] = target_column
        if problem_type:
            state["task_type"] = problem_type

        state["user_preferences"] = user_prefs
        if task:
            state["controller_task"] = task
        if prompt:
            state["prompt"] = prompt
            state["nl_query"] = prompt

        self.logger.info("\n" + "=" * 50)
        self.logger.info("MODEL AGENT (Sequential Execution)")
        self.logger.info("=" * 50)

        # Execute Node 1: Model Selection
        state = model_selection_node(state)
        if state.get("error"):
            self.logger.warning(f"ModelAgent finished with error: {state['error']}")
            return state

        # Execute Node 2: Training
        if optuna_trials is not None:
            plan = state.get("training_plan") or {}
            optuna_cfg = plan.get("optuna_config") or {}
            optuna_cfg["n_trials"] = int(optuna_trials)
            plan["optuna_config"] = optuna_cfg
            state["training_plan"] = plan

        state = training_node(state)
        if state.get("error"):
            self.logger.warning(f"ModelAgent finished with error: {state['error']}")
            return state

        # Execute Node 3: Evaluation
        state = evaluation_node(state)
        if state.get("error"):
            self.logger.warning(f"ModelAgent finished with error: {state['error']}")
            return state

        self.logger.info(
            f"ModelAgent finished — step={state.get('step')} "
            f"status={state.get('status')}"
        )
        return state
