"""ModelAgent — LangGraph orchestrator for plan → train → evaluate."""
from __future__ import annotations

from typing import Any

from agents.dynamic.model_agent.graph import build_model_graph
from agents.dynamic.model_agent.state import ModelAgentState
from tools.pipeline_state import empty_state


class ModelAgent:
    """
    LangGraph agent that runs the training workflow by calling the tools layer:
      plan_training → train_* → evaluate
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
        ask_before_training: bool = True,
        auto_approve_plan: bool = False,
        training_approach: str = "",
        target_column: str = "",
        problem_type: str = "",
        optuna_trials: str | int | None = None,
        plan_input: dict | None = None,
        train_input: dict | None = None,
        evaluate_input: dict | None = None,
    ) -> dict:
        config = {
            "ask_before_training": ask_before_training,
            "auto_approve_plan": auto_approve_plan,
            "training_approach": training_approach,
            "target_column": target_column,
            "problem_type": problem_type,
            "plan_input": plan_input or {},
            "train_input": train_input or {},
            "evaluate_input": evaluate_input or {},
        }
        if optuna_trials is not None:
            config["optuna_trials"] = str(optuna_trials)

        graph = build_model_graph(self.llm, self.registry, config)
        initial: ModelAgentState = {
            "data_path": data_path,
            "prompt": prompt,
            "task": task,
            "pipeline_state": pipeline_state or empty_state(data_path, prompt),
            "step": "model_agent_start",
        }

        self.logger.info("\n" + "=" * 50)
        self.logger.info("MODEL AGENT (LangGraph)")
        self.logger.info("=" * 50)

        final_state: ModelAgentState = graph.invoke(initial)
        pipeline_state = final_state.get("pipeline_state") or initial["pipeline_state"]

        if final_state.get("error"):
            self.logger.warn(f"ModelAgent finished with error: {final_state['error']}")
        else:
            self.logger.info(
                f"ModelAgent finished — step={pipeline_state.get('step')} "
                f"status={pipeline_state.get('status')}"
            )

        return pipeline_state
