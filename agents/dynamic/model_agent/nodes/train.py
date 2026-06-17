"""LangGraph node: run the train tool selected in the training plan."""
from __future__ import annotations

from typing import Any, Callable

from agents.dynamic.model_agent.state import ModelAgentState
from agents.dynamic.model_agent.tool_runner import invoke_tool


def make_train_node(llm: Any, registry: Any, config: dict) -> Callable[[ModelAgentState], ModelAgentState]:
    def train_node(state: ModelAgentState) -> ModelAgentState:
        pipeline_state = state.get("pipeline_state") or {}
        plan = pipeline_state.get("training_plan") or {}
        train_tool_name = plan.get("train_tool")

        if not plan.get("approved"):
            return {
                **state,
                "error": "Training plan not approved",
                "step": "train_skipped",
            }

        tool = registry.get(train_tool_name)
        if tool is None:
            return {
                **state,
                "error": f"Unknown train tool: {train_tool_name}",
                "step": "error",
            }

        tool_input = dict(config.get("train_input") or {})
        if train_tool_name == "train_simple_optuna" and "optuna_trials" not in tool_input:
            plan_optuna = plan.get("optuna_config") or {}
            tool_input["optuna_trials"] = config.get(
                "optuna_trials", plan_optuna.get("n_trials", 30)
            )

        result, pipeline_state = invoke_tool(
            tool,
            task=f"Train with {train_tool_name}",
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get("data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        out: ModelAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": train_tool_name,
            "last_result": result,
            "step": pipeline_state.get("step", "model_trained"),
        }
        if result.get("status") == "error":
            out["error"] = result.get("error", f"{train_tool_name} failed")
        return out

    return train_node
