"""LangGraph node: build training plan via plan_training tool."""
from __future__ import annotations

from typing import Any, Callable

from agents.dynamic.model_agent.state import ModelAgentState
from agents.dynamic.model_agent.tool_runner import invoke_tool


def make_plan_node(llm: Any, registry: Any, config: dict) -> Callable[[ModelAgentState], ModelAgentState]:
    plan_tool = registry.get("plan_training")
    if plan_tool is None:
        raise RuntimeError("plan_training tool is not registered")

    def plan_node(state: ModelAgentState) -> ModelAgentState:
        pipeline_state = state.get("pipeline_state") or {}
        tool_input = dict(config.get("plan_input") or {})

        controller_task = (
            state.get("task")
            or pipeline_state.get("controller_task")
            or "Build training plan"
        )
        tool_input.setdefault("controller_task", controller_task)

        if config.get("training_approach"):
            tool_input["training_approach"] = config["training_approach"]
        if config.get("target_column"):
            tool_input["target_column"] = config["target_column"]
        if config.get("problem_type"):
            tool_input["problem_type"] = config["problem_type"]

        result, pipeline_state = invoke_tool(
            plan_tool,
            task=controller_task,
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get("data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        out: ModelAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": "plan_training",
            "last_result": result,
            "step": pipeline_state.get("step", "plan_complete"),
        }
        if result.get("status") == "error":
            out["error"] = result.get("error", "plan_training failed")
        return out

    return plan_node
