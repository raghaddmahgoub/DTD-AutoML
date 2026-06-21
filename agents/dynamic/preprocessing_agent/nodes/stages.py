"""Reusable LangGraph nodes for the dynamic preprocessing tool stages."""
from __future__ import annotations

from typing import Any, Callable

from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from agents.dynamic.preprocessing_agent.tool_runner import invoke_tool


def make_stage_node(
    tool_name: str,
    default_task: str,
    llm: Any,
    registry: Any,
    config: dict,
) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    tool = registry.get(tool_name)
    if tool is None:
        raise RuntimeError(f"{tool_name} is not registered")

    def stage_node(state: PreprocessingAgentState) -> PreprocessingAgentState:
        pipeline_state = state.get("pipeline_state") or {}
        tool_input = {
            "plan": pipeline_state.get("preprocessing_plan"),
            "target_column": pipeline_state.get("target_column"),
            "output_folder": config["output_folder"],
            "test_size": config.get("test_size", 0.2),
            "random_state": config.get("random_state", 42),
            "X_train_path": pipeline_state.get("X_train_path"),
            "X_test_path": pipeline_state.get("X_test_path"),
            "y_train_path": pipeline_state.get("y_train_path"),
            "y_test_path": pipeline_state.get("y_test_path"),
        }
        result, pipeline_state = invoke_tool(
            tool,
            task=default_task,
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get("data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )
        out: PreprocessingAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": tool_name,
            "last_result": result,
            "step": pipeline_state.get("step", tool_name),
        }
        if result.get("status") == "error":
            out["error"] = result.get("error", f"{tool_name} failed")
        return out

    return stage_node
