"""LangGraph node: evaluate trained model via evaluate tool."""
from __future__ import annotations

from typing import Any, Callable

from agents.dynamic.model_agent.state import ModelAgentState
from agents.dynamic.model_agent.tool_runner import invoke_tool


def make_evaluate_node(llm: Any, registry: Any, config: dict) -> Callable[[ModelAgentState], ModelAgentState]:
    evaluate_tool = registry.get("evaluate")
    if evaluate_tool is None:
        raise RuntimeError("evaluate tool is not registered")

    def evaluate_node(state: ModelAgentState) -> ModelAgentState:
        pipeline_state = state.get("pipeline_state") or {}

        result, pipeline_state = invoke_tool(
            evaluate_tool,
            task="Evaluate trained model",
            tool_input=dict(config.get("evaluate_input") or {}),
            prompt=state.get("prompt", ""),
            data_path=state.get("data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        out: ModelAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": "evaluate",
            "last_result": result,
            "step": pipeline_state.get("step", "evaluated"),
        }
        if result.get("status") == "error":
            out["error"] = result.get("error", "evaluate failed")
        return out

    return evaluate_node
