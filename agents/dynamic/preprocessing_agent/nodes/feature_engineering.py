"""LangGraph node: append selected LLM-generated feature combinations."""
from __future__ import annotations

from typing import Any, Callable

from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from agents.dynamic.preprocessing_agent.tool_runner import invoke_tool


def make_feature_engineering_node(
    llm: Any,
    registry: Any,
    config: dict,
) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    feature_tool = registry.get("feature_engineering_execution")
    if feature_tool is None:
        raise RuntimeError("feature_engineering_execution tool is not registered")

    def feature_engineering_node(
        state: PreprocessingAgentState,
    ) -> PreprocessingAgentState:
        pipeline_state = state.get("pipeline_state") or {}
        tool_input = dict(config.get("feature_engineering_input") or {})
        tool_input.setdefault("top_k", config.get("feature_top_k", 4))
        tool_input.setdefault("use_llm", config.get("use_llm", True))

        result, pipeline_state = invoke_tool(
            feature_tool,
            task=state.get("task") or "Generate, evaluate, and select new feature combinations",
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get("data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        out: PreprocessingAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": "feature_engineering_execution",
            "last_result": result,
            "step": pipeline_state.get("step", "feature_engineering_complete"),
        }
        if result.get("status") == "error":
            out["error"] = result.get(
                "error", "feature_engineering_execution failed"
            )
        return out

    return feature_engineering_node
