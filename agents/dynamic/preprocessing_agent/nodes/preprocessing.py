"""LangGraph node: execute preprocessing via preprocessing_execution tool."""
from __future__ import annotations

from typing import Any, Callable

from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from agents.dynamic.preprocessing_agent.tool_runner import invoke_tool


def make_preprocessing_node(llm: Any, registry: Any, config: dict) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    """
    Create a preprocessing execution node that calls the preprocessing_execution tool.

    Args:
        llm: Language model instance
        registry: Tool registry
        config: Configuration dict with:
            - preprocessing_input: dict with tool parameters
            - target_column: str (optional)
            - test_size: float (optional)
            - use_llm: bool (optional)

    Returns:
        Node function for LangGraph
    """
    preprocessing_tool = registry.get("preprocessing_execution")
    if preprocessing_tool is None:
        raise RuntimeError("preprocessing_execution tool is not registered")

    def preprocessing_node(state: PreprocessingAgentState) -> PreprocessingAgentState:
        pipeline_state = state.get("pipeline_state") or {}
        tool_input = dict(config.get("preprocessing_input") or {})

        # Apply config overrides
        if config.get("target_column"):
            tool_input["target_column"] = config["target_column"]
        if config.get("test_size"):
            tool_input["test_size"] = config["test_size"]
        if config.get("use_llm") is not None:
            tool_input["use_llm"] = config["use_llm"]

        result, pipeline_state = invoke_tool(
            preprocessing_tool,
            task=state.get("task") or "Execute preprocessing pipeline",
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get(
                "data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        out: PreprocessingAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": "preprocessing_execution",
            "last_result": result,
            "step": pipeline_state.get("step", "preprocessing_complete"),
        }
        if result.get("status") == "error":
            out["error"] = result.get(
                "error", "preprocessing_execution failed")
        return out

    return preprocessing_node
