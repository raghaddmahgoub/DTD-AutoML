"""Invoke LangChain training tools with a consistent signature."""
from __future__ import annotations

from typing import Any


def invoke_tool(
    tool: Any,
    *,
    task: str,
    tool_input: dict,
    prompt: str,
    data_path: str,
    llm: Any,
    pipeline_state: dict,
) -> tuple[dict, dict]:
    return tool.invoke(
        {
            "task": task,
            "tool_input": tool_input,
            "prompt": prompt,
            "data_path": data_path,
            "llm": llm,
            "state": pipeline_state,
        }
    )
