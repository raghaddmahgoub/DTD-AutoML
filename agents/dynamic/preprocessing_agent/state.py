"""LangGraph state for PreprocessingAgent."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class PreprocessingAgentState(TypedDict, total=False):
    """State passed between PreprocessingAgent LangGraph nodes."""

    data_path: str
    prompt: str
    task: str
    pipeline_state: dict[str, Any]
    last_tool: str
    last_result: dict[str, Any]
    step: str
    error: Optional[str]
