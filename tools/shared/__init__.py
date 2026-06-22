"""
tools/shared/
─────────────
Pure helpers and shared utilities — no @tool decorator, no agent logic.
All agents import from here instead of from tools/ root.

Exports:
    get_llm                      — LLM factory (ChatGoogleGenerativeAI)
    build_prompt_eda             — EDA system / user prompt builder
    build_prompt_intent_detector — Intent detection prompt builder
    build_prompt_preprocessing   — Preprocessing prompt builder
    extract_schema               — DataFrame column schema extractor
    TargetSuggestionAgent        — Target column suggester
    empty_state / ensure_state / merge_state / parse_tool_input — state helpers
    LARGE_DATA_ROW_THRESHOLD, load_planning_dataframe, load_preprocessed_splits,
    pipeline_to_graph_state, require_preprocessed_splits, resolve_problem_type
"""

from .llm_client import get_llm
from .prompt_builder import (
    build_prompt_eda,
    build_prompt_intent_detector,
    build_prompt_preprocessing,
)
from .schema_extractor import extract_schema
from .target_suggester import TargetSuggestionAgent
from state.pipeline_state import (
    empty_state,
    ensure_state,
    merge_state,
    parse_tool_input,
)
from .training_common import (
    LARGE_DATA_ROW_THRESHOLD,
    load_planning_dataframe,
    load_preprocessed_splits,
    pipeline_to_graph_state,
    require_preprocessed_splits,
    resolve_problem_type,
)

__all__ = [
    # LLM
    "get_llm",
    # Prompts
    "build_prompt_eda",
    "build_prompt_intent_detector",
    "build_prompt_preprocessing",
    # Schema & target helpers
    "extract_schema",
    "TargetSuggestionAgent",
    # State helpers
    "empty_state",
    "ensure_state",
    "merge_state",
    "parse_tool_input",
    # Training common
    "LARGE_DATA_ROW_THRESHOLD",
    "load_planning_dataframe",
    "load_preprocessed_splits",
    "pipeline_to_graph_state",
    "require_preprocessed_splits",
    "resolve_problem_type",
]
