"""LangGraph-compatible state for tools pipeline nodes."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class TrainingGraphState(TypedDict, total=False):
    data_path: str
    data: Any
    target_column: Optional[str]
    problem_type: Optional[str]
    report: dict
    automl_directives: dict
    use_dask: bool
    use_automl: bool
    automl_config: dict
    selected_models: list[str]
    optuna_config: dict
    llm_approach: str
    model_selection_reasoning: str
    error: Optional[str]
    step: str
