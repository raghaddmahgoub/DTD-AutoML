"""Shared pipeline state passed between controller and tools."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def empty_state(data_path: str, prompt: str = "") -> dict[str, Any]:
    return {
        "data_path": data_path,
        "prompt": prompt,
        "target_column": None,
        "problem_type": None,
        "report": {},
        "user_preferences": {
            "preferred_models": [],
            "time_preference": "",
            "hw_complexity": "",
            "user_training_prompt": "",
            "ask_before_training": True,
        },
        "training_plan": {
            "approved": False,
            "approach": None,  # simple | simple_optuna | autogluon
            "training_method": None,
            "train_tool": None,
            "use_dask_training": False,
            "selected_models": [],
            "automl_config": {},
            "reasoning": "",
        },
        "model_metrics": {},
        "saved_files": {},
        "X_train_path": None,
        "X_test_path": None,
        "y_train_path": None,
        "y_test_path": None,
        "X_train_engineered_path": None,
        "X_test_engineered_path": None,
        "preprocessing_output": {},
        "feature_engineering_output": {},
        "step": "initialized",
        "status": "running",
    }


def parse_tool_input(tool_input: Any) -> dict[str, Any]:
    if isinstance(tool_input, dict):
        return tool_input
    if isinstance(tool_input, str):
        raw = tool_input.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"note": raw}
    return {}


def ensure_state(state: Any, data_path: str, prompt: str = "") -> dict[str, Any]:
    if isinstance(state, dict) and state.get("data_path"):
        return deepcopy(state)
    return empty_state(data_path, prompt)


def merge_state(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(state)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = deepcopy(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged
