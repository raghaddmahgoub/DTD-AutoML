"""Numerical scaling tool fitted on training data."""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from tools.pipeline_state import ensure_state, merge_state
from tools.preprocessing_common import load_split, save_split


@tool
def preprocessing_scaling(task, tool_input, prompt, data_path, llm, state=None):
    """Scale selected numeric columns and preserve explicitly excluded columns."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        X_train, X_test, y_train, y_test = load_split(tool_input)
        method = plan.get("scaling", {}).get("method", "none")
        skip = {
            column
            for column, decision in plan.get("columns", {}).items()
            if decision.get("skip_scaling")
        }
        columns = [
            column
            for column in X_train.columns
            if pd.api.types.is_numeric_dtype(X_train[column]) and column not in skip
        ]
        scaler = None
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "quantile":
            scaler = QuantileTransformer(
                n_quantiles=min(1000, max(1, len(X_train))),
                output_distribution="normal",
                random_state=42,
            )
        elif method == "power":
            scaler = PowerTransformer()

        warning = ""
        if scaler is not None and columns:
            try:
                X_train[columns] = scaler.fit_transform(X_train[columns])
                X_test[columns] = scaler.transform(X_test[columns])
            except Exception as exc:
                warning = f"Scaling was skipped because {method} failed: {exc}"
                method = "none"

        paths = save_split(
            X_train,
            X_test,
            y_train,
            y_test,
            tool_input["output_folder"],
            "scaled",
        )
        metadata = {
            "method": method,
            "columns": columns if method != "none" else [],
            "excluded_by_user": sorted(skip),
            "warning": warning,
        }
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "scaling_output": metadata,
                "step": "scaling_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "metadata": metadata}, pipeline_state
    except Exception as exc:
        message = f"Scaling failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "scaling_failed", "status": "error", "error": message}
        )
