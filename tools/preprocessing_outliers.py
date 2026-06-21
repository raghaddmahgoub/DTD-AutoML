"""Outlier handling tool fitted on training data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state
from tools.preprocessing_common import load_split, save_split, safe_numeric


@tool
def preprocessing_outliers(task, tool_input, prompt, data_path, llm, state=None):
    """Apply per-column numeric outlier strategies using training-derived bounds."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        X_train, X_test, y_train, y_test = load_split(tool_input)
        drop_train = pd.Series(False, index=X_train.index)
        drop_test = pd.Series(False, index=X_test.index)
        actions = {}

        for column in X_train.columns:
            decision = plan["columns"].get(column, {})
            method = decision.get("outlier", "keep")
            if decision.get("type") != "numeric" or method == "keep":
                continue
            train = safe_numeric(X_train[column])
            test = safe_numeric(X_test[column])
            q1, q3 = train.quantile(0.25), train.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            actions[column] = {
                "method": method,
                "lower": float(lower),
                "upper": float(upper),
            }
            if method == "clip_iqr":
                X_train[column] = train.clip(lower, upper)
                X_test[column] = test.clip(lower, upper)
            elif method == "log_transform":
                X_train[column] = np.sign(train) * np.log1p(np.abs(train))
                X_test[column] = np.sign(test) * np.log1p(np.abs(test))
            elif method == "remove_rows":
                drop_train |= (train < lower) | (train > upper)
                drop_test |= (test < lower) | (test > upper)

        if drop_train.any():
            keep = ~drop_train
            X_train, y_train = X_train.loc[keep], y_train.loc[keep]
        if drop_test.any():
            keep = ~drop_test
            X_test, y_test = X_test.loc[keep], y_test.loc[keep]

        paths = save_split(
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            tool_input["output_folder"],
            "outliers",
        )
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "outlier_actions": actions,
                "step": "outliers_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "actions": actions}, pipeline_state
    except Exception as exc:
        message = f"Outlier handling failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "outliers_failed", "status": "error", "error": message}
        )
