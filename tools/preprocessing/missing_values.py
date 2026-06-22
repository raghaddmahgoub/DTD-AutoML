"""Missing-value handling tool fitted on training data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from sklearn.impute import KNNImputer

from tools.shared import ensure_state, merge_state
from .common import load_split, save_split, safe_numeric


@tool
def preprocessing_missing_values(task, tool_input, prompt, data_path, llm, state=None):
    """Apply per-column missing-value strategies without overriding user choices."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        X_train, X_test, y_train, y_test = load_split(tool_input)
        actions = {}
        drop_train = pd.Series(False, index=X_train.index)
        drop_test = pd.Series(False, index=X_test.index)
        knn_columns = []

        for column in X_train.columns:
            decision = plan["columns"].get(column, {})
            method = decision.get("missing", "none")
            is_numeric = decision.get("type") == "numeric"
            actions[column] = {"method": method}

            if is_numeric:
                X_train[column] = safe_numeric(X_train[column])
                X_test[column] = safe_numeric(X_test[column])
            if method == "none":
                continue
            if method == "drop_rows":
                drop_train |= X_train[column].isna()
                drop_test |= X_test[column].isna()
                continue
            if method == "knn" and is_numeric:
                knn_columns.append(column)
                continue

            if is_numeric:
                train_numeric = X_train[column]
                test_numeric = X_test[column]
                if method == "mean":
                    fill = float(train_numeric.mean()) if train_numeric.notna().any() else 0.0
                elif method == "constant":
                    fill = 0.0
                else:
                    fill = float(train_numeric.median()) if train_numeric.notna().any() else 0.0
                X_train[column] = train_numeric.fillna(fill)
                X_test[column] = test_numeric.fillna(fill)
            else:
                train_text = X_train[column].astype("string")
                test_text = X_test[column].astype("string")
                if method == "constant":
                    fill = "missing"
                else:
                    modes = train_text.mode(dropna=True)
                    fill = str(modes.iloc[0]) if not modes.empty else "missing"
                X_train[column] = train_text.fillna(fill)
                X_test[column] = test_text.fillna(fill)
            actions[column]["fill_value"] = fill

        if knn_columns:
            neighbors = min(5, max(1, len(X_train) - 1))
            imputer = KNNImputer(n_neighbors=neighbors)
            X_train[knn_columns] = imputer.fit_transform(X_train[knn_columns])
            X_test[knn_columns] = imputer.transform(X_test[knn_columns])

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
            "missing",
        )
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "missing_value_actions": actions,
                "step": "missing_values_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "actions": actions}, pipeline_state
    except Exception as exc:
        message = f"Missing-value handling failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "missing_values_failed", "status": "error", "error": message}
        )
