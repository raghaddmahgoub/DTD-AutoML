"""Row normalization tool."""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool
from sklearn.preprocessing import Normalizer

from tools.shared import ensure_state, merge_state
from .common import load_split, save_split


@tool
def preprocessing_normalization(task, tool_input, prompt, data_path, llm, state=None):
    """Apply optional L1/L2/max normalization to numeric features."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        X_train, X_test, y_train, y_test = load_split(tool_input)
        method = plan.get("normalization", {}).get("method", "none")
        columns = X_train.select_dtypes(include="number").columns.tolist()
        if method in {"l1", "l2", "max"} and columns:
            normalizer = Normalizer(norm=method)
            X_train[columns] = normalizer.fit_transform(X_train[columns])
            X_test[columns] = normalizer.transform(X_test[columns])
        else:
            method = "none"

        paths = save_split(
            X_train,
            X_test,
            y_train,
            y_test,
            tool_input["output_folder"],
            "normalized",
        )
        metadata = {"method": method, "columns": columns if method != "none" else []}
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "normalization_output": metadata,
                "step": "normalization_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "metadata": metadata}, pipeline_state
    except Exception as exc:
        message = f"Normalization failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "normalization_failed", "status": "error", "error": message}
        )
