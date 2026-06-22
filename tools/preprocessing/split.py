"""Train/test split and dataset preparation tool."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from sklearn.model_selection import train_test_split

from tools.shared import ensure_state, merge_state
from .common import save_split


@tool
def preprocessing_split(task, tool_input, prompt, data_path, llm, state=None):
    """Apply approved drops/duplicate handling and create the train/test split."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        target = tool_input.get("target_column") or pipeline_state["target_column"]
        output_folder = tool_input["output_folder"]
        test_size = float(tool_input.get("test_size", 0.2))
        random_state = int(tool_input.get("random_state", 42))

        df = pd.read_csv(data_path)
        original_rows = len(df)
        duplicates_removed = 0
        if plan.get("duplicates") == "drop":
            duplicates_removed = int(df.duplicated().sum())
            df = df.drop_duplicates().copy()

        target_missing = int(df[target].isna().sum())
        if target_missing:
            df = df[df[target].notna()].copy()

        dropped = [
            column
            for column, decision in plan["columns"].items()
            if decision.get("drop") and column in df.columns
        ]
        X = df.drop(columns=[target, *dropped])
        y = df[target]
        stratify = None
        if pipeline_state.get("problem_type") == "classification":
            counts = y.value_counts()
            if len(counts) > 1 and int(counts.min()) >= 2:
                stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        paths = save_split(
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            output_folder,
            "split",
        )
        metadata = {
            "original_rows": original_rows,
            "duplicates_removed": duplicates_removed,
            "target_missing_rows_removed": target_missing,
            "dropped_columns": dropped,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "preprocessing_metadata": metadata,
                "step": "preprocessing_split_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "metadata": metadata}, pipeline_state
    except Exception as exc:
        message = f"Data split failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "preprocessing_split_failed", "status": "error", "error": message}
        )
