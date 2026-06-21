"""Model-readiness validation and final preprocessing output tool."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state
from tools.preprocessing_common import load_split, write_json


@tool
def preprocessing_validation(task, tool_input, prompt, data_path, llm, state=None):
    """Save final outputs and report blockers without overriding user choices."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        X_train, X_test, y_train, y_test = load_split(tool_input)
        output_folder = Path(tool_input["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)

        # Remove legacy intermediate CSVs produced by earlier agent versions.
        final_dataset_names = {
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
        }
        for csv_path in output_folder.glob("*.csv"):
            if csv_path.name not in final_dataset_names:
                csv_path.unlink()

        non_numeric = [
            column
            for column in X_train.columns
            if not pd.api.types.is_numeric_dtype(X_train[column])
        ]
        missing = {
            column: int(X_train[column].isna().sum() + X_test[column].isna().sum())
            for column in X_train.columns
            if X_train[column].isna().any() or X_test[column].isna().any()
        }
        infinite = []
        for column in X_train.select_dtypes(include="number").columns:
            if np.isinf(X_train[column]).any() or np.isinf(X_test[column]).any():
                infinite.append(column)

        blockers = []
        if non_numeric:
            blockers.append(
                "Non-numeric feature columns remain: " + ", ".join(non_numeric)
            )
        if missing:
            blockers.append(
                "Missing values remain in: " + ", ".join(sorted(missing))
            )
        if infinite:
            blockers.append(
                "Infinite values remain in: " + ", ".join(infinite)
            )
        if X_train.empty or X_train.shape[1] == 0:
            blockers.append("No training features remain.")

        final_paths = {
            "X_train_path": str(output_folder / "X_train.csv"),
            "X_test_path": str(output_folder / "X_test.csv"),
            "y_train_path": str(output_folder / "y_train.csv"),
            "y_test_path": str(output_folder / "y_test.csv"),
        }
        for source_key, destination in final_paths.items():
            shutil.copyfile(tool_input[source_key], destination)

        readiness = {
            "modeling_ready": not blockers,
            "blockers": blockers,
            "warnings": list(
                (pipeline_state.get("preprocessing_plan") or {}).get("warnings", [])
            ),
            "non_numeric_columns": non_numeric,
            "missing_values": missing,
            "infinite_columns": infinite,
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
        }
        readiness_path = write_json(
            output_folder / "modeling_readiness.json", readiness
        )
        summary = {
            "dataset": str(Path(data_path).name),
            "target_column": pipeline_state.get("target_column"),
            "task_type": pipeline_state.get("problem_type"),
            "modeling_ready": readiness["modeling_ready"],
            "blockers": blockers,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "plan_path": pipeline_state.get("preprocessing_plan_path"),
            "readiness_path": readiness_path,
            "preparation": pipeline_state.get("preprocessing_metadata", {}),
            "missing_values": pipeline_state.get("missing_value_actions", {}),
            "outliers": pipeline_state.get("outlier_actions", {}),
            "encoding": pipeline_state.get("encoding_actions", {}),
            "scaling": pipeline_state.get("scaling_output", {}),
            "normalization": pipeline_state.get("normalization_output", {}),
            "balancing": pipeline_state.get("balancing_output", {}),
        }
        summary_path = write_json(
            output_folder / "preprocessing_summary.json", summary
        )
        preprocessing_output = {
            **final_paths,
            "summary_path": summary_path,
            "readiness_path": readiness_path,
            "plan_path": pipeline_state.get("preprocessing_plan_path"),
            "modeling_ready": readiness["modeling_ready"],
            "blockers": blockers,
        }
        pipeline_state = merge_state(
            pipeline_state,
            {
                **final_paths,
                "preprocessing_output": preprocessing_output,
                "modeling_ready": readiness["modeling_ready"],
                "modeling_blockers": blockers,
                "step": "preprocessing_complete",
                "status": "success",
            },
        )
        return {
            "status": "success",
            "preprocessing_output": preprocessing_output,
            "readiness": readiness,
        }, pipeline_state
    except Exception as exc:
        message = f"Model-readiness validation failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state,
            {"step": "preprocessing_validation_failed", "status": "error", "error": message},
        )
