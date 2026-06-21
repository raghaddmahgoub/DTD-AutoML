"""Dataset inspection tool for the dynamic preprocessing agent."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state
from tools.preprocessing_common import write_json


@tool
def preprocessing_inspection(task, tool_input, prompt, data_path, llm, state=None):
    """Inspect a CSV and save evidence used to build a preprocessing plan."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        target = tool_input.get("target_column") or pipeline_state.get("target_column")
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        df = pd.read_csv(data_path)
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} is not in the dataset")

        profiles = {}
        for column in df.columns:
            series = df[column]
            numeric = pd.to_numeric(series, errors="coerce")
            non_missing_count = int(series.notna().sum())
            profiles[column] = {
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "unique_ratio": float(series.nunique(dropna=True) / max(len(series), 1)),
                "numeric_parse_ratio": float(
                    numeric.notna().sum() / max(non_missing_count, 1)
                ),
                "sample_values": series.dropna().astype(str).head(5).tolist(),
            }

        y = df[target]
        task_type = (
            "classification"
            if y.dtype == "object" or y.nunique(dropna=True) <= 20
            else "regression"
        )
        counts = y.value_counts(dropna=False)
        evidence = {
            "dataset_path": str(Path(data_path).resolve()),
            "rows": int(len(df)),
            "columns": df.columns.tolist(),
            "target_column": target,
            "task_type": task_type,
            "duplicate_rows": int(df.duplicated().sum()),
            "column_profiles": profiles,
            "target": {
                "missing_count": int(y.isna().sum()),
                "unique_count": int(y.nunique(dropna=True)),
                "class_counts": {str(key): int(value) for key, value in counts.items()},
                "imbalance_ratio": (
                    float(counts.max() / max(counts.min(), 1))
                    if task_type == "classification" and not counts.empty
                    else 1.0
                ),
            },
        }
        output_folder = Path(
            tool_input.get("output_folder")
            or Path("Output") / "Preprocessing" / Path(data_path).stem
        )
        evidence_path = write_json(output_folder / "preprocessing_evidence.json", evidence)
        pipeline_state = merge_state(
            pipeline_state,
            {
                "target_column": target,
                "problem_type": task_type,
                "preprocessing_evidence": evidence,
                "preprocessing_evidence_path": evidence_path,
                "step": "preprocessing_inspected",
                "status": "success",
            },
        )
        return {"status": "success", "evidence": evidence, "evidence_path": evidence_path}, pipeline_state
    except Exception as exc:
        message = f"Dataset inspection failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "preprocessing_inspection_failed", "status": "error", "error": message}
        )
