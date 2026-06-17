"""Data loading and feature prep for tools (standalone — no AutoMLAgent)."""
from __future__ import annotations

from typing import Any

import pandas as pd

LARGE_DATA_ROW_THRESHOLD = 700_000


def _estimate_row_count(data_path: str, report: dict | None = None) -> int:
    if report:
        summary = report.get("dataset_summary") or {}
        if summary.get("n_rows"):
            return int(summary["n_rows"])
        shape = summary.get("shape")
        if isinstance(shape, (list, tuple)) and shape:
            return int(shape[0])
    if data_path.lower().endswith(".csv"):
        with open(data_path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    return len(load_dataframe(data_path))


def load_dataframe(data_path: str) -> pd.DataFrame:
    lower = data_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(data_path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(data_path)
    if lower.endswith(".json"):
        return pd.read_json(data_path)
    raise ValueError(f"Unsupported file format: {data_path}")


def load_training_data(
    data_path: str, pipeline_state: dict[str, Any] | None = None
) -> tuple[Any, bool, int]:
    import dask.dataframe as dd

    report = (pipeline_state or {}).get("report") or {}
    n_rows = _estimate_row_count(data_path, report)

    if n_rows > LARGE_DATA_ROW_THRESHOLD and data_path.lower().endswith(".csv"):
        return dd.read_csv(data_path, blocksize="256MB", assume_missing=True), True, n_rows
    if n_rows > LARGE_DATA_ROW_THRESHOLD and data_path.lower().endswith(".json"):
        return dd.read_json(data_path, blocksize="256MB", lines=True), True, n_rows
    return load_dataframe(data_path), False, n_rows


def prepare_training_xy(data: Any, target_column: str) -> tuple[Any, Any, bool]:
    import dask.dataframe as dd

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if isinstance(data, dd.DataFrame):

        def convert_numeric(df):
            cat = df.select_dtypes(include=["object", "category"]).columns
            out = df.copy()
            for col in df.columns:
                if col not in cat:
                    out[col] = pd.to_numeric(df[col], errors="coerce")
            return out

        X = X.map_partitions(convert_numeric, meta=X)
        y = y.map_partitions(lambda s: s.astype(float), meta=y)
        frame = dd.concat([X, y.rename("target")], axis=1).dropna(subset=["target"])
        X = frame.drop(columns=["target"])
        y = frame["target"]
        for col in X.columns:
            if str(X[col].dtype) in ("object", "category"):
                X[col] = X[col].astype("category").cat.codes
        return X, y, True

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in X.columns:
        if col not in cat_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    frame = pd.concat([X, y.rename("target")], axis=1).dropna(subset=["target"])
    X = frame.drop(columns=["target"])
    y = frame["target"]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if cat_cols:
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(drop="first", sparse_output=False)
        X_cat = pd.DataFrame(
            encoder.fit_transform(X[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X.index,
        )
        X = pd.concat([X[num_cols], X_cat], axis=1) if num_cols else X_cat
    return X, y, False


def pipeline_to_graph_state(pipeline_state: dict[str, Any]) -> dict[str, Any]:
    report = pipeline_state.get("report") or {}
    if isinstance(report, dict) and "report" in report:
        report = report["report"]
    report = {
        **report,
        "dataset_summary": report.get("dataset_summary") or {},
        "target_analysis": report.get("target_analysis") or {},
        "data_quality_report": report.get("data_quality_report")
        or {"duplicates": {"duplicate_ratio": 0.0}},
        "multicollinearity": report.get("multicollinearity") or {"pairs": []},
        "encoding_hints": report.get("encoding_hints") or {},
        "signal_analysis": report.get("signal_analysis") or {},
    }
    data, use_dask, n_rows = load_training_data(
        pipeline_state["data_path"], pipeline_state
    )
    if not report.get("dataset_summary"):
        report["dataset_summary"] = {
            "n_rows": n_rows,
            "n_columns": len(data.columns) if hasattr(data, "columns") else 0,
        }
    return {
        "data_path": pipeline_state["data_path"],
        "target_column": pipeline_state.get("target_column"),
        "problem_type": pipeline_state.get("problem_type"),
        "data": data,
        "use_dask": use_dask,
        "use_automl": False,
        "automl_config": {},
        "selected_models": [],
        "optuna_config": {},
        "llm_approach": "",
        "model_selection_reasoning": "",
        "report": report,
        "automl_directives": {
            "report": report,
            "task_type": pipeline_state.get("problem_type"),
            "user": {
                "task_prompt": str(pipeline_state.get("prompt", ""))[:500],
                "training_note": (pipeline_state.get("user_preferences") or {}).get(
                    "user_training_prompt", ""
                )[:300],
                "time_preference": (pipeline_state.get("user_preferences") or {}).get(
                    "time_preference", ""
                ),
                "hw_complexity": (pipeline_state.get("user_preferences") or {}).get(
                    "hw_complexity", ""
                ),
                "preferred_models": (pipeline_state.get("user_preferences") or {}).get(
                    "preferred_models", []
                ),
            },
        },
        "step": "initialized",
    }
