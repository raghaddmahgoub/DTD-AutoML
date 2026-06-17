"""Small shared utilities used by training tool files."""
from __future__ import annotations

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from tools.nodes.training_engines import train_dask_xgb
from tools.pipeline_state import merge_state
from tools.training_data import load_training_data, prepare_training_xy

APPROACH_SIMPLE = "simple"
APPROACH_SIMPLE_OPTUNA = "simple_optuna"
APPROACH_AUTOGLUON = "autogluon"


def require_approved_plan(pipeline_state: dict[str, Any], expected: str) -> str | None:
    plan = pipeline_state.get("training_plan") or {}
    if not plan.get("approved"):
        return "Training plan not approved. Run plan_training and approve first."
    if plan.get("approach") != expected:
        return (
            f"Plan approach is '{plan.get('approach')}', not '{expected}'. "
            f"Call {plan.get('train_tool')} instead."
        )
    return None


def load_training_context(pipeline_state: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    target = pipeline_state.get("target_column")
    problem_type = pipeline_state.get("problem_type")
    if not target or not problem_type:
        return None, "Missing target_column/problem_type. Run plan_training first."

    plan = pipeline_state.get("training_plan") or {}
    data, use_dask, n_rows = load_training_data(
        pipeline_state["data_path"], pipeline_state
    )
    X, y, is_dask = prepare_training_xy(data, target)
    use_dask = bool(use_dask or is_dask)

    return {
        "target": target,
        "problem_type": problem_type,
        "plan": plan,
        "X": X,
        "y": y,
        "use_dask": use_dask,
        "n_rows": n_rows,
    }, None


def train_test_split_xy(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _prediction_labels(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> np.ndarray:
    if y_pred_np.ndim > 1:
        return y_pred_np.argmax(axis=1)
    if np.issubdtype(y_pred_np.dtype, np.floating):
        unique_pred = np.unique(y_pred_np)
        if len(unique_pred) <= 2 and y_pred_np.min() >= 0.0 and y_pred_np.max() <= 1.0:
            return (y_pred_np > 0.5).astype(int)
    return np.rint(y_pred_np).astype(int)


def apply_test_metrics(metrics: dict, y_test, preds, problem_type: str) -> dict:
    metrics = dict(metrics or {})
    if problem_type == "classification":
        y_test_np = np.asarray(y_test)
        y_pred_np = np.asarray(preds)
        y_pred_labels = _prediction_labels(y_test_np, y_pred_np)

        tuning_best = metrics.get("best_score")
        if tuning_best is not None:
            metrics["tuning_best_score"] = float(tuning_best)

        metrics["test_accuracy"] = float(accuracy_score(y_test_np, y_pred_labels))
        metrics["test_f1_score"] = float(
            f1_score(y_test_np, y_pred_labels, average="weighted")
        )
        metrics["f1_score"] = metrics["test_f1_score"]
        metrics["best_score"] = metrics["test_accuracy"]
        metrics["metric_name"] = "accuracy"
        metrics["confusion_matrix"] = confusion_matrix(
            y_test_np, y_pred_labels
        ).tolist()
    else:
        tuning_best = metrics.get("best_score")
        if tuning_best is not None:
            metrics["tuning_best_score"] = float(tuning_best)
        metrics["rmse"] = float(root_mean_squared_error(y_test, preds))
        metrics["r2_score"] = float(r2_score(y_test, preds))
        metrics["test_r2_score"] = metrics["r2_score"]
        metrics["best_score"] = metrics["r2_score"]
        metrics["metric_name"] = "r2_score"
    return metrics


def save_model_artifact(model: Any, subfolder: str) -> dict[str, str]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = str(Path("output") / "dynamic_pipeline" / stamp / subfolder)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return {"pickle": model_path, "output_dir": out_dir}


def complete_training(
    pipeline_state: dict[str, Any],
    *,
    model: Any,
    metrics: dict,
    training_method: str,
    used_dask: bool,
    n_rows: int,
    subfolder: str = "training",
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = dict(metrics or {})
    metrics["training_method"] = training_method
    metrics["used_dask"] = used_dask
    metrics["n_rows"] = n_rows
    saved_files = save_model_artifact(model, subfolder)

    pipeline_state = merge_state(
        pipeline_state,
        {"step": "model_trained", "status": "success"},
    )
    pipeline_state["model_metrics"] = metrics
    pipeline_state["saved_files"] = saved_files
    result = {
        "status": "success",
        "training_method": metrics.get("training_method"),
        "best_model": metrics.get("best_model"),
        "best_score": metrics.get("best_score"),
        "used_dask": used_dask,
        "saved_files": saved_files,
    }
    if "autogluon_used" in metrics:
        result["autogluon_used"] = metrics["autogluon_used"]
        result["planned_method"] = metrics.get("planned_method")
    if metrics.get("fallback_reason"):
        result["fallback_reason"] = metrics["fallback_reason"]
        result["warning"] = metrics["fallback_reason"]
    return result, pipeline_state


def run_dask_xgb_training(
    pipeline_state: dict[str, Any], ctx: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    problem_type = ctx["problem_type"]
    model, y_test, y_pred, metrics = train_dask_xgb(ctx["X"], ctx["y"], problem_type)
    metrics = apply_test_metrics(metrics, y_test, y_pred, problem_type)
    return complete_training(
        pipeline_state,
        model=model,
        metrics=metrics,
        training_method="Dask-XGBoost (large dataset)",
        used_dask=True,
        n_rows=ctx["n_rows"],
        subfolder="training_dask",
    )
