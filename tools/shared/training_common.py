"""Shared training workflow: load preprocessed splits, train helpers, metrics."""
from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score, root_mean_squared_error

from state.pipeline_state import merge_state

LARGE_DATA_ROW_THRESHOLD = 700_000

APPROACH_SIMPLE = "simple"
APPROACH_SIMPLE_OPTUNA = "simple_optuna"
APPROACH_AUTOGLUON = "autogluon"


def resolve_preprocessed_paths(pipeline_state: dict[str, Any]) -> dict[str, str] | None:
    """Return train/test CSV paths from preprocessing agent output (engineered preferred)."""
    fe_out = pipeline_state.get("feature_engineering_output") or {}
    prep_out = pipeline_state.get("preprocessing_output") or {}

    eng_train = (
        pipeline_state.get("X_train_engineered_path")
        or fe_out.get("X_train_engineered_path")
    )
    eng_test = (
        pipeline_state.get("X_test_engineered_path")
        or fe_out.get("X_test_engineered_path")
    )
    if eng_train and eng_test and Path(eng_train).exists() and Path(eng_test).exists():
        x_train_path, x_test_path = eng_train, eng_test
    else:
        x_train_path = pipeline_state.get("X_train_path") or prep_out.get("X_train_path")
        x_test_path = pipeline_state.get("X_test_path") or prep_out.get("X_test_path")

    y_train_path = pipeline_state.get("y_train_path") or prep_out.get("y_train_path")
    y_test_path = pipeline_state.get("y_test_path") or prep_out.get("y_test_path")

    paths = {
        "X_train_path": x_train_path,
        "X_test_path": x_test_path,
        "y_train_path": y_train_path,
        "y_test_path": y_test_path,
    }
    if not all(paths.values()) or not all(Path(p).exists() for p in paths.values()):
        return None
    return paths


def require_preprocessed_splits(pipeline_state: dict[str, Any]) -> str | None:
    if resolve_preprocessed_paths(pipeline_state):
        return None
    return (
        "Preprocessed train/test splits not found. "
        "Run PreprocessingAgent (preprocessing_execution + feature_engineering_execution) "
        "before plan_training or train_* tools."
    )


def load_preprocessed_splits(
    pipeline_state: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    paths = resolve_preprocessed_paths(pipeline_state)
    if not paths:
        raise FileNotFoundError(require_preprocessed_splits(pipeline_state))

    X_train = pd.read_csv(paths["X_train_path"])
    X_test = pd.read_csv(paths["X_test_path"])
    y_train = pd.read_csv(paths["y_train_path"]).iloc[:, 0]
    y_test = pd.read_csv(paths["y_test_path"]).iloc[:, 0]
    n_rows = len(X_train) + len(X_test)
    return X_train, X_test, y_train, y_test, n_rows


def resolve_problem_type(pipeline_state: dict[str, Any]) -> str | None:
    if pipeline_state.get("problem_type"):
        return pipeline_state["problem_type"]
    if pipeline_state.get("task_type") in ("classification", "regression"):
        return pipeline_state["task_type"]

    summary_path = (pipeline_state.get("preprocessing_output") or {}).get("summary_path")
    if summary_path and Path(summary_path).exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        task_type = summary.get("task_type")
        if task_type in ("classification", "regression"):
            return task_type
    return None


def load_planning_dataframe(pipeline_state: dict[str, Any]) -> pd.DataFrame:
    """Build a train-set frame (features + target) for plan subgraph profiling."""
    paths = resolve_preprocessed_paths(pipeline_state)
    if not paths:
        raise FileNotFoundError(require_preprocessed_splits(pipeline_state))

    target_column = pipeline_state.get("target_column")
    if not target_column:
        raise ValueError("target_column missing from pipeline_state after preprocessing.")

    X_train = pd.read_csv(paths["X_train_path"])
    y_train = pd.read_csv(paths["y_train_path"]).iloc[:, 0]
    df = X_train.copy()
    df[target_column] = y_train.values
    return df


def pipeline_to_graph_state(pipeline_state: dict[str, Any]) -> dict[str, Any]:
    err = require_preprocessed_splits(pipeline_state)
    if err:
        raise ValueError(err)

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

    data = load_planning_dataframe(pipeline_state)
    n_rows = len(data)
    if not report.get("dataset_summary"):
        report["dataset_summary"] = {
            "n_rows": n_rows,
            "n_columns": len(data.columns),
        }

    problem_type = resolve_problem_type(pipeline_state)

    return {
        "data_path": pipeline_state["data_path"],
        "target_column": pipeline_state.get("target_column"),
        "problem_type": problem_type,
        "data": data,
        "use_dask": False,
        "use_automl": False,
        "automl_config": {},
        "selected_models": [],
        "optuna_config": {},
        "llm_approach": "",
        "model_selection_reasoning": "",
        "report": report,
        "automl_directives": {
            "report": report,
            "task_type": problem_type,
            "user": {
                "task_prompt": str(pipeline_state.get("prompt", ""))[:1200],
                "controller_task": str(pipeline_state.get("controller_task") or "")[:1200],
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
    if err := require_preprocessed_splits(pipeline_state):
        return None, err

    target = pipeline_state.get("target_column")
    problem_type = resolve_problem_type(pipeline_state)
    if not target:
        return None, "Missing target_column. Run preprocessing first."
    if not problem_type:
        return None, "Missing problem_type. Set it in plan_training or ensure preprocessing_summary has task_type."

    try:
        X_train, X_test, y_train, y_test, n_rows = load_preprocessed_splits(pipeline_state)
    except FileNotFoundError as exc:
        return None, str(exc)

    plan = pipeline_state.get("training_plan") or {}
    return {
        "target": target,
        "problem_type": problem_type,
        "plan": plan,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "n_rows": n_rows,
    }, None


# def _prediction_labels(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> np.ndarray:
#     print("y_pred dtype:", y_pred_np.dtype)
#     print("sample preds:", y_pred_np[:10])
#     if y_pred_np.ndim > 1:
#         return y_pred_np.argmax(axis=1)
#     if np.issubdtype(y_pred_np.dtype, np.floating):
#         unique_pred = np.unique(y_pred_np)
#         if len(unique_pred) <= 2 and y_pred_np.min() >= 0.0 and y_pred_np.max() <= 1.0:
#             return (y_pred_np > 0.5).astype(int)
#     return np.rint(y_pred_np).astype(int)
def _prediction_labels(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray
) -> np.ndarray:
    # multiclass probabilities
    if y_pred_np.ndim > 1:
        return y_pred_np.argmax(axis=1)

    # already labels (strings/categories)
    if y_pred_np.dtype.kind in {"O", "U", "S"}:
        return y_pred_np

    # probabilities
    if np.issubdtype(y_pred_np.dtype, np.floating):
        unique_pred = np.unique(y_pred_np)

        if (
            len(unique_pred) <= 2
            and y_pred_np.min() >= 0
            and y_pred_np.max() <= 1
        ):
            return (y_pred_np > 0.5).astype(int)

        return np.rint(y_pred_np).astype(int)

    # integer labels
    return y_pred_np.astype(int)

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
    n_rows: int,
    subfolder: str = "training",
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = dict(metrics or {})
    metrics["training_method"] = training_method

    # Extract feature importance for sklearn models (AutoGluon sets this in the engine)
    if "feature_importance" not in metrics:
        try:
            fi = None
            if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
                fi = dict(sorted(
                    zip(model.feature_names_in_, model.feature_importances_.tolist()),
                    key=lambda x: x[1], reverse=True,
                ))
            elif hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
                coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                fi = dict(sorted(
                    zip(model.feature_names_in_, np.abs(coef).tolist()),
                    key=lambda x: x[1], reverse=True,
                ))
            if fi:
                metrics["feature_importance"] = fi
        except Exception:
            pass

    saved_files = save_model_artifact(model, subfolder)

    pipeline_state = merge_state(
        pipeline_state,
        {"step": "model_trained", "status": "success"},
    )
    pipeline_state["model_metrics"] = metrics
    pipeline_state["saved_files"] = saved_files

    results_paths = save_automl_results_json(pipeline_state)
    saved_files = {**saved_files, **results_paths}
    pipeline_state["saved_files"] = saved_files
    pipeline_state["results_json"] = results_paths.get("json")

    result = {
        "status": "success",
        "training_method": metrics.get("training_method"),
        "best_model": metrics.get("best_model"),
        "best_score": metrics.get("best_score"),
        "used_dask": False,
        "saved_files": saved_files,
        "results_json": results_paths.get("json"),
    }
    if "autogluon_used" in metrics:
        result["autogluon_used"] = metrics["autogluon_used"]
        result["planned_method"] = metrics.get("planned_method")
    if metrics.get("fallback_reason"):
        result["fallback_reason"] = metrics["fallback_reason"]
        result["warning"] = metrics["fallback_reason"]
    return result, pipeline_state


def build_automl_results_payload(pipeline_state: dict[str, Any], *, timestamp: str | None = None) -> dict[str, Any]:
    """Build results JSON payload matching static AutoMLAgent output structure."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = dict(pipeline_state.get("model_metrics") or {})
    plan = pipeline_state.get("training_plan") or {}
    use_automl = plan.get("approach") == APPROACH_AUTOGLUON

    # Normalize automl_config key names (LLM may produce time_limit_seconds, preset_mode, etc.)
    raw_cfg = plan.get("automl_config") or {}
    automl_config = {
        "models": raw_cfg.get("models") or raw_cfg.get("models_to_prioritize") or [],
        "time_limit": raw_cfg.get("time_limit") or raw_cfg.get("time_limit_seconds"),
        "preset": raw_cfg.get("preset") or raw_cfg.get("preset_mode"),
    }

    reasoning = plan.get("reasoning") or pipeline_state.get("model_selection_reasoning")

    best_score = metrics.get("best_score")
    score_str = f"{best_score:.4f}" if isinstance(best_score, float) else str(best_score or "")

    training_results = {
        "best_model": metrics.get("best_model"),
        "best_score": best_score,
        "feature_importance": metrics.get("feature_importance"),
        "models_trained": metrics.get("models_trained"),
        "all_models": metrics.get("all_models", []),
        "all_scores": metrics.get("all_scores", []),
        "training_method": metrics.get("training_method"),
        "f1_score": metrics.get("f1_score") or metrics.get("test_f1_score"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }

    return {
        "run_timestamp": stamp,
        "data_path": pipeline_state.get("data_path"),
        "target_column": pipeline_state.get("target_column"),
        "problem_type": pipeline_state.get("problem_type") or pipeline_state.get("task_type"),
        "model_selection": {
            "use_automl": use_automl,
            "automl_config": automl_config,
            "selected_models": plan.get("selected_models") or [],
            "model_selection_reasoning": reasoning,
        },
        "training_results": training_results,
        "agent_messages": [
            {
                "agent": "training",
                "message": f"Training complete. Training completed. Best score: {score_str}",
            }
        ],
        "workflow": {
            "final_step": pipeline_state.get("step"),
            "error": pipeline_state.get("error"),
        },
    }


def save_automl_results_json(
    pipeline_state: dict[str, Any],
    output_dir: str = "output/automl",
) -> dict[str, str]:
    """
    Save training outputs in the same JSON layout as static AutoMLAgent:
    output/automl/results_{timestamp}.json
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = build_automl_results_payload(pipeline_state, timestamp=timestamp)

    json_path = out_dir / f"results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    saved = {"json": str(json_path), "output_dir": str(out_dir)}

    model_path = (pipeline_state.get("saved_files") or {}).get("pickle")
    if model_path and Path(model_path).exists():
        dest = out_dir / f"best_model_{timestamp}.pkl"
        dest.write_bytes(Path(model_path).read_bytes())
        saved["pickle"] = str(dest)

    print(f"[save_automl_results] JSON -> {json_path}")
    return saved
