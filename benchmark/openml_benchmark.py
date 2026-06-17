"""
OpenML Benchmark — Full Pipeline Aligned
=========================================
Benchmarks the AutoML system against OpenML datasets by running the exact
same code path as orchestrator.py (Stages 2, 3 & 4):

  Stage 2 — PreprocessingNode           →  clean train CSV
  Stage 3 — EDAAgent(run_type="clean")  →  automl_directives
  Stage 4 — AutoMLAgent.run()            →  trained model + metrics

The official OpenML train/test split (repeat=0, fold=0) is used:
  • Training split feeds Stages 3 & 4 (EDA + model training).
  • Test split is used for final metric evaluation (accuracy / RMSE).

Environment knobs:
  OPENML_ONLY_TASK=359949        — run only listed task id(s); comma-separated.
  OPENML_AG_TIME_LIMIT=120       — AutoGluon time limit (sec); default 120.
  OPENML_AG_PRESET=good_quality_faster_inference — AutoGluon preset.
  OPENML_SKIP_REFERENCE=1        — skip fetching OpenML reference runs.

Run from repo root:
  python3 benchmark/openml_benchmark.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openml  # noqa: E402
from sklearn.metrics import accuracy_score, root_mean_squared_error  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from agents.static.eda_agent.eda_agent import EDAAgent  # noqa: E402
from agents.static.automl_agent.automl_agent import AutoMLAgent  # noqa: E402
from agents.static.preprocessing_agent.preprocessing_node import preprocessing_node  # noqa: E402

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
OPENML_TASKS: list[tuple[int, str]] = [

    (359949, "house_sales"),
    (211986, "diabetes130us"),
    
]

RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
HISTORY_PATH = RESULTS_DIR / "openml_runs.jsonl"


# ---------------------------------------------------------------------------
# Helpers — task metadata
# ---------------------------------------------------------------------------

def _problem_type(task: openml.tasks.task.OpenMLTask) -> str:
    t = str(task.task_type)
    if "Classification" in t:
        return "classification"
    if "Regression" in t:
        return "regression"
    raise ValueError(f"Unsupported task type: {t}")


def _tasks_to_run() -> list[tuple[int, str]]:
    raw = os.environ.get("OPENML_ONLY_TASK", "").strip()
    if not raw:
        return list(OPENML_TASKS)
    ids: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if part:
            ids.add(int(part))
    return [(tid, n) for tid, n in OPENML_TASKS if tid in ids]


# ---------------------------------------------------------------------------
# Helpers — OpenML data loading
# ---------------------------------------------------------------------------

def _load_openml_split(
    task_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """Return (X_train, X_test, y_train, y_test, task) using the official fold-0 split."""
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=task.target_name, dataset_format="dataframe")
    train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = pd.Series(y.iloc[train_idx].values, index=X_train.index, name=task.target_name)
    y_test = pd.Series(y.iloc[test_idx].values, index=X_test.index, name=task.target_name)
    return X_train, X_test, y_train, y_test, task


# ---------------------------------------------------------------------------
# Helpers — scoring
# ---------------------------------------------------------------------------

def _score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    measure: str,
    problem_type: str,
) -> dict[str, float]:
    if measure == "predictive_accuracy" or (
        problem_type == "classification" and measure != "root_mean_squared_error"
    ):
        return {"predictive_accuracy": float(accuracy_score(y_true, y_pred))}
    if measure == "root_mean_squared_error":
        return {"root_mean_squared_error": float(root_mean_squared_error(y_true, y_pred))}
    # Fallbacks
    if problem_type == "classification":
        return {"predictive_accuracy": float(accuracy_score(y_true, y_pred))}
    return {"root_mean_squared_error": float(root_mean_squared_error(y_true, y_pred))}


# ---------------------------------------------------------------------------
# Helpers — OpenML reference (unchanged from original)
# ---------------------------------------------------------------------------

def _fetch_openml_reference(task_id: int, measure: str, limit: int = 20) -> list[dict[str, Any]]:
    if os.environ.get("OPENML_SKIP_REFERENCE", "").lower() in ("1", "true", "yes"):
        return []
    try:
        evs = openml.evaluations.list_evaluations(
            function=measure,
            tasks=[task_id],
            size=limit * 3,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for ev in evs.values():
        rows.append(
            {
                "flow_name": getattr(ev, "flow_name", None),
                "value": float(ev.value) if ev.value is not None else None,
                "run_id": getattr(ev, "run_id", None),
                "upload_time": str(getattr(ev, "upload_time", "")),
            }
        )
    higher_better = measure == "predictive_accuracy"
    rows = [r for r in rows if r["value"] is not None and not np.isnan(r["value"])]
    if measure == "root_mean_squared_error":
        rows = [r for r in rows if float(r["value"]) > 0.0]
    rows.sort(key=lambda r: r["value"], reverse=higher_better)
    return rows[:limit]


# ---------------------------------------------------------------------------
# Core: run one OpenML task through the full pipeline (Stages 3 + 4)
# ---------------------------------------------------------------------------

def _run_task_through_pipeline(
    task_id: int,
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
    target_name: str,
    measure: str,
) -> dict[str, Any]:

    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix=f"openml_{task_id}_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # ── 1. Preprocess ONLY the train split (Stage 2) ──
        print(f"  🛠️  [Stage 2] Running PreprocessingNode on train split only...")

        raw_train_df = X_train.copy()
        raw_train_df[target_name] = y_train.values
        raw_train_csv = tmpdir_path / "raw_train.csv"
        raw_train_df.to_csv(raw_train_csv, index=False)

        preprocessing_output_dir = str(tmpdir_path / "preprocessing_output")
        prep_state = preprocessing_node({
            "dataset_path":  str(raw_train_csv),
            "target_column": target_name,
            "output_folder": preprocessing_output_dir,
            "use_llm":       True,
        })

        if prep_state.get("status") != "success":
            raise RuntimeError(f"PreprocessingNode failed: {prep_state.get('error')}")

        # ── 2. Load the preprocessed train splits (Stage 3 & 4 Handoff) ──
        X_train_clean_path = prep_state["X_train_path"]
        X_test_clean_path  = prep_state["X_test_path"]
        y_train_clean_path = prep_state["y_train_path"]
        y_test_clean_path  = prep_state["y_test_path"]

        X_train_clean = pd.read_csv(X_train_clean_path)
        y_train_clean = pd.read_csv(y_train_clean_path).squeeze("columns")

        # Build clean train CSV for EDA
        clean_train_df = X_train_clean.copy()
        clean_train_df[target_name] = y_train_clean.values
        clean_train_csv = tmpdir_path / "train_clean.csv"
        clean_train_df.to_csv(clean_train_csv, index=False)

        print(f"  ✅ Preprocessing complete. Cleaned Train: {X_train_clean.shape}")

        # ── 3. Stage 3 — EDA on clean training data ──
        print(f"  📊 [Stage 3] Running EDA on clean training data...")
        eda_output_dir = str(tmpdir_path / "eda_output")

        eda_agent = EDAAgent(
            df=clean_train_df,
            target_column=target_name,
            df_name=f"openml_{task_id}_{name}",
        )
        eda_agent.run(run_type="clean")

        automl_directives = eda_agent.generate_automl_context(
            plan_dir=eda_output_dir,
            output_dir=eda_output_dir,
        )

        # Standardizing directives structure
        directives = automl_directives or {}
        if "report" not in directives or directives["report"] is None:
            directives["report"] = {
                "target_analysis": {"column": target_name, "skew_severity": "N/A"}
            }

        # ── 4. Stage 4 — AutoMLAgent.run() ──
        print(f"  🤖 [Stage 4] Running AutoMLAgent LangGraph workflow...")

        automl_output_dir = str(tmpdir_path / "automl_output")
        automl_agent = AutoMLAgent()

        # Set env variables for AutoGluon if needed
        ag_time_limit = int(os.environ.get("OPENML_AG_TIME_LIMIT", "120"))
        ag_preset     = os.environ.get("OPENML_AG_PRESET", "good_quality_faster_inference")
        os.environ["OPENML_AG_TIME_LIMIT"] = str(ag_time_limit)
        os.environ["OPENML_AG_PRESET"]     = ag_preset

        final_state = automl_agent.run(
            data_path=str(clean_train_csv),
            target_column=target_name,
            output_dir=automl_output_dir,
            automl_directives=directives,
            problem_type=problem_type,
            X_train_path=X_train_clean_path,
            X_test_path=X_test_clean_path,
            y_train_path=y_train_clean_path,
            y_test_path=y_test_clean_path,
        )

        if final_state.get("error"):
            raise RuntimeError(f"AutoMLAgent failed: {final_state['error']}")

        trained_model = final_state.get("trained_model")
        fit_s = round(time.time() - t0, 2)

        # ── 5. Evaluate on the PREPROCESSED test split ──
        # CRITICAL: We evaluate using the preprocessed output to align with pipeline transformations
        print(f"  🔬 Evaluating on preprocessed test split...")

        X_test_final = pd.read_csv(X_test_clean_path)
        y_true       = pd.read_csv(y_test_clean_path).squeeze("columns").values
        
        use_automl = final_state.get("use_automl", False)

        if use_automl:
            # AutoGluon handles internal alignment automatically
            preds = trained_model.predict(X_test_final)
        else:
            # sklearn/simple models — align columns exactly to training schema
            if hasattr(trained_model, "feature_names_in_"):
                seen_cols = list(trained_model.feature_names_in_)
                X_test_final = X_test_final.reindex(columns=seen_cols, fill_value=0)
            preds = trained_model.predict(X_test_final)

        # ── 6. Score using aligned numeric values ──
        # No re-encoding needed because y_true and preds already share the same mapping
        preds_arr = np.asarray(preds)
        if problem_type == "classification":
            preds_arr = preds_arr.astype(np.int64)
            
        scores = _score(y_true, preds_arr, measure, problem_type)

        # ── 7. Collect result metadata ──
        raw_metrics = final_state.get("model_metrics") or {}
        return {
            "training_method": "AutoGluon" if use_automl else raw_metrics.get("training_method", "sklearn"),
            "fit_wall_time_sec": fit_s,
            "model_selection_reasoning_preview": (final_state.get("model_selection_reasoning") or "")[:400],
            "use_automl":      use_automl,
            "automl_config":   final_state.get("automl_config"),
            "selected_models": final_state.get("selected_models"),
            "train_metrics_from_agent": {
                k: float(v)
                for k, v in raw_metrics.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            },
            "test_scores":       scores,
            "measure_reported":  measure,
            "n_train":           int(len(X_train_clean)),
            "n_test":            int(len(X_test_final)),
            "n_features_raw":    int(X_train.shape[1]),
        }
# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_openml_benchmark() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    bundle_tasks: list[dict[str, Any]] = []
    tasks = _tasks_to_run()
    if not tasks:
        raise ValueError("OPENML_ONLY_TASK did not match any known task ids in OPENML_TASKS")

    for task_id, name in tasks:
        print(f"\n=== OpenML task {task_id} ({name}) ===")
        t0 = time.time()
        record: dict[str, Any] = {
            "task_id": task_id,
            "name": name,
            "openml_task_url": f"https://www.openml.org/t/{task_id}",
        }
        try:
            print(f"  ⬇️  Loading OpenML split for task {task_id}...")
            X_train, X_test, y_train, y_test, task = _load_openml_split(task_id)
            problem_type = _problem_type(task)
            measure = str(task.evaluation_measure)
            target_name = task.target_name

            print(
                f"  ℹ️  problem={problem_type}, measure={measure}, "
                f"target={target_name}, train={len(X_train)}, test={len(X_test)}"
            )

            eval_payload = _run_task_through_pipeline(
                task_id=task_id,
                name=name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                problem_type=problem_type,
                target_name=target_name,
                measure=measure,
            )

            record["status"] = "success"
            record["task_type"] = str(task.task_type)
            record["target_name"] = target_name
            record["evaluation_measure_openml"] = measure
            record.update(eval_payload)
            record["openml_reference_runs"] = _fetch_openml_reference(task_id, measure)
            record["wall_time_sec"] = round(time.time() - t0, 2)

            print(
                f"  ✅ Task {task_id} done. "
                f"Scores: {eval_payload['test_scores']} "
                f"({eval_payload['training_method']}, {eval_payload['fit_wall_time_sec']}s)"
            )

        except Exception as e:
            record["status"] = "failed"
            record["error"] = str(e)
            record["wall_time_sec"] = round(time.time() - t0, 2)
            warnings.warn(f"Task {task_id} failed: {e}", stacklevel=1)
            print(f"  ❌ Task {task_id} FAILED: {e}")

        bundle_tasks.append(record)

    bundle = {
        "run_id": run_id,
        "created_at_utc": run_id,
        "pipeline": "full_orchestrator_stages_3_and_4",
        "config": {
            "OPENML_AG_TIME_LIMIT": os.environ.get("OPENML_AG_TIME_LIMIT", "120"),
            "OPENML_AG_PRESET": os.environ.get("OPENML_AG_PRESET", "good_quality_faster_inference"),
            "OPENML_ONLY_TASK": os.environ.get("OPENML_ONLY_TASK", ""),
        },
        "tasks": bundle_tasks,
    }

    out_path = RESULTS_DIR / f"openml_benchmark_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(bundle, separators=(",", ":"), default=str) + "\n")

    latest = RESULTS_DIR / "openml_benchmark_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)

    print(f"\n✅ OpenML benchmark finished. Latest → {latest}")
    print(f"   This run → {out_path}")
    return bundle


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_openml_benchmark()
