"""
End-to-end runner
- Imports local automl_preprocessing_search
- Creates plan/ and output/ directories
- Exports preprocessed TRAIN/TEST/FULL datasets as CSV (forced)
- Saves best pipeline to plan/best_pipeline.pkl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import joblib

from agents.static.preprocessing_agent.automl_preprocessing_search import automl_preprocessing_search  # local import


# ================== CONFIG ==================
DATASET_NAME = "Titanic-Dataset"   
TARGETS = {
    "adult_census": "income",
    "breast_cancer": "diagnosis",
    "titanic": "2urvived",
    "Titanic-Dataset":"Survived"
}
TARGET_COLUMN = TARGETS[DATASET_NAME]

DATASET_PATHS = {
    "adult_census": "Datasets/adult_census.csv",
    "breast_cancer": "Datasets/Breast_cancer.csv",
    "titanic": "Datasets/titanic.csv",
    "Titanic-Dataset":"Datasets/Titanic-Dataset.csv"
}

# Treat literal '?' as missing values (common in Adult Census)
NA_VALUES = ["?"]

PLAN_DIR = Path("Output/static/Plan")
OUTPUT_DIR = Path("Output/static/Preprocessing")

TASK_JSON = PLAN_DIR / "task_definition.json"
REPORT_JSON = PLAN_DIR / "final_report.json"
BEST_PIPELINE_PATH = PLAN_DIR / "best_pipeline.pkl"
# ===========================================


def _heuristic_task(df: pd.DataFrame, target: str) -> Dict[str, str]:
    s = df[target]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 30:
        return {"task_type": "regression", "metric": "r2"}
    return {"task_type": "classification", "metric": "accuracy"}


def load_task_definition(df: pd.DataFrame) -> Dict[str, str]:
    if TASK_JSON.exists():
        with open(TASK_JSON, "r") as f:
            return json.load(f)
    return _heuristic_task(df, TARGET_COLUMN)


def evaluate(task_type: str, y_true, y_pred):
    if task_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
        }
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False))
    }


def _export_csv(df: pd.DataFrame, base_path: Path) -> Path:
    """
    Save a DataFrame as CSV (forced).
    Returns the final path.
    """
    p = base_path.with_suffix(".csv")
    df.to_csv(p, index=False)
    return p


def _get_feature_names(preproc) -> List[str]:
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        return []


def main():
    for d in (PLAN_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    data_path = Path(DATASET_PATHS[DATASET_NAME])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    # Read with NA handling so imputers treat '?' as missing
    df = pd.read_csv(data_path, na_values=NA_VALUES)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column {TARGET_COLUMN!r} not found in dataset.")

    task = load_task_definition(df)
    task_type = task["task_type"]
    metric = task.get("metric")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # holdout for final report (AutoML search uses CV inside)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if task_type == "classification" else None
    )

    best_pipeline, best_cv_score, best_config = automl_preprocessing_search(
        X_train, y_train, task_type=task_type, metric=metric, cv=5,  
    )

    # Fit the final pipeline on train and evaluate
    best_pipeline.fit(X_train, y_train)
    preds = best_pipeline.predict(X_test)
    test_metrics = evaluate(task_type, y_test, preds)

    # ================= Export transformed features (CSV forced) =================
    preproc = best_pipeline.named_steps["preprocessing"]

    # Transform splits and full dataset
    X_train_proc = preproc.transform(X_train)
    X_test_proc = preproc.transform(X_test)
    X_all_proc = preproc.transform(X)

    # Feature names if available
    feature_names = _get_feature_names(preproc)
    if not feature_names:
        feature_names = [f"f_{i}" for i in range(X_train_proc.shape[1])]

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    X_all_df = pd.DataFrame(X_all_proc, columns=feature_names)

    # Add target back for convenience
    train_out = X_train_df.copy()
    train_out[TARGET_COLUMN] = y_train.reset_index(drop=True)

    test_out = X_test_df.copy()
    test_out[TARGET_COLUMN] = y_test.reset_index(drop=True)

    full_out = X_all_df.copy()
    full_out[TARGET_COLUMN] = y.reset_index(drop=True)

    # Save as CSV
    train_path = _export_csv(train_out, OUTPUT_DIR / f"{DATASET_NAME}_train_preprocessed")
    test_path  = _export_csv(test_out,  OUTPUT_DIR / f"{DATASET_NAME}_test_preprocessed")
    full_path  = _export_csv(full_out,  OUTPUT_DIR / f"{DATASET_NAME}_full_preprocessed")

    print(f"\nSaved preprocessed TRAIN CSV to: {train_path}")
    print(f"Saved preprocessed TEST  CSV to: {test_path}")
    print(f"Saved preprocessed FULL  CSV to: {full_path}")
    # ===========================================================================

    # Save report + pipeline
    report = {
        "dataset": DATASET_NAME,
        "target": TARGET_COLUMN,
        "task_type": task_type,
        "metric_used_in_search": best_config["scoring"],
        "best_cv_score": float(best_cv_score),
        "best_config": best_config,
        "holdout_test_metrics": test_metrics,
        "exports": {
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            "full_csv": str(full_path)
        }
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    joblib.dump(best_pipeline, BEST_PIPELINE_PATH)  # <-- correct name

    print("\n=== FINAL REPORT ===\n")
    print(json.dumps(report, indent=2))
    print(f"\nSaved pipeline to: {BEST_PIPELINE_PATH.resolve()}")


if __name__ == "__main__":
    main()
