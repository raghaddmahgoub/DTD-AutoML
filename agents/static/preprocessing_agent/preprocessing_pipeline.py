"""
Unified Preprocessing Pipeline Agent
-------------------------------------

This file merges:
- Task inference
- Automated preprocessing search
- Final pipeline fitting
- CSV export
- Config persistence

Compatible with orchestrator.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# =========================
# CONFIG
# =========================

DATA_PATH = "assets/data/Classification Datasets/breast_cancer.csv"
OUTPUT_DIR = Path("output/preprocessing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLAN_DIR = Path("Plan")
PLAN_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# TASK INFERENCE (Simplified)
# =========================


def infer_task(df: pd.DataFrame, target_column: str) -> Dict[str, str]:
    s = df[target_column]

    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 30:
        return {"task_type": "regression", "metric": "r2"}

    return {"task_type": "classification", "metric": "accuracy"}


# =========================
# PREPROCESSING AGENT
# =========================

class PreprocessingPipelineAgent:
    """
    Agent-style preprocessing runner.
    Designed to plug directly into orchestrator.
    """

    def __init__(self):
        self.best_pipeline = None
        self.best_config = None
        self.best_score = None

    # ---------------------------------------
    # MAIN RUN
    # ---------------------------------------

    def run(self,
            data_path: str = DATA_PATH,
            target_column: Optional[str] = None
            ) -> Dict[str, Any]:

        print("🔄 Loading dataset...")
        df = pd.read_csv(data_path)

        if target_column is None:
            target_column = df.columns[-1]

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # ---------------------------
        # TASK INFERENCE
        # ---------------------------
        task_info = infer_task(df, target_column)
        task_type = task_info["task_type"]
        metric = task_info["metric"]

        print(f"📌 Task detected: {task_type}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ---------------------------
        # TRAIN TEST SPLIT
        # ---------------------------
        use_stratify = False

        if task_type == "classification":
            class_counts = y.value_counts()
            
            if class_counts.min() < 2:
                print("⚠️ Dropping rare classes with <2 samples...")
                valid_classes = class_counts[class_counts >= 2].index
                df = df[df[target_column].isin(valid_classes)]
                X = df.drop(columns=[target_column])
                y = df[target_column]
                use_stratify = True
            else:
                use_stratify = True

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if use_stratify else None
        )

        # ---------------------------
        # IMPORT YOUR SEARCH FUNCTION
        # ---------------------------
        from agents.static.preprocessing_agent.automl_preprocessing_search import automl_preprocessing_search

        print("🔍 Running preprocessing + model search...")

        best_pipeline, best_cv_score, best_config = automl_preprocessing_search(
            X_train,
            y_train,
            task_type=task_type,
            metric=metric,
            cv=5,
            output_dir=OUTPUT_DIR
        )

        # Save internal state
        self.best_pipeline = best_pipeline
        self.best_config = best_config
        self.best_score = best_cv_score

        # ---------------------------
        # FINAL FIT
        # ---------------------------
        best_pipeline.fit(X_train, y_train)

        # ---------------------------
        # EXPORT PREPROCESSED CSV
        # ---------------------------
        print("💾 Exporting preprocessed datasets...")

        preproc = best_pipeline.named_steps["preprocessing"]

        X_train_proc = preproc.transform(X_train)
        X_test_proc = preproc.transform(X_test)
        X_all_proc = preproc.transform(X)

        feature_names = self._get_feature_names(preproc)

        X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
        X_all_df = pd.DataFrame(X_all_proc, columns=feature_names)

        X_train_df[target_column] = y_train.reset_index(drop=True)
        X_test_df[target_column] = y_test.reset_index(drop=True)
        X_all_df[target_column] = y.reset_index(drop=True)

        train_path = OUTPUT_DIR / "train_preprocessed.csv"
        test_path = OUTPUT_DIR / "test_preprocessed.csv"
        full_path = OUTPUT_DIR / "full_preprocessed.csv"

        X_train_df.to_csv(train_path, index=False)
        X_test_df.to_csv(test_path, index=False)
        X_all_df.to_csv(full_path, index=False)

        # ---------------------------
        # SAVE PIPELINE
        # ---------------------------
        joblib.dump(best_pipeline, OUTPUT_DIR / "best_pipeline.pkl")

        # ---------------------------
        # SAVE CONFIG
        # ---------------------------
        with open(OUTPUT_DIR / "best_preprocessing.json", "w") as f:
            json.dump(best_config, f, indent=2)

        print("✅ Preprocessing complete.")

        return {
            "task_type": task_type,
            "metric": metric,
            "best_cv_score": float(best_cv_score),
            "best_config": best_config,
            "exports": {
                "train": str(train_path),
                "test": str(test_path),
                "full": str(full_path)
            }
        }

    # ---------------------------------------
    # UTIL
    # ---------------------------------------

    def _get_feature_names(self, preproc):
        try:
            return list(preproc.get_feature_names_out())
        except Exception:
            return [f"f_{i}" for i in range(preproc.transform(pd.DataFrame([{}])).shape[1])]


# =========================
# DIRECT EXECUTION
# =========================

if __name__ == "__main__":
    agent = PreprocessingPipelineAgent()
    result = agent.run()
    print("\n🏁 DONE")
    print(json.dumps(result, indent=2))
