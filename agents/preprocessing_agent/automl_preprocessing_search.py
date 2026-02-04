"""
Improved AutoML preprocessing + model search
- Dense OHE for tree models (version-safe)
- Optional: save best preprocessing config to Output/
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, List
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def _make_ohe() -> OneHotEncoder:
    """Return an OneHotEncoder that outputs dense arrays across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # sklearn <= 1.1


def _pick_model(task_type: str) -> List[Tuple[str, object]]:
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")

    if task_type == "classification":
        return [
            ("logreg", LogisticRegression(max_iter=2000)),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    return [
        ("ridge", Ridge(alpha=1.0)),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ]


def _default_scoring(task_type: str, metric: Optional[str]) -> str:
    if metric:
        return metric
    return "accuracy" if task_type == "classification" else "r2"


def _build_preprocessor(
    X: pd.DataFrame,
    num_imputer: str,
    cat_imputer: str,
    scaler
) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy=num_imputer))]
    if len(num_cols) > 0 and scaler is not None:
        num_steps.append(("scaler", scaler))

    num_pipeline = Pipeline(num_steps)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_imputer)),
        ("encoder", _make_ohe()),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    if not transformers:
        raise ValueError("X has no numeric or categorical columns to preprocess.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def automl_preprocessing_search(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    metric: Optional[str] = None,
    cv: int = 5,
    output_dir: Optional[Path] = None,   # <- NEW: where to save the chosen config (e.g., Path('Output'))
) -> Tuple[Pipeline, float, Dict[str, str]]:
    scoring = _default_scoring(task_type, metric)

    # Preprocessing search space
    search_space = [
        # (num_imputer, cat_imputer, scaler)
        ("mean", "most_frequent", StandardScaler()),
        ("median", "most_frequent", StandardScaler()),
        ("mean", "most_frequent", MinMaxScaler()),
        ("median", "most_frequent", MinMaxScaler()),
        ("median", "most_frequent", None),  # trees often don't need scaling
    ]

    best: Dict[str, object] = {"score": -np.inf, "pipeline": None, "config": None}

    for num_imp, cat_imp, scaler in search_space:
        preprocessor = _build_preprocessor(X, num_imp, cat_imp, scaler)

        for model_name, model in _pick_model(task_type):
            pipe = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model),
            ])

            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            mean_score = float(np.mean(scores))

            if mean_score > best["score"]:
                best["score"] = mean_score
                best["pipeline"] = pipe
                best["config"] = {
                    "num_imputer": num_imp,
                    "cat_imputer": cat_imp,
                    "scaler": scaler.__class__.__name__ if scaler is not None else "None",
                    "model": model_name,
                    "cv": cv,
                    "scoring": scoring,
                    "cv_mean_score": mean_score,
                }

    # Optionally save the chosen preprocessing/model config into output_dir
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = output_dir / "best_preprocessing.json"
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(best["config"], f, indent=2)  # type: ignore[arg-type]
        except Exception as e:
            # Non-fatal: just print and continue
            print(f"Warning: failed to write best_preprocessing.json: {e}")

    return best["pipeline"], float(best["score"]), best["config"]  # type: ignore[return-value]
