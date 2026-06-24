"""Standalone training engines (no AutoMLAgent import)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from src.utils.logger import Logger

logger = Logger()

# Legacy preset names the LLM may suggest but AutoGluon 1.5 no longer registers.
_LEGACY_PRESET_ALIASES = {
    "good_quality_faster_inference": "good_quality",
    "high_quality_fast_inference": "high_quality",
}


def normalize_autogluon_preset(preset: str) -> str:
    """Map LLM / legacy preset strings to a name AutoGluon 1.5 accepts."""
    token = str(preset).strip()
    if not token:
        return "good_quality"
    try:
        from autogluon.tabular.configs.presets_configs import (
            tabular_presets_alias,
            tabular_presets_dict,
        )

        if token in tabular_presets_dict or token in tabular_presets_alias:
            return token
    except ImportError:
        pass
    return _LEGACY_PRESET_ALIASES.get(token, token)


try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]


def _build_sklearn_model(model_name: str, problem_type: str, params: dict | None = None):
    params = params or {}
    name = model_name.lower()
    if "randomforest" in name:
        return (
            RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            if problem_type == "classification"
            else RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        )
    if ("xgboost" in name or "xgb" in name) and xgb is not None:
        return (
            xgb.XGBClassifier(**params, random_state=42, eval_metric="logloss")
            if problem_type == "classification"
            else xgb.XGBRegressor(**params, random_state=42)
        )
    if "gradient" in name:
        return (
            GradientBoostingClassifier(**params, random_state=42)
            if problem_type == "classification"
            else GradientBoostingRegressor(**params, random_state=42)
        )
    if "logistic" in name:
        return LogisticRegression(**params, max_iter=1000, random_state=42)
    if "linear" in name:
        return LinearRegression(**params)
    return (
        RandomForestClassifier(random_state=42, n_jobs=-1)
        if problem_type == "classification"
        else RandomForestRegressor(random_state=42, n_jobs=-1)
    )


def train_simple_defaults(
    X: pd.DataFrame, y: pd.Series, problem_type: str, model_names: list[str]
) -> tuple[object, dict]:
    if not model_names:
        model_names = ["RandomForest", "GradientBoosting"]

    Xp = pd.get_dummies(X, drop_first=True)
    X_tune, X_val, y_tune, y_val = train_test_split(Xp, y, test_size=0.2, random_state=42)
    metric_name = "accuracy" if problem_type == "classification" else "r2_score"

    best_name, best_score = None, -float("inf")
    all_results = []
    for name in model_names:
        try:
            model = _build_sklearn_model(name, problem_type)
            model.fit(X_tune, y_tune)
            preds = model.predict(X_val)
            score = (
                accuracy_score(y_val, preds)
                if problem_type == "classification"
                else r2_score(y_val, preds)
            )
            all_results.append({"model_name": name, "score": float(score), "best_params": {}})
            if score > best_score:
                best_score, best_name = score, name
        except Exception as exc:
            logger.warn(f"train_simple_defaults failed for {name}: {exc}")

    if best_name is None:
        best_name = "RandomForest"
        best_score = 0.0

    final_model = _build_sklearn_model(best_name, problem_type)
    final_model.fit(Xp, y)
    return final_model, {
        "best_model": best_name,
        "best_score": float(best_score),
        "metric_name": metric_name,
        "models_trained": len(all_results),
        "all_models": [r["model_name"] for r in all_results],
        "all_scores": [r["score"] for r in all_results],
        "training_method": "Simple+Defaults",
    }


def _default_optuna_space(model_name: str) -> dict[str, dict]:
    low = model_name.lower()
    if "randomforest" in low:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 3, "high": 20},
        }
    if "gradient" in low:
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        }
    if "logistic" in low:
        return {"C": {"type": "float", "low": 1e-3, "high": 10.0, "log": True}}
    if "xgboost" in low or low == "xgb":
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 300},
            "max_depth": {"type": "int", "low": 3, "high": 12},
            "learning_rate": {"type": "float", "low": 1e-3, "high": 0.3, "log": True},
        }
    return {}


def _resolve_model_space(model_name: str, search_space: dict) -> dict[str, dict]:
    if model_name in search_space:
        return search_space[model_name]
    low = model_name.lower()
    for key, specs in search_space.items():
        if key.lower() in low or low in key.lower():
            return specs
    return _default_optuna_space(model_name)


def _suggest_params(trial, specs: dict[str, dict]) -> dict:
    params: dict = {}
    for name, bounds in specs.items():
        kind = bounds.get("type")
        low, high = bounds.get("low"), bounds.get("high")
        if kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, float(low), float(high), log=bool(bounds.get("log"))
            )
    return params


def train_simple_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    model_names: list[str],
    optuna_config: dict | None = None,
) -> tuple[object, dict]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if not model_names:
        model_names = ["RandomForest", "GradientBoosting"]

    cfg = dict(optuna_config or {})
    search_space = cfg.get("search_space") or {}
    n_trials = int(cfg.get("n_trials", 30))
    n_trials = max(5, min(n_trials, 100))

    Xp = pd.get_dummies(X, drop_first=True)
    X_tune, X_val, y_tune, y_val = train_test_split(Xp, y, test_size=0.2, random_state=42)
    metric_name = "accuracy" if problem_type == "classification" else "r2_score"

    def objective(trial, model_name: str) -> float:
        specs = _resolve_model_space(model_name, search_space)
        params = _suggest_params(trial, specs) if specs else {}
        model = _build_sklearn_model(model_name, problem_type, params)
        model.fit(X_tune, y_tune)
        preds = model.predict(X_val)
        return (
            accuracy_score(y_val, preds)
            if problem_type == "classification"
            else r2_score(y_val, preds)
        )

    best_model, best_name, best_score, all_results = None, None, -float("inf"), []
    for model_name in model_names:
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda t: objective(t, model_name), n_trials=n_trials, show_progress_bar=False
            )
            params = study.best_params
            model = _build_sklearn_model(model_name, problem_type, params)
            model.fit(Xp, y)
            all_results.append(
                {"model_name": model_name, "score": float(study.best_value), "best_params": params}
            )
            if study.best_value > best_score:
                best_score, best_name, best_model = study.best_value, model_name, model
        except Exception as exc:
            logger.warn(f"train_simple_optuna failed for {model_name}: {exc}")

    if best_model is None:
        best_model, metrics = train_simple_defaults(X, y, problem_type, model_names)
        metrics["training_method"] = "Simple+Optuna (fallback defaults)"
        return best_model, metrics

    return best_model, {
        "best_model": best_name,
        "best_score": float(best_score),
        "metric_name": metric_name,
        "models_trained": len(all_results),
        "all_models": [r["model_name"] for r in all_results],
        "all_scores": [r["score"] for r in all_results],
        "best_params_per_model": {r["model_name"]: r["best_params"] for r in all_results},
        "training_method": "Simple+Optuna",
        "optuna_trials": n_trials,
        "optuna_search_space": search_space,
    }


def train_autogluon(
    X: pd.DataFrame, y: pd.Series, problem_type: str, config: dict
) -> tuple[object, dict]:
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        reason = (
            "AutoGluon is not installed in this environment. "
            "Install with: pip install autogluon"
        )
        logger.warn(f"{reason} — falling back to sklearn defaults (RandomForest).")
        model, metrics = train_simple_defaults(X, y, problem_type, ["RandomForest"])
        metrics["autogluon_used"] = False
        metrics["fallback_reason"] = reason
        metrics["training_method"] = "Simple+Defaults (AutoGluon unavailable)"
        metrics["planned_method"] = "AutoGluon"
        return model, metrics

    train_data = X.copy()
    train_data["target"] = y
    ag_problem = problem_type
    if problem_type == "classification":
        ag_problem = "binary" if y.nunique() == 2 else "multiclass"

    predictor_path = Path("output/dynamic_pipeline/autogluon") / f"run_{int(time.time())}"
    predictor_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label="target",
        problem_type=ag_problem,
        path=str(predictor_path),
    )
    time_limit = int(config.get("time_limit", config.get("time_limit_seconds", 120)))
    preset = normalize_autogluon_preset(
        config.get("preset", config.get("preset_mode", "good_quality"))
    )
    models = config.get("models", config.get("models_to_prioritize", ["GBM", "XGB"]))
    hyperparameters = {m: {} for m in models}

    predictor.fit(
        train_data,
        time_limit=time_limit,
        presets=preset,
        hyperparameters=hyperparameters,
        # Prevent Ray/Dynamic Stacking
        dynamic_stacking=False,
        auto_stack=False,

        # Simpler and more stable on Windows
        num_bag_folds=0,
        num_stack_levels=0,
    )
    leaderboard = predictor.leaderboard(silent=True)
    best_model = leaderboard.iloc[0]["model"] if len(leaderboard) else "unknown"
    best_score = float(leaderboard.iloc[0]["score_val"]) if len(leaderboard) else 0.0
    return predictor, {
        "best_model": best_model,
        "best_score": best_score,
        "models_trained": len(leaderboard),
        "all_models": leaderboard["model"].tolist(),
        "all_scores": leaderboard["score_val"].tolist(),
        "training_method": "AutoGluon",
        "autogluon_used": True,
        "planned_method": "AutoGluon",
    }


def train_dask_xgb(X, y, problem_type: str) -> tuple[object, np.ndarray, np.ndarray, dict]:
    import dask.dataframe as dd
    from dask.distributed import Client, LocalCluster
    from dask_ml.model_selection import train_test_split as dask_split
    from xgboost import dask as dxgb

    cluster = LocalCluster(n_workers=2, threads_per_worker=2, memory_limit="5GB")
    client = Client(cluster)
    try:
        X = X.astype("float32")
        y = y.astype("float32")
        X_train, X_test, y_train, y_test = dask_split(X, y, test_size=0.2, random_state=42)
        dtrain = dxgb.DaskDMatrix(client, X_train, y_train)
        params = (
            {"objective": "binary:logistic", "eval_metric": "logloss"}
            if problem_type == "classification"
            else {"objective": "reg:squarederror", "eval_metric": "rmse"}
        )
        params["tree_method"] = "hist"
        output = dxgb.train(client, params, dtrain, num_boost_round=100)
        booster = output["booster"]
        y_pred = dxgb.predict(client, booster, X_test).compute()
        y_true = y_test.compute()
        metrics = {"best_model": "Dask-XGBoost", "training_method": "Dask-XGBoost"}
        return booster, y_true, y_pred, metrics
    finally:
        client.close()
        cluster.close()
