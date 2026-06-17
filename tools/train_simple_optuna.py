"""
Tool: simple sklearn training with Optuna hyperparameter tuning.

Uses standalone tools/nodes/training_engines.py (NOT AutoMLAgent).
"""
from langchain_core.tools import tool

from tools.nodes.training_engines import train_simple_optuna as run_simple_optuna
from tools.pipeline_state import ensure_state, parse_tool_input
from tools.training_common import (
    APPROACH_SIMPLE_OPTUNA,
    apply_test_metrics,
    complete_training,
    load_training_context,
    require_approved_plan,
    run_dask_xgb_training,
    train_test_split_xy,
)


@tool
def train_simple_optuna(task, tool_input, prompt, data_path, llm, state=None):
    """Train with sklearn + Optuna. Requires approved plan approach=simple_optuna."""
    print("=========================================================================")
    print(f"[TOOL] Train simple + Optuna: {task}")

    pipeline_state = ensure_state(state, data_path, prompt)
    cfg = parse_tool_input(tool_input)

    if err := require_approved_plan(pipeline_state, APPROACH_SIMPLE_OPTUNA):
        return {"status": "error", "error": err}, pipeline_state

    ctx, ctx_err = load_training_context(pipeline_state)
    if ctx_err:
        return {"status": "error", "error": ctx_err}, pipeline_state

    if ctx["use_dask"]:
        print("[train_simple_optuna] Large dataset → Dask-XGBoost")
        return run_dask_xgb_training(pipeline_state, ctx)

    models = ctx["plan"].get("selected_models") or ["RandomForest", "GradientBoosting"]
    optuna_config = ctx["plan"].get("optuna_config") or {}
    if cfg.get("optuna_trials"):
        optuna_config = {**optuna_config, "n_trials": int(cfg["optuna_trials"])}

    X_train, X_test, y_train, y_test = train_test_split_xy(ctx["X"], ctx["y"])
    model, metrics = run_simple_optuna(
        X_train, y_train, ctx["problem_type"], models, optuna_config=optuna_config
    )
    preds = model.predict(X_test)
    metrics = apply_test_metrics(metrics, y_test, preds, ctx["problem_type"])

    return complete_training(
        pipeline_state,
        model=model,
        metrics=metrics,
        training_method="Simple+Optuna",
        used_dask=False,
        n_rows=ctx["n_rows"],
        subfolder="training_simple_optuna",
    )
