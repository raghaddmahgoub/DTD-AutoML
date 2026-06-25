"""
Tool: simple sklearn training with Optuna hyperparameter tuning.

Uses preprocessed train/test splits from PreprocessingAgent (no in-tool preprocessing).
"""
from langchain_core.tools import tool

from tools.training.training_engines import train_simple_optuna as run_simple_optuna
from tools.shared import ensure_state, parse_tool_input
from tools.shared.training_common import (
    APPROACH_SIMPLE_OPTUNA,
    align_features_for_model,
    apply_test_metrics,
    complete_training,
    load_training_context,
    require_approved_plan,
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

    models = ctx["plan"].get("selected_models") or ["RandomForest", "GradientBoosting"]
    optuna_config = ctx["plan"].get("optuna_config") or {}
    if cfg.get("optuna_trials"):
        optuna_config = {**optuna_config, "n_trials": int(cfg["optuna_trials"])}

    model, metrics = run_simple_optuna(
        ctx["X_train"],
        ctx["y_train"],
        ctx["problem_type"],
        models,
        optuna_config=optuna_config,
    )
    X_test = align_features_for_model(model, ctx["X_test"])
    preds = model.predict(X_test)
    metrics = apply_test_metrics(metrics, ctx["y_test"], preds, ctx["problem_type"])

    return complete_training(
        pipeline_state,
        model=model,
        metrics=metrics,
        training_method="Simple+Optuna",
        n_rows=ctx["n_rows"],
        subfolder="training_simple_optuna",
    )
