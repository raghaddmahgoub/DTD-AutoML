"""
Tool: simple sklearn training with default hyperparameters (no Optuna).

Uses standalone tools/nodes/training_engines.py (NOT AutoMLAgent).
"""
from langchain_core.tools import tool

from tools.nodes.training_engines import train_simple_defaults
from tools.pipeline_state import ensure_state, parse_tool_input
from tools.training_common import (
    APPROACH_SIMPLE,
    apply_test_metrics,
    complete_training,
    load_training_context,
    require_approved_plan,
    run_dask_xgb_training,
    train_test_split_xy,
)


@tool
def train_simple(task, tool_input, prompt, data_path, llm, state=None):
    """Train with default sklearn hyperparameters. Requires approved plan approach=simple."""
    print("=========================================================================")
    print(f"[TOOL] Train simple (defaults): {task}")

    pipeline_state = ensure_state(state, data_path, prompt)
    cfg = parse_tool_input(tool_input)

    if err := require_approved_plan(pipeline_state, APPROACH_SIMPLE):
        return {"status": "error", "error": err}, pipeline_state

    ctx, ctx_err = load_training_context(pipeline_state)
    if ctx_err:
        return {"status": "error", "error": ctx_err}, pipeline_state

    if ctx["use_dask"]:
        print("[train_simple] Large dataset → Dask-XGBoost")
        return run_dask_xgb_training(pipeline_state, ctx)

    models = ctx["plan"].get("selected_models") or ["RandomForest", "GradientBoosting"]
    if cfg.get("models"):
        models = cfg["models"] if isinstance(cfg["models"], list) else [
            m.strip() for m in str(cfg["models"]).split(",") if m.strip()
        ]

    X_train, X_test, y_train, y_test = train_test_split_xy(ctx["X"], ctx["y"])
    model, metrics = train_simple_defaults(X_train, y_train, ctx["problem_type"], models)
    preds = model.predict(X_test)
    metrics = apply_test_metrics(metrics, y_test, preds, ctx["problem_type"])

    return complete_training(
        pipeline_state,
        model=model,
        metrics=metrics,
        training_method="Simple+Defaults",
        used_dask=False,
        n_rows=ctx["n_rows"],
        subfolder="training_simple",
    )
