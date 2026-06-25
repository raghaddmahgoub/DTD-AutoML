"""
Tool: simple sklearn training with default hyperparameters (no Optuna).

Uses preprocessed train/test splits from PreprocessingAgent (no in-tool preprocessing).
"""
from langchain_core.tools import tool

from tools.training.training_engines import train_simple_defaults
from tools.shared import ensure_state, parse_tool_input
from tools.shared.training_common import (
    APPROACH_SIMPLE,
    align_features_for_model,
    apply_test_metrics,
    complete_training,
    load_training_context,
    require_approved_plan,
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

    models = ctx["plan"].get("selected_models") or ["RandomForest", "GradientBoosting"]
    if cfg.get("models"):
        models = cfg["models"] if isinstance(cfg["models"], list) else [
            m.strip() for m in str(cfg["models"]).split(",") if m.strip()
        ]

    model, metrics = train_simple_defaults(
        ctx["X_train"], ctx["y_train"], ctx["problem_type"], models
    )
    X_test = align_features_for_model(model, ctx["X_test"])
    preds = model.predict(X_test)
    metrics = apply_test_metrics(metrics, ctx["y_test"], preds, ctx["problem_type"])

    return complete_training(
        pipeline_state,
        model=model,
        metrics=metrics,
        training_method="Simple+Defaults",
        n_rows=ctx["n_rows"],
        subfolder="training_simple",
    )
