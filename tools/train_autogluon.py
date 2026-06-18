"""
Tool: AutoGluon AutoML training.

Uses preprocessed train/test splits from PreprocessingAgent (no in-tool preprocessing).
"""
from langchain_core.tools import tool

from tools.nodes.training_engines import train_autogluon as run_autogluon
from tools.pipeline_state import ensure_state, parse_tool_input
from tools.training_common import (
    APPROACH_AUTOGLUON,
    apply_test_metrics,
    complete_training,
    load_training_context,
    require_approved_plan,
)


@tool
def train_autogluon(task, tool_input, prompt, data_path, llm, state=None):
    """Train with AutoGluon. Requires approved plan approach=autogluon."""
    print("=========================================================================")
    print(f"[TOOL] Train AutoGluon: {task}")

    pipeline_state = ensure_state(state, data_path, prompt)
    cfg = parse_tool_input(tool_input)

    if err := require_approved_plan(pipeline_state, APPROACH_AUTOGLUON):
        return {"status": "error", "error": err}, pipeline_state

    ctx, ctx_err = load_training_context(pipeline_state)
    if ctx_err:
        return {"status": "error", "error": ctx_err}, pipeline_state

    automl_config = dict(ctx["plan"].get("automl_config") or {})
    if cfg.get("time_limit"):
        automl_config["time_limit"] = int(cfg["time_limit"])
    if cfg.get("preset"):
        automl_config["preset"] = cfg["preset"]

    model, metrics = run_autogluon(
        ctx["X_train"], ctx["y_train"], ctx["problem_type"], automl_config
    )

    if not metrics.get("autogluon_used", True):
        print("\n" + "!" * 70)
        print("[train_autogluon] AutoGluon was NOT used.")
        print(f"[train_autogluon] {metrics.get('fallback_reason')}")
        print(f"[train_autogluon] Actual trainer: {metrics.get('training_method')}")
        print("!" * 70 + "\n")

    preds = model.predict(ctx["X_test"])
    metrics = apply_test_metrics(metrics, ctx["y_test"], preds, ctx["problem_type"])

    return complete_training(
        pipeline_state,
        model=model,
        metrics=metrics,
        training_method=metrics.get("training_method", "AutoGluon"),
        n_rows=ctx["n_rows"],
        subfolder="training_autogluon",
    )
