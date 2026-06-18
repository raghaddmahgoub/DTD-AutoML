"""Decide training strategy via LangGraph plan subgraph + user approval."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from tools.plan_graph import build_plan_graph
from tools.pipeline_state import ensure_state, merge_state, parse_tool_input
from tools.training_common import (
    LARGE_DATA_ROW_THRESHOLD,
    load_planning_dataframe,
    load_preprocessed_splits,
    pipeline_to_graph_state,
    require_preprocessed_splits,
    resolve_problem_type,
)


def _is_yes(text: str) -> bool:
    return text.strip().lower() in {"y", "yes", "true", "1"}


APPROACH_LABELS = {
    "1": ("simple", "train_simple", "Simple (default hyperparameters)"),
    "2": ("simple_optuna", "train_simple_optuna", "Simple + Optuna HPO"),
    "3": ("autogluon", "train_autogluon", "AutoGluon"),
}


def _resolve_approach(
    use_automl: bool, user_choice: str = "", llm_approach: str = ""
) -> tuple[str, str, str]:
    choice = user_choice.strip().lower()
    if choice in APPROACH_LABELS:
        return APPROACH_LABELS[choice]
    if choice in ("simple", "simple_optuna", "autogluon"):
        return {
            "simple": APPROACH_LABELS["1"],
            "simple_optuna": APPROACH_LABELS["2"],
            "autogluon": APPROACH_LABELS["3"],
        }[choice]
    llm = llm_approach.strip().lower()
    if llm:
        if "autogluon" in llm:
            return APPROACH_LABELS["3"]
        if "optuna" in llm or "simple+optuna" in llm.replace(" ", ""):
            return APPROACH_LABELS["2"]
        if "dask" in llm:
            return APPROACH_LABELS["2"]
        if "simple" in llm and "optuna" not in llm:
            return APPROACH_LABELS["1"]
        if "optuna" in llm or "simple+optuna" in llm.replace(" ", ""):
            return APPROACH_LABELS["2"]
    if use_automl:
        return APPROACH_LABELS["3"]
    return APPROACH_LABELS["1"]


def _apply_user_overrides(plan: dict[str, Any], prefs: dict[str, Any]) -> dict[str, Any]:
    out = dict(plan)
    preferred = prefs.get("preferred_models") or []
    time_pref = (prefs.get("time_preference") or "").lower()
    hw = (prefs.get("hw_complexity") or "").lower()

    if preferred:
        if out.get("approach") == "autogluon":
            cfg = dict(out.get("automl_config") or {})
            cfg["models_to_prioritize"] = preferred
            out["automl_config"] = cfg
        else:
            out["selected_models"] = preferred[:3]

    cfg = dict(out.get("automl_config") or {})
    if time_pref == "fast":
        cfg.update({"preset_mode": "medium_quality", "time_limit_seconds": 90})
    elif time_pref == "balanced":
        cfg.update({"preset_mode": "good_quality", "time_limit_seconds": 180})
    elif time_pref == "best":
        cfg.update({"preset_mode": "best_quality", "time_limit_seconds": 600})
    if hw == "low":
        cfg["preset_mode"] = "optimize_for_deployment"
    if cfg:
        out["automl_config"] = cfg
    return out


@tool
def plan_training(task, tool_input, prompt, data_path, llm, state=None):
    """
    LangGraph planning tool: identify_target → model_selection.
    Does NOT train. Requires user approval before train_* tools run.
    """
    print("=========================================================================")
    print(f"[TOOL] Plan training (LangGraph): {task}")

    pipeline_state = ensure_state(state, data_path, prompt)
    cfg = parse_tool_input(tool_input)

    if prep_err := require_preprocessed_splits(pipeline_state):
        return {"status": "error", "error": prep_err}, merge_state(
            pipeline_state, {"status": "error", "step": "plan_failed"}
        )

    prefs = dict(pipeline_state.get("user_preferences") or {})
    ask_user = bool(cfg.get("ask_before_training", prefs.get("ask_before_training", True)))

    load_planning_dataframe(pipeline_state)  # validate preprocessed splits
    target_column = cfg.get("target_column") or pipeline_state.get("target_column")
    if not target_column:
        return {"status": "error", "error": "target_column missing after preprocessing."}, merge_state(
            pipeline_state, {"status": "error", "step": "plan_failed"}
        )
    problem_type = cfg.get("problem_type") or resolve_problem_type(pipeline_state)
    user_note = prefs.get("user_training_prompt", "")
    preferred_models = list(prefs.get("preferred_models") or [])
    time_preference = prefs.get("time_preference", "")
    hw_complexity = prefs.get("hw_complexity", "")
    training_approach = cfg.get("training_approach", prefs.get("training_approach", ""))

    if ask_user:
        t_in = input(f"Target column [{target_column}]: ").strip()
        if t_in:
            target_column = t_in
        pt_in = input(
            f"Problem type [auto/classification/regression] (current={problem_type or 'auto'}): "
        ).strip().lower()
        if pt_in in ("classification", "regression"):
            problem_type = pt_in
        models_in = input("Only these models? comma-separated or empty: ").strip()
        if models_in:
            preferred_models = [m.strip() for m in models_in.split(",") if m.strip()]
        time_in = input("Time preference [fast/balanced/best] or empty: ").strip().lower()
        if time_in:
            time_preference = time_in
        hw_in = input("HW complexity [low/medium/high] or empty: ").strip().lower()
        if hw_in:
            hw_complexity = hw_in
        note_in = input("Training suggestion (optional): ").strip()
        if note_in:
            user_note = note_in
        print("Training approach:\n  1=Simple  2=Simple+Optuna  3=AutoGluon")
        approach_in = input("Choose [1/2/3] or empty for LLM default: ").strip()
        if approach_in:
            training_approach = approach_in

    prefs.update(
        {
            "preferred_models": preferred_models,
            "time_preference": time_preference,
            "hw_complexity": hw_complexity,
            "user_training_prompt": user_note,
            "training_approach": training_approach,
        }
    )

    pipeline_state = merge_state(
        pipeline_state,
        {
            "target_column": target_column,
            "problem_type": problem_type,
            "user_preferences": prefs,
        },
    )
    if isinstance(cfg.get("report"), dict):
        pipeline_state = merge_state(pipeline_state, {"report": cfg["report"]})

    graph_state = pipeline_to_graph_state(pipeline_state)
    graph_state["target_column"] = target_column
    graph_state["problem_type"] = problem_type
    graph_state["automl_directives"]["user"] = {
        "task_prompt": str(prompt or pipeline_state.get("prompt", ""))[:500],
        "training_note": user_note[:300],
        "time_preference": time_preference,
        "hw_complexity": hw_complexity,
        "preferred_models": preferred_models,
    }

    plan_graph = build_plan_graph(llm)
    graph_state = plan_graph.invoke(graph_state)

    if graph_state.get("error"):
        return {"status": "error", "error": graph_state["error"]}, merge_state(
            pipeline_state, {"status": "error", "step": "plan_failed"}
        )

    use_automl = bool(graph_state.get("use_automl"))
    llm_approach = graph_state.get("llm_approach", "")

    approach, train_tool, training_method = _resolve_approach(
        use_automl,
        training_approach or prefs.get("training_approach", ""),
        llm_approach,
    )
    _, _, _, _, n_rows = load_preprocessed_splits(pipeline_state)
    use_dask = n_rows > LARGE_DATA_ROW_THRESHOLD
    dask_note = (
        f"Large dataset ({n_rows:,} rows): note — training uses preprocessed splits from PreprocessingAgent."
        if use_dask
        else f"Dataset size {n_rows:,} rows — preprocessed train/test splits."
    )

    plan = {
        "approach": approach,
        "training_method": training_method,
        "selected_models": graph_state.get("selected_models") or [],
        "automl_config": graph_state.get("automl_config") or {},
        "optuna_config": graph_state.get("optuna_config") or {},
        "reasoning": (graph_state.get("model_selection_reasoning") or "")[:1200],
        "approved": False,
        "train_tool": train_tool,
        "use_dask_training": use_dask,
        "n_rows": n_rows,
        "dask_note": dask_note,
    }
    plan = _apply_user_overrides(plan, prefs)

    if approach in ("simple", "simple_optuna") and not plan.get("selected_models"):
        defaults = {
            "classification": ["RandomForest", "GradientBoosting", "LogisticRegression"],
            "regression": ["RandomForest", "GradientBoosting", "LinearRegression"],
        }
        plan["selected_models"] = defaults.get(
            graph_state.get("problem_type") or "classification",
            ["RandomForest", "GradientBoosting"],
        )[:3]
    preview = {
        **plan,
        "llm_approach": graph_state.get("llm_approach", ""),
        "target_column": graph_state.get("target_column"),
        "problem_type": graph_state.get("problem_type"),
        "time_preference": time_preference or "none",
        "hw_complexity": hw_complexity or "none",
        "user_hint": user_note[:300] if user_note else str(prompt or "")[:300],
    }
    print("[TRAINING PLAN]")
    print(json.dumps(preview, indent=2))
    print("\n" + "=" * 70)
    print("TRAINING PLAN SUMMARY")
    print(f"  Target:   {preview.get('target_column')} ({preview.get('problem_type')})")
    print(f"  Method:   {preview.get('training_method')}")
    print(f"  Models:   {preview.get('selected_models') or preview.get('automl_config', {}).get('models', [])}")
    print(f"  Tool:     {preview.get('train_tool')}")
    print(f"  Dataset:  {preview.get('n_rows'):,} rows — {preview.get('dask_note')}")
    print("=" * 70)

    approved = bool(cfg.get("auto_approve_plan", False))
    if ask_user:
        approved = _is_yes(
            input("\nContinue with this LLM-suggested training plan? [y/N]: ").strip()
        )
    plan["approved"] = approved
    preview["approved"] = approved

    pipeline_state = merge_state(
        pipeline_state,
        {
            "target_column": graph_state.get("target_column"),
            "problem_type": graph_state.get("problem_type"),
            "training_plan": plan,
            "step": "plan_ready" if approved else "plan_rejected",
            "status": "planned" if approved else "cancelled",
        },
    )
    return {
        "status": "planned" if approved else "cancelled",
        "message": "Plan approved. Call train_tool next." if approved else "Plan rejected.",
        "plan_preview": preview,
        "train_tool": plan["train_tool"],
    }, pipeline_state
