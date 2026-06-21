"""Decide training strategy via LangGraph plan subgraph."""
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


APPROACH_LABELS = {
    "1": ("simple", "train_simple", "Simple (default hyperparameters)"),
    "2": ("simple_optuna", "train_simple_optuna", "Simple + Optuna HPO"),
    "3": ("autogluon", "train_autogluon", "AutoGluon"),
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


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


def _build_plan_from_graph(
    *,
    pipeline_state: dict[str, Any],
    graph_state: dict[str, Any],
    prefs: dict[str, Any],
    prompt: str,
    training_approach: str,
    time_preference: str,
    hw_complexity: str,
    controller_task: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "approved": True,
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
        "user_hint": str(prompt or "")[:300],
        "controller_task": controller_task[:200] if controller_task else "",
    }
    return plan, preview


@tool
def plan_training(task, tool_input, prompt, data_path, llm, state=None):
    """
    LangGraph planning tool: identify_target → model_selection.
    Does NOT train. Plan is auto-approved for the train_* tools.
    """
    print("=========================================================================")
    print(f"[TOOL] Plan training (LangGraph): {task}")

    pipeline_state = ensure_state(state, data_path, prompt)
    cfg = parse_tool_input(tool_input)

    controller_task = str(
        cfg.get("controller_task")
        or pipeline_state.get("controller_task")
        or task
        or ""
    ).strip()
    if controller_task:
        pipeline_state = merge_state(pipeline_state, {"controller_task": controller_task})

    if prep_err := require_preprocessed_splits(pipeline_state):
        return {"status": "error", "error": prep_err}, merge_state(
            pipeline_state, {"status": "error", "step": "plan_failed"}
        )

    prefs = dict(pipeline_state.get("user_preferences") or {})

    target_column = cfg.get("target_column") or pipeline_state.get("target_column")
    if not target_column:
        return {"status": "error", "error": "target_column missing after preprocessing."}, merge_state(
            pipeline_state, {"status": "error", "step": "plan_failed"}
        )
    problem_type = cfg.get("problem_type") or resolve_problem_type(pipeline_state)
    preferred_models = _as_list(
        cfg.get("preferred_models")
        or cfg.get("models")
        or prefs.get("preferred_models")
    )
    time_preference = cfg.get("time_preference", prefs.get("time_preference", ""))
    hw_complexity = cfg.get("hw_complexity", prefs.get("hw_complexity", ""))
    training_approach = cfg.get("training_approach", prefs.get("training_approach", ""))

    prefs.update(
        {
            "preferred_models": preferred_models,
            "time_preference": time_preference,
            "hw_complexity": hw_complexity,
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
    load_planning_dataframe(pipeline_state)  # validate preprocessed splits
    if isinstance(cfg.get("report"), dict):
        pipeline_state = merge_state(pipeline_state, {"report": cfg["report"]})

    graph_state = pipeline_to_graph_state(pipeline_state)
    graph_state["target_column"] = target_column
    graph_state["problem_type"] = problem_type
    graph_state["automl_directives"]["user"] = {
        "controller_task": controller_task[:1200],
        "task_prompt": str(prompt or pipeline_state.get("prompt", ""))[:1200],
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

    plan, preview = _build_plan_from_graph(
        pipeline_state=pipeline_state,
        graph_state=graph_state,
        prefs=prefs,
        prompt=prompt,
        training_approach=training_approach,
        time_preference=time_preference,
        hw_complexity=hw_complexity,
        controller_task=controller_task,
    )

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

    plan["approved"] = True
    preview["approved"] = True

    pipeline_state = merge_state(
        pipeline_state,
        {
            "target_column": graph_state.get("target_column"),
            "problem_type": graph_state.get("problem_type"),
            "training_plan": plan,
            "user_preferences": prefs,
            "step": "plan_ready",
            "status": "planned",
        },
    )
    return {
        "status": "planned",
        "message": "Plan ready. Call train_tool next.",
        "plan_preview": preview,
        "train_tool": plan.get("train_tool"),
    }, pipeline_state
