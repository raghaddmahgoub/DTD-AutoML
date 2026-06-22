"""LangGraph node: LLM model selection (standalone, no AutoMLAgent)."""
from __future__ import annotations

import json
import re
from typing import Any

import dask.dataframe as dd
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from tools.shared.training_common import LARGE_DATA_ROW_THRESHOLD
from src.utils.logger import Logger

logger = Logger()

# Exactly three user-facing training paths (Dask is automatic for huge data at train time).
VALID_APPROACHES = ("Simple", "Simple+Optuna", "AutoGluon")

SKLEARN_MODELS = {
    "classification": ["RandomForest", "GradientBoosting", "LogisticRegression", "XGBoost"],
    "regression": ["RandomForest", "GradientBoosting", "LinearRegression", "XGBoost"],
}

# AutoGluon 1.5 presets (+ legacy aliases the LLM may still emit).
AUTOGLUON_PRESETS = [
    "medium_quality",
    "medium_quality_faster_train",
    "good_quality",
    "good_quality_faster_inference",
    "high_quality",
    "best_quality",
    "optimize_for_deployment",
]

AUTOGLUON_MODELS = ["GBM", "XGB", "RF", "CAT", "XT", "NN_TORCH", "FASTAI"]

# Default Optuna bounds (LLM may tighten/widen per data profile).
OPTUNA_PARAM_CATALOG = """
Per-model tunable params (use in optuna_settings.search_space):
- RandomForest: n_estimators {int,low,high}, max_depth {int,low,high}
- GradientBoosting: n_estimators {int,low,high}, learning_rate {float,low,high,log:true}
- LogisticRegression: C {float,low,high,log:true}
- LinearRegression: (no tunable params — use defaults)
- XGBoost: n_estimators {int,low,high}, max_depth {int,low,high}, learning_rate {float,low,high,log:true}
"""

_SYSTEM = """You are a tabular ML planner. Reply with ONE JSON object only (no markdown).
You MUST pick exactly one approach from: Simple | Simple+Optuna | AutoGluon (no other values).
Base the choice on DATA PROFILE and the Task / User prompt when provided.
Only use model names and presets from the allowed lists in the user message."""


def _extract_json_block(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    return json.loads(match.group(0))


def _head_df(data: Any, n: int = 3) -> pd.DataFrame:
    if isinstance(data, dd.DataFrame):
        return data.head(n).compute()
    return data.head(n)


def _compact_data_profile(
    data: Any,
    *,
    target_column: str | None,
    problem_type: str | None,
    report: dict,
) -> dict[str, Any]:
    n_rows = int(data.shape[0].compute()) if isinstance(data, dd.DataFrame) else int(data.shape[0])
    n_cols = int(data.shape[1])
    sample = _head_df(data, 3)
    if n_cols > 10:
        keep = [c for c in sample.columns[:9]]
        if target_column and target_column in sample.columns and target_column not in keep:
            keep.append(target_column)
        sample = sample[keep]

    dtypes = {str(c): str(sample[c].dtype) for c in sample.columns}
    missing_pct = float(sample.isnull().mean().mean() * 100) if len(sample.columns) else 0.0

    target_stats: dict[str, Any] = {}
    if target_column and target_column in data.columns:
        target = data[target_column]
        if isinstance(data, dd.DataFrame):
            n_unique = int(target.nunique().compute())
        else:
            n_unique = int(target.nunique())
        target_stats = {"column": target_column, "n_unique": n_unique, "problem_type": problem_type}

    summary = report.get("summary") or report.get("dataset_summary") or {}
    eda_line = ""
    if isinstance(summary, str):
        eda_line = summary[:200]
    elif isinstance(summary, dict):
        eda_line = str(summary)[:200]

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dtypes": dtypes,
        "missing_pct_sample": round(missing_pct, 1),
        "target": target_stats,
        "eda_summary": eda_line,
        "sample_rows": sample.to_dict(orient="records"),
        "large_dataset_dask_auto": n_rows > LARGE_DATA_ROW_THRESHOLD,
    }


def _user_requirements_block(directives: dict) -> str:
    user = directives.get("user") or {}
    lines = []
    controller_task = (user.get("controller_task") or "").strip()
    user_prompt = (user.get("task_prompt") or "").strip()
    if controller_task:
        lines.append(f"Task: {controller_task[:450]}")
    if user_prompt:
        lines.append(f"User prompt: {user_prompt[:400]}")
    time_pref = (user.get("time_preference") or "").strip()
    hw = (user.get("hw_complexity") or "").strip()
    models = user.get("preferred_models") or []
    if time_pref:
        lines.append(f"Time preference: {time_pref}")
    if hw:
        lines.append(f"Hardware: {hw}")
    if models:
        lines.append(f"Preferred models: {', '.join(models[:5])}")
    if not lines:
        return ""
    return "USER PROMPT:\n" + "\n".join(lines)


def _normalize_approach(raw: str) -> str:
    a = raw.strip().lower().replace(" ", "")
    if "autogluon" in a:
        return "AutoGluon"
    if "optuna" in a or a == "simple+optuna":
        return "Simple+Optuna"
    if a == "simple":
        return "Simple"
    return ""


def _filter_models(names: list, problem_type: str) -> list[str]:
    allowed = {m.lower(): m for m in SKLEARN_MODELS.get(problem_type, SKLEARN_MODELS["classification"])}
    out = []
    for name in names:
        key = str(name).strip().lower().replace("_", "").replace("-", "")
        for ak, av in allowed.items():
            if ak.replace(" ", "") in key or key in ak.replace(" ", ""):
                if av not in out:
                    out.append(av)
                break
    return out[:3]


def _filter_ag_models(names: list) -> list[str]:
    allowed = {m.upper() for m in AUTOGLUON_MODELS}
    out = []
    for name in names:
        token = str(name).strip().upper()
        if token in allowed and token not in out:
            out.append(token)
    return out[:4] or ["GBM", "XGB"]


def _filter_preset(raw: str) -> str:
    from tools.training.training_engines import normalize_autogluon_preset

    token = str(raw).strip()
    if token in AUTOGLUON_PRESETS:
        return normalize_autogluon_preset(token)
    for p in AUTOGLUON_PRESETS:
        if p in token:
            return normalize_autogluon_preset(p)
    return normalize_autogluon_preset("good_quality")


def _sanitize_search_space(raw: dict, model_names: list[str]) -> dict:
    out: dict[str, dict] = {}
    if not isinstance(raw, dict):
        return out
    for model in model_names:
        specs = raw.get(model)
        if not isinstance(specs, dict):
            continue
        clean: dict[str, dict] = {}
        for param, bounds in specs.items():
            if not isinstance(bounds, dict) or bounds.get("type") not in ("int", "float"):
                continue
            low, high = bounds.get("low"), bounds.get("high")
            if low is None or high is None:
                continue
            entry: dict[str, Any] = {"type": bounds["type"], "low": low, "high": high}
            if bounds.get("log"):
                entry["log"] = True
            clean[param] = entry
        if clean:
            out[model] = clean
    return out


def _parse_model_decision(
    reasoning: str,
    problem_type: str,
    n_rows: int,
    n_cols: int,
) -> tuple[bool, bool, dict, list[str], str, dict]:
    use_automl = None
    use_dask = n_rows > LARGE_DATA_ROW_THRESHOLD
    automl_config: dict[str, Any] = {}
    optuna_config: dict[str, Any] = {}
    selected_models: list[str] = []
    llm_approach = ""

    try:
        data = _extract_json_block(reasoning)
        if data:
            llm_approach = _normalize_approach(str(data.get("approach", "")))
            if not llm_approach:
                logger.warn(f"Invalid approach '{data.get('approach')}' — must be one of {VALID_APPROACHES}")

            use_automl = llm_approach == "AutoGluon"

            if use_automl:
                settings = data.get("autogluon_settings") or {}
                models = _filter_ag_models(settings.get("models_to_prioritize") or [])
                automl_config = {
                    "models": models,
                    "time_limit": int(settings.get("time_limit_seconds", 180)),
                    "preset": _filter_preset(settings.get("preset_mode", "")),
                }
            else:
                selected_models = _filter_models(
                    data.get("simple_models") or [], problem_type
                )
                if llm_approach == "Simple+Optuna":
                    optuna_raw = data.get("optuna_settings") or {}
                    n_trials = int(optuna_raw.get("n_trials", 30))
                    n_trials = max(5, min(n_trials, 100))
                    search_space = _sanitize_search_space(
                        optuna_raw.get("search_space") or {}, selected_models
                    )
                    optuna_config = {"n_trials": n_trials, "search_space": search_space}
    except Exception as exc:
        logger.warn(f"Model decision JSON parse failed: {exc}")

    if not llm_approach:
        use_automl = False
        llm_approach = "Simple+Optuna"
        logger.warn("Model selection JSON missing — using minimal parse fallback")

    if not automl_config and use_automl:
        automl_config = {
            "models": ["GBM", "XGB"],
            "time_limit": 180,
            "preset": "good_quality_faster_inference",
        }

    if not selected_models and not use_automl:
        selected_models = SKLEARN_MODELS.get(problem_type, ["RandomForest"])[:3]

    return bool(use_automl), use_dask, automl_config, selected_models[:3], llm_approach, optuna_config


def _catalog_block(problem_type: str) -> str:
    models = SKLEARN_MODELS.get(problem_type, SKLEARN_MODELS["classification"])
    return f"""ALLOWED OPTIONS (do not invent others):
- approach: exactly one of {list(VALID_APPROACHES)}
- simple_models: up to 3 from {models}
- autogluon_settings.preset_mode: one of {AUTOGLUON_PRESETS}
- autogluon_settings.models_to_prioritize: subset of {AUTOGLUON_MODELS}
{OPTUNA_PARAM_CATALOG}
If rows>{LARGE_DATA_ROW_THRESHOLD:,}, Dask-XGBoost runs automatically at train time (do not add a 4th approach)."""


def model_selection_node(state: dict, llm) -> dict:
    try:
        directives = state.get("automl_directives") or {}
        report = directives.get("report") or state.get("report") or {}
        task_type = state.get("problem_type") or directives.get("task_type") or "classification"
        target_column = state.get("target_column")

        data = state["data"]
        profile = _compact_data_profile(
            data,
            target_column=target_column,
            problem_type=task_type,
            report=report,
        )
        n_rows = profile["n_rows"]
        n_cols = profile["n_cols"]

        user_block = _user_requirements_block(directives)
        profile_json = json.dumps(profile, separators=(",", ":"))
        catalog = _catalog_block(task_type)

        parts = [f"DATA PROFILE: {profile_json}", catalog]
        if user_block:
            parts.insert(0, user_block)
        body = "\n\n".join(parts)

        prompt = f"""{body}

Pick exactly ONE approach from {list(VALID_APPROACHES)} using the data profile{" and user prompt" if user_block else ""}.
Reasoning must cite profile facts{" and user prompt" if user_block else ""}.

Return JSON (only fields relevant to chosen approach):
{{
  "approach": "Simple|Simple+Optuna|AutoGluon",
  "reasoning": "<=50 words",
  "simple_models": ["..."],
  "optuna_settings": {{
    "n_trials": 30,
    "search_space": {{
      "RandomForest": {{"n_estimators": {{"type":"int","low":50,"high":200}}, "max_depth": {{"type":"int","low":3,"high":15}}}}
    }}
  }},
  "autogluon_settings": {{
    "models_to_prioritize": ["GBM","XGB"],
    "time_limit_seconds": 180,
    "preset_mode": "good_quality_faster_inference"
  }}
}}
For Simple: include simple_models only.
For Simple+Optuna: include simple_models + optuna_settings tuned to data size/complexity.
For AutoGluon: include autogluon_settings with preset and models suited to the profile."""

        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        reasoning = getattr(response, "content", str(response))

        use_automl, use_dask, automl_config, selected_models, llm_approach, optuna_config = (
            _parse_model_decision(reasoning, task_type, n_rows, n_cols)
        )

        state["llm_approach"] = llm_approach
        state["model_selection_reasoning"] = reasoning
        state["use_automl"] = use_automl
        state["use_dask"] = use_dask or bool(state.get("use_dask"))
        state["automl_config"] = automl_config
        state["optuna_config"] = optuna_config
        state["selected_models"] = selected_models
        state["step"] = "models_selected"
        logger.info(
            f"Model selection: approach={llm_approach}, use_automl={use_automl}, "
            f"use_dask={state['use_dask']}, models={selected_models}, optuna={bool(optuna_config)}"
        )
    except Exception as exc:
        state["error"] = f"Failed in model selection: {exc}"
        state["step"] = "error"
    return state
