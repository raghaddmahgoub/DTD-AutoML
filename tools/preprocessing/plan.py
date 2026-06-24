"""LLM-assisted preprocessing plan tool with enforceable user constraints."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from langchain_core.tools import tool

from tools.shared import ensure_state, merge_state
from .common import (
    extract_json_object,
    mentioned_columns,
    read_json,
    write_json,
)


ALLOWED = {
    "missing": {"mean", "median", "mode", "constant", "knn", "drop_rows", "none"},
    "outlier": {"keep", "clip_iqr", "remove_rows", "log_transform"},
    "encoding": {"none", "onehot", "ordinal", "label", "frequency"},
    "scaling": {"none", "standard", "minmax", "robust", "quantile", "power"},
    "normalization": {"none", "l1", "l2", "max"},
    "balancing": {
        "none",
        "class_weight",
        "oversample",
        "undersample",
        "smote",
        "adasyn",
    },
}


def _section(value: Any, default_method: str) -> dict[str, Any]:
    """Normalize LLM sections such as 'standard' or null into method objects."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"method": value.strip().lower()}
    return {"method": default_method}


def _feature_section(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return {**default, **value}
    if isinstance(value, (int, float, str)):
        try:
            return {**default, "top_k": int(value)}
        except (TypeError, ValueError):
            return dict(default)
    return dict(default)


def _constraints(prompt: str, columns: list[str]) -> dict[str, Any]:
    explicit_drop = mentioned_columns(prompt, columns, ["drop"])
    global_no_encode = bool(
        re.search(
            r"(?:do\s+not|don't|never)\s+encode\s+"
            r"(?:the\s+)?(?:rest|remaining|other\s+columns|anything\s+else)",
            prompt.casefold(),
        )
    )
    disable_feature_engineering = bool(
        re.search(
            r"(?:(?:do\s+not|don\'t|never)\s+(?:run\s+|do\s+|perform\s+|use\s+)?feature\s+engineering|(?:without|no)\s+feature\s+engineering)",
            prompt.casefold(),
        )
    )
    return {
        "drop": explicit_drop,
        "disable_feature_engineering": disable_feature_engineering,
        "do_not_drop": mentioned_columns(
            prompt, columns, ["do not drop", "don't drop", "never drop", "keep"]
        ),
        "do_not_encode": mentioned_columns(
            prompt, columns, ["do not encode", "don't encode", "never encode"]
        ),
        "do_not_scale": mentioned_columns(
            prompt, columns, ["do not scale", "don't scale", "never scale"]
        ),
        "leave_missing": mentioned_columns(
            prompt,
            columns,
            ["do not impute", "don't impute", "leave missing", "keep missing"],
        ),
        "global_do_not_encode": global_no_encode,
    }


def _default_plan(evidence: dict[str, Any], prompt: str, top_k: int) -> dict[str, Any]:
    target = evidence["target_column"]
    columns = {}
    for column, profile in evidence["column_profiles"].items():
        if column == target:
            continue
        numeric = profile["numeric_parse_ratio"] >= 0.9
        high_cardinality = profile["unique_ratio"] > 0.95 and profile["unique_count"] > 20
        columns[column] = {
            "drop": bool(high_cardinality),
            "type": "numeric" if numeric else "categorical",
            "missing": "median" if numeric else "mode",
            "outlier": "clip_iqr" if numeric else "keep",
            "encoding": "none" if numeric else (
                "onehot" if profile["unique_count"] <= 15 else "frequency"
            ),
            "reason": (
                "Likely identifier/high-cardinality column"
                if high_cardinality
                else "Evidence-based default"
            ),
        }
    balancing = (
        "class_weight"
        if evidence["task_type"] == "classification"
        and evidence["target"]["imbalance_ratio"] >= 2.0
        else "none"
    )
    return {
        "summary": "Evidence-based preprocessing plan",
        "user_request": prompt,
        "duplicates": "drop",
        "columns": columns,
        "scaling": {"method": "standard"},
        "normalization": {"method": "none"},
        "balancing": {"method": balancing},
        "feature_engineering": {"enabled": True, "top_k": top_k, "max_candidates": max(12, top_k * 3)},
        "warnings": [],
    }


def _merge_llm(default: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(suggested, dict):
        return default
    plan = {**default, **{key: value for key, value in suggested.items() if key != "columns"}}
    plan["columns"] = {name: dict(value) for name, value in default["columns"].items()}
    suggested_columns = suggested.get("columns", {})
    if isinstance(suggested_columns, list):
        normalized_columns = {}
        for item in suggested_columns:
            if not isinstance(item, dict):
                continue
            column_name = (
                item.get("column")
                or item.get("name")
                or item.get("column_name")
            )
            if column_name:
                normalized_columns[str(column_name)] = {
                    key: value
                    for key, value in item.items()
                    if key not in {"column", "name", "column_name"}
                }
        suggested_columns = normalized_columns
    if not isinstance(suggested_columns, dict):
        suggested_columns = {}
    for column, value in suggested_columns.items():
        if column in plan["columns"] and isinstance(value, dict):
            plan["columns"][column].update(value)
    plan["scaling"] = _section(plan.get("scaling"), "standard")
    plan["normalization"] = _section(plan.get("normalization"), "none")
    plan["balancing"] = _section(plan.get("balancing"), "none")
    plan["feature_engineering"] = _feature_section(
        plan.get("feature_engineering"),
        default["feature_engineering"],
    )
    duplicate_value = plan.get("duplicates")
    if isinstance(duplicate_value, dict):
        duplicate_value = (
            duplicate_value.get("action")
            or duplicate_value.get("method")
        )
    plan["duplicates"] = (
        duplicate_value
        if isinstance(duplicate_value, str)
        and duplicate_value in {"drop", "keep"}
        else default["duplicates"]
    )
    if not isinstance(plan.get("warnings"), list):
        plan["warnings"] = (
            [str(plan["warnings"])] if plan.get("warnings") else []
        )
    return plan


def _validate(plan: dict[str, Any], default: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        plan = default
    if not isinstance(plan.get("columns"), dict):
        plan["columns"] = {
            name: dict(value) for name, value in default["columns"].items()
        }
    for column, fallback in default["columns"].items():
        decision = plan["columns"].get(column)
        if not isinstance(decision, dict):
            plan["columns"][column] = dict(fallback)

    for column, decision in plan["columns"].items():
        if column not in default["columns"]:
            continue
        fallback = default["columns"][column]
        if decision.get("type") not in {"numeric", "categorical", "datetime", "text"}:
            decision["type"] = fallback["type"]
        if fallback["type"] == "numeric":
            decision["type"] = "numeric"
        for key in ("missing", "outlier", "encoding"):
            if decision.get(key) not in ALLOWED[key]:
                decision[key] = fallback[key]
        decision["drop"] = bool(decision.get("drop", False))

    for group, fallback_method in (
        ("scaling", "standard"),
        ("normalization", "none"),
        ("balancing", "none"),
    ):
        plan[group] = _section(plan.get(group), fallback_method)
        method = plan[group].get("method")
        if method not in ALLOWED[group]:
            plan[group] = {"method": fallback_method}

    raw_warnings = plan.get("warnings")
    if isinstance(raw_warnings, list):
        warnings = [str(item) for item in raw_warnings]
    elif raw_warnings:
        warnings = [str(raw_warnings)]
    else:
        warnings = []
    for column in constraints["drop"]:
        if column in plan["columns"]:
            plan["columns"][column]["drop"] = True
            plan["columns"][column]["user_override"] = "drop"
            plan["columns"][column]["reason"] = "Explicit user request"
    for column in constraints["do_not_drop"]:
        if column in plan["columns"]:
            plan["columns"][column]["drop"] = False
            plan["columns"][column]["user_override"] = "do_not_drop"
    for column in constraints["do_not_encode"]:
        if column in plan["columns"]:
            plan["columns"][column]["encoding"] = "none"
            plan["columns"][column]["user_override"] = "do_not_encode"
            if plan["columns"][column]["type"] != "numeric":
                warnings.append(
                    f"{column} will remain non-numeric because the user prohibited encoding."
                )
    if constraints["global_do_not_encode"]:
        for column, decision in plan["columns"].items():
            if not decision.get("drop"):
                decision["encoding"] = "none"
                decision["user_override"] = "global_do_not_encode"
                if decision.get("type") != "numeric":
                    warnings.append(
                        f"{column} will remain non-numeric because the user prohibited encoding the remaining columns."
                    )
    for column in constraints["leave_missing"]:
        if column in plan["columns"]:
            plan["columns"][column]["missing"] = "none"
            plan["columns"][column]["user_override"] = "leave_missing"
            warnings.append(
                f"{column} may retain missing values because the user prohibited imputation."
            )
    for column in constraints["do_not_scale"]:
        if column in plan["columns"]:
            plan["columns"][column]["skip_scaling"] = True
            plan["columns"][column]["user_override"] = "do_not_scale"

    plan["user_constraints"] = constraints
    plan["warnings"] = sorted(set(warnings))
    plan["feature_engineering"] = _feature_section(
        plan.get("feature_engineering"),
        default["feature_engineering"],
    )
    feature_plan = plan["feature_engineering"]
    feature_plan["enabled"] = True
    try:
        feature_top_k = int(feature_plan.get("top_k", 4))
    except (TypeError, ValueError):
        feature_top_k = int(default["feature_engineering"]["top_k"])
    feature_plan["top_k"] = max(0, min(20, feature_top_k))
    feature_plan["enabled"] = feature_plan["top_k"] > 0
    try:
        max_candidates = int(feature_plan.get("max_candidates", 12))
    except (TypeError, ValueError):
        max_candidates = int(default["feature_engineering"]["max_candidates"])
    feature_plan["max_candidates"] = max(
        feature_plan["top_k"], min(60, max_candidates)
    )
    return plan


@tool
def preprocessing_plan(task, tool_input, prompt, data_path, llm, state=None):
    """Build a JSON preprocessing plan from evidence and free-text instructions."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        evidence = (
            tool_input.get("evidence")
            or pipeline_state.get("preprocessing_evidence")
            or read_json(tool_input["evidence_path"])
        )
        top_k = max(0, min(20, int(tool_input.get("feature_top_k", 4))))
        constraints = _constraints(prompt, evidence["columns"])
        if constraints.get("disable_feature_engineering"):
            top_k = 0
        default = _default_plan(evidence, prompt, top_k)
        plan = default
        llm_error = ""
        if tool_input.get("use_llm", True) and llm is not None:
            llm_prompt = f"""
You are a preprocessing planner. Respect the user's request literally, especially
instructions not to drop, encode, scale, or impute specific columns.

User request:
{prompt}

Hard user constraints extracted from that request:
{constraints}

Dataset evidence:
{evidence}

Return only JSON with:
- summary
- duplicates: drop|keep
- columns: one object per non-target column with drop, type, missing, outlier,
  encoding, reason
- scaling: method
- normalization: method
- balancing: method
- feature_engineering: enabled=true, top_k, max_candidates
- warnings

Allowed values:
missing={sorted(ALLOWED["missing"])}
outlier={sorted(ALLOWED["outlier"])}
encoding={sorted(ALLOWED["encoding"])}
scaling={sorted(ALLOWED["scaling"])}
normalization={sorted(ALLOWED["normalization"])}
balancing={sorted(ALLOWED["balancing"])}
"""
            try:
                response = llm.invoke(llm_prompt)
                suggested = extract_json_object(getattr(response, "content", response))
                plan = _merge_llm(default, suggested)
            except Exception as exc:
                llm_error = str(exc)

        plan = _validate(plan, default, constraints)
        plan["feature_engineering"]["top_k"] = top_k
        plan["feature_engineering"]["enabled"] = top_k > 0
        try:
            planned_candidates = int(
                plan["feature_engineering"].get(
                    "max_candidates", max(12, top_k * 3)
                )
            )
        except (TypeError, ValueError):
            planned_candidates = max(12, top_k * 3)
        plan["feature_engineering"]["max_candidates"] = max(
            top_k,
            min(60, planned_candidates),
        )
        plan["planner"] = "llm" if not llm_error and llm is not None else "deterministic"
        if llm_error:
            plan["warnings"].append(f"LLM planning failed; defaults were used: {llm_error}")

        output_folder = Path(
            tool_input.get("output_folder")
            or Path("Output") / "Preprocessing" / Path(data_path).stem
        )
        plan_path = write_json(output_folder / "preprocessing_plan.json", plan)
        pipeline_state = merge_state(
            pipeline_state,
            {
                "preprocessing_plan": plan,
                "preprocessing_plan_path": plan_path,
                "step": "preprocessing_plan_ready",
                "status": "success",
            },
        )
        return {"status": "success", "plan": plan, "plan_path": plan_path}, pipeline_state
    except Exception as exc:
        message = f"Preprocessing plan failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "preprocessing_plan_failed", "status": "error", "error": message}
        )
