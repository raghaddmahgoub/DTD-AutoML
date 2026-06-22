"""Feature engineering tool with LLM-suggested feature combinations."""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.shared import ensure_state, merge_state


SUPPORTED_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "mean",
    "absolute_difference",
    "square",
}

OPERATION_ALIASES = {
    "sum": "add",
    "difference": "subtract",
    "product": "multiply",
    "interaction": "multiply",
    "x": "multiply",
    "*": "multiply",
    "ratio": "divide",
    "average": "mean",
    "abs_difference": "absolute_difference",
    "absolute difference": "absolute_difference",
    "squared": "square",
}


@tool
def feature_engineering_execution(task, tool_input, prompt, data_path, llm, state=None):
    """
    Create LLM-suggested feature combinations and append the best 3-4 new
    features, ranked by absolute correlation with the training target.

    Inputs (via tool_input):
    - X_train_path: path to the preprocessed training features
    - X_test_path: path to the preprocessed test features
    - y_train_path: path to the training target
    - top_k: number of new features to append (1-20, default 4)
    - use_llm: whether to ask the LLM for feature recipes (default True)
    - max_candidates: maximum valid generated candidates to evaluate (default 12)
    - output_folder: optional output folder; defaults beside X_train.csv

    Outputs:
    - X_train_engineered.csv and X_test_engineered.csv containing every
      original feature plus only the selected new features
    - feature_engineering_report.json containing all generated recipes,
      their correlations, rejected recipes, and the selected features
    """
    pipeline_state = ensure_state(state, data_path, prompt)

    try:
        tool_input = tool_input if isinstance(tool_input, dict) else {}
        X_train_path = tool_input.get(
            "X_train_path") or pipeline_state.get("X_train_path")
        X_test_path = tool_input.get(
            "X_test_path") or pipeline_state.get("X_test_path")
        y_train_path = tool_input.get(
            "y_train_path") or pipeline_state.get("y_train_path")

        _validate_input_path(X_train_path, "X_train")
        _validate_input_path(X_test_path, "X_test")
        _validate_input_path(y_train_path, "y_train")

        top_k = max(1, min(20, int(tool_input.get("top_k", 4))))
        max_candidates = max(top_k, min(
            30, int(tool_input.get("max_candidates", 12))))
        use_llm = _as_bool(tool_input.get("use_llm", True))
        output_folder = tool_input.get(
            "output_folder") or str(Path(X_train_path).parent)

        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train_frame = pd.read_csv(y_train_path)
        if y_train_frame.empty or y_train_frame.shape[1] == 0:
            raise ValueError("y_train file does not contain a target column")

        raw_y_train = y_train_frame.iloc[:, 0].reset_index(drop=True)
        y_train = pd.to_numeric(raw_y_train, errors="coerce")
        if y_train.notna().sum() < max(2, len(y_train) // 2):
            y_train = pd.Series(
                pd.factorize(raw_y_train.astype("string"), sort=True)[0],
                name=raw_y_train.name,
                dtype=float,
            )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train row counts differ: {len(X_train)} != {len(y_train)}"
            )
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError(
                "X_train and X_test must have the same columns in the same order")

        numeric_columns = X_train.select_dtypes(
            include=[np.number]).columns.tolist()
        if not numeric_columns:
            raise ValueError(
                "No numeric preprocessed columns are available for feature combinations")

        original_correlations = _calculate_correlations(
            X_train[numeric_columns], y_train)
        recipes = []
        llm_reasoning = ""
        suggestion_source = "deterministic_fallback"

        if use_llm and llm is not None:
            recipes, llm_reasoning = _get_llm_feature_recipes(
                numeric_columns=numeric_columns,
                dtypes={column: str(X_train[column].dtype)
                        for column in numeric_columns},
                correlations=original_correlations,
                target_name=str(y_train_frame.columns[0]),
                llm=llm,
                max_candidates=max_candidates,
                task=str(task or ""),
                prompt=str(prompt or pipeline_state.get("prompt", "")),
            )
            if recipes:
                recipes = [
                    {**recipe, "source": "llm"}
                    for recipe in recipes
                    if isinstance(recipe, dict)
                ]
                if recipes:
                    suggestion_source = "llm"

        if not recipes:
            recipes = _build_fallback_recipes(
                numeric_columns=numeric_columns,
                correlations=original_correlations,
                max_candidates=max_candidates,
            )
            if use_llm:
                llm_reasoning = (
                    f"{llm_reasoning} "
                    "The LLM returned no usable recipe objects; deterministic "
                    "fallback recipes were used."
                ).strip()
            elif not use_llm:
                llm_reasoning = "LLM feature suggestions were disabled; deterministic fallback recipes were used."

        generated_train, generated_test, generated_features, rejected_features = (
            _materialize_recipes(
                X_train=X_train,
                X_test=X_test,
                recipes=recipes,
                max_candidates=max_candidates,
            )
        )

        if suggestion_source == "llm" and len(generated_features) < top_k:
            fallback_train, fallback_test, fallback_features, fallback_rejected = (
                _materialize_recipes(
                    X_train=X_train,
                    X_test=X_test,
                    recipes=_build_fallback_recipes(
                        numeric_columns=numeric_columns,
                        correlations=original_correlations,
                        max_candidates=max_candidates,
                    ),
                    max_candidates=max_candidates - len(generated_features),
                    reserved_names=set(generated_train.columns),
                )
            )
            generated_train = pd.concat(
                [generated_train, fallback_train], axis=1
            )
            generated_test = pd.concat(
                [generated_test, fallback_test], axis=1
            )
            generated_features.extend(fallback_features)
            rejected_features.extend(fallback_rejected)
            llm_reasoning = (
                f"{llm_reasoning} "
                "Fallback recipes supplemented the LLM output because fewer than "
                f"{top_k} valid LLM features were generated."
            ).strip()

        if generated_train.empty:
            raise ValueError(
                "No valid new feature columns could be generated. "
                f"Rejected recipes: {rejected_features}"
            )

        generated_correlations = _calculate_correlations(
            generated_train, y_train)
        ranked_features = sorted(
            generated_train.columns,
            key=lambda column: (-abs(generated_correlations.get(column, 0.0)), column),
        )
        selected_features = ranked_features[: min(top_k, len(ranked_features))]

        X_train_engineered = pd.concat(
            [X_train, generated_train[selected_features]], axis=1
        )
        X_test_engineered = pd.concat(
            [X_test, generated_test[selected_features]], axis=1
        )

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        X_train_engineered_path = str(output_path / "X_train_engineered.csv")
        X_test_engineered_path = str(output_path / "X_test_engineered.csv")
        report_path = str(output_path / "feature_engineering_report.json")

        X_train_engineered.to_csv(X_train_engineered_path, index=False)
        X_test_engineered.to_csv(X_test_engineered_path, index=False)

        generated_by_name = {item["name"]: item for item in generated_features}
        evaluated_features = []
        for feature_name in ranked_features:
            item = dict(generated_by_name[feature_name])
            item["correlation_with_target"] = float(
                generated_correlations.get(feature_name, 0.0)
            )
            item["absolute_correlation"] = float(
                abs(generated_correlations.get(feature_name, 0.0))
            )
            item["selected"] = feature_name in selected_features
            evaluated_features.append(item)

        report = {
            "status": "success",
            "task": task,
            "target_column": str(y_train_frame.columns[0]),
            "suggestion_source": suggestion_source,
            "llm_enabled": use_llm,
            "llm_reasoning": llm_reasoning,
            "original_columns": X_train.columns.tolist(),
            "original_feature_count": int(X_train.shape[1]),
            "generated_features": evaluated_features,
            "llm_generated_features": [
                item for item in evaluated_features if item["source"] == "llm"
            ],
            "fallback_generated_features": [
                item
                for item in evaluated_features
                if item["source"] == "deterministic_fallback"
            ],
            "rejected_features": rejected_features,
            "selected_features": selected_features,
            "selected_correlations": {
                column: float(generated_correlations[column])
                for column in selected_features
            },
            "selection_method": "largest absolute Pearson correlation with y_train",
            "requested_top_k": top_k,
            "generated_candidate_count": len(evaluated_features),
            "selected_feature_count": len(selected_features),
            "final_columns": X_train_engineered.columns.tolist(),
            "final_feature_count": int(X_train_engineered.shape[1]),
            "X_train_engineered_shape": list(X_train_engineered.shape),
            "X_test_engineered_shape": list(X_test_engineered.shape),
            "X_train_engineered_path": X_train_engineered_path,
            "X_test_engineered_path": X_test_engineered_path,
        }

        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=False)

        feature_output = {
            "X_train_engineered_path": X_train_engineered_path,
            "X_test_engineered_path": X_test_engineered_path,
            "feature_report_path": report_path,
            "feature_summary_path": report_path,
            "generated_features": evaluated_features,
            "selected_features": selected_features,
            "selected_correlations": report["selected_correlations"],
        }
        pipeline_state = merge_state(
            pipeline_state,
            {
                "step": "feature_engineering_complete",
                "status": "success",
                "feature_engineering_output": feature_output,
                "X_train_engineered_path": X_train_engineered_path,
                "X_test_engineered_path": X_test_engineered_path,
                "feature_names": X_train_engineered.columns.tolist(),
                "n_features": int(X_train_engineered.shape[1]),
            },
        )

        result = {
            "status": "success",
            "message": (
                f"Feature engineering appended {len(selected_features)} new features "
                f"to {X_train.shape[1]} original features"
            ),
            "feature_engineering_output": feature_output,
        }
        return result, pipeline_state

    except Exception as exc:
        import traceback

        error_msg = f"Feature engineering failed: {exc}\n{traceback.format_exc()}"
        pipeline_state = merge_state(
            pipeline_state,
            {
                "step": "feature_engineering_failed",
                "status": "error",
                "error": error_msg,
            },
        )
        return {"status": "error", "error": error_msg}, pipeline_state


def _validate_input_path(path_value: Any, label: str) -> None:
    if not path_value:
        raise ValueError(f"{label}_path was not provided")
    if not Path(path_value).exists():
        raise FileNotFoundError(f"{label} not found: {path_value}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _calculate_correlations(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calculate finite Pearson correlations between numeric features and target."""
    correlations: dict[str, float] = {}
    numeric_y = pd.to_numeric(y, errors="coerce")
    for column in X.columns:
        numeric_x = pd.to_numeric(X[column], errors="coerce")
        valid = numeric_x.notna() & numeric_y.notna()
        if valid.sum() < 2 or numeric_x[valid].nunique() < 2 or numeric_y[valid].nunique() < 2:
            correlations[column] = 0.0
            continue
        correlation = numeric_x[valid].corr(numeric_y[valid])
        correlations[column] = float(
            correlation) if pd.notna(correlation) else 0.0
    return correlations


def _get_llm_feature_recipes(
    *,
    numeric_columns: list[str],
    dtypes: dict[str, str],
    correlations: dict[str, float],
    target_name: str,
    llm: Any,
    max_candidates: int,
    task: str = "",
    prompt: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Ask the LLM for structured, executable feature-combination recipes."""
    column_info = [
        {
            "name": column,
            "dtype": dtypes[column],
            "correlation_with_target": round(correlations.get(column, 0.0), 6),
        }
        for column in numeric_columns
    ]
    llm_prompt = f"""
You are creating candidate numeric features for a machine-learning dataset.
The target is {target_name!r}. The listed columns are already preprocessed numeric columns.

Suggest between 8 and {max_candidates} useful NEW feature combinations. Do not select or
rename existing columns. Each feature must be reproducible on test data without the target.

Allowed operations:
- add: columns[0] + columns[1]
- subtract: columns[0] - columns[1]
- multiply: columns[0] * columns[1]
- divide: columns[0] / columns[1]
- mean: average of two columns
- absolute_difference: absolute difference of two columns
- square: square of one column

Available columns:
{json.dumps(column_info, indent=2)}

Respect the user task when choosing feature combinations.

Task: {(task or "")[:400]}
User prompt: {(prompt or "")[:400]}

Return only one JSON object with this exact shape:
{{
  "features": [
    {{
      "name": "short_unique_feature_name",
      "operation": "multiply",
      "columns": ["exact_column_1", "exact_column_2"],
      "reason": "why this combination may help"
    }}
  ],
  "reasoning": "brief overall reasoning"
}}
"""

    try:
        response = llm.invoke(llm_prompt)
        content = getattr(response, "content", response)
        parsed = _extract_json_object(content)
        raw_features = parsed.get("features", [])
        if not isinstance(raw_features, list):
            return [], "The LLM response did not contain a feature list."
        return raw_features[:max_candidates], str(parsed.get("reasoning", ""))
    except Exception as exc:
        return [], f"LLM feature suggestion failed: {exc}"


def _extract_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object")
    parsed = json.loads(text[start: end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _build_fallback_recipes(
    *,
    numeric_columns: list[str],
    correlations: dict[str, float],
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Generate a bounded deterministic recipe set when the LLM is unavailable."""
    ranked_columns = sorted(
        numeric_columns,
        key=lambda column: (-abs(correlations.get(column, 0.0)), column),
    )
    source_columns = ranked_columns[: min(6, len(ranked_columns))]
    recipes: list[dict[str, Any]] = []

    for left, right in combinations(source_columns, 2):
        for operation in ("multiply", "divide", "add", "absolute_difference"):
            recipes.append(
                {
                    "name": f"fe_{_safe_name(left)}_{operation}_{_safe_name(right)}",
                    "operation": operation,
                    "columns": [left, right],
                    "reason": "Deterministic fallback using highly target-correlated source columns.",
                    "source": "deterministic_fallback",
                }
            )
            if len(recipes) >= max_candidates:
                return recipes

    for column in source_columns:
        recipes.append(
            {
                "name": f"fe_{_safe_name(column)}_square",
                "operation": "square",
                "columns": [column],
                "reason": "Deterministic fallback nonlinear transformation.",
                "source": "deterministic_fallback",
            }
        )
        if len(recipes) >= max_candidates:
            break
    return recipes


def _materialize_recipes(
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    recipes: list[dict[str, Any]],
    max_candidates: int,
    reserved_names: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    train_features: dict[str, pd.Series] = {}
    test_features: dict[str, pd.Series] = {}
    generated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_names = set(X_train.columns)
    used_names.update(reserved_names or set())
    used_signatures: set[tuple[str, tuple[str, ...]]] = set()

    for raw_recipe in recipes:
        if len(generated) >= max_candidates:
            break
        try:
            recipe = _normalize_recipe(
                raw_recipe, X_train.columns.tolist(), used_names)
            signature = _recipe_signature(recipe)
            if signature in used_signatures:
                raise ValueError(
                    "duplicate feature recipe; reordered commutative operations "
                    "are treated as the same feature"
                )
            train_series = _apply_recipe(X_train, recipe)
            test_series = _apply_recipe(X_test, recipe)

            train_series = pd.to_numeric(train_series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            test_series = pd.to_numeric(test_series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            fill_value = train_series.median(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0.0
            train_series = train_series.fillna(float(fill_value)).astype(float)
            test_series = test_series.fillna(float(fill_value)).astype(float)

            feature_scale = max(
                1.0,
                float(train_series.abs().max(skipna=True)),
            )
            feature_std = float(train_series.std(ddof=0))
            if (
                train_series.nunique(dropna=False) < 2
                or not np.isfinite(feature_std)
                or feature_std <= 1e-12 * feature_scale
            ):
                raise ValueError(
                    "generated feature is constant or numerically near-constant "
                    "in training data"
                )

            name = recipe["name"]
            train_features[name] = train_series
            test_features[name] = test_series
            used_names.add(name)
            used_signatures.add(signature)
            generated.append({**recipe, "fill_value": float(fill_value)})
        except Exception as exc:
            rejected.append({"recipe": raw_recipe, "reason": str(exc)})

    return (
        pd.DataFrame(train_features, index=X_train.index),
        pd.DataFrame(test_features, index=X_test.index),
        generated,
        rejected,
    )


def _normalize_recipe(
    raw_recipe: Any,
    available_columns: list[str],
    used_names: set[str],
) -> dict[str, Any]:
    if not isinstance(raw_recipe, dict):
        raise ValueError("recipe must be a JSON object")

    operation = str(raw_recipe.get("operation", "")).strip().lower()
    operation = OPERATION_ALIASES.get(operation, operation)
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported operation: {operation!r}")

    columns = raw_recipe.get("columns")
    if not isinstance(columns, list):
        columns = [
            value
            for value in (raw_recipe.get("left"), raw_recipe.get("right"))
            if value is not None
        ]
    columns = [str(column) for column in columns]
    required_count = 1 if operation == "square" else 2
    if len(columns) != required_count:
        raise ValueError(
            f"{operation} requires exactly {required_count} source column(s)")
    missing = [column for column in columns if column not in available_columns]
    if missing:
        raise ValueError(f"unknown source columns: {missing}")

    requested_name = str(raw_recipe.get("name", "")).strip()
    default_name = f"fe_{'_'.join(_safe_name(column) for column in columns)}_{operation}"
    name = _safe_name(requested_name) if requested_name else default_name
    if not name.lower().startswith("fe_"):
        name = f"fe_{name}"
    if name in used_names:
        suffix = 2
        base_name = name
        while f"{base_name}_{suffix}" in used_names:
            suffix += 1
        name = f"{base_name}_{suffix}"

    return {
        "name": name,
        "operation": operation,
        "columns": columns,
        "reason": str(raw_recipe.get("reason", "")).strip(),
        "source": str(
            raw_recipe.get("source", "deterministic_fallback")
        ).strip(),
    }


def _apply_recipe(frame: pd.DataFrame, recipe: dict[str, Any]) -> pd.Series:
    operation = recipe["operation"]
    columns = recipe["columns"]
    left = pd.to_numeric(frame[columns[0]], errors="coerce")

    if operation == "square":
        return left.pow(2)

    right = pd.to_numeric(frame[columns[1]], errors="coerce")
    if operation == "add":
        return left + right
    if operation == "subtract":
        return left - right
    if operation == "multiply":
        return left * right
    if operation == "divide":
        safe_denominator = right.where(right.abs() > 1e-12)
        return left / safe_denominator
    if operation == "mean":
        return (left + right) / 2.0
    if operation == "absolute_difference":
        return (left - right).abs()
    raise ValueError(f"unsupported operation: {operation}")


def _recipe_signature(recipe: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Canonicalize recipes so A*B and B*A cannot both be generated."""
    operation = recipe["operation"]
    columns = tuple(recipe["columns"])
    if operation in {"add", "multiply", "mean", "absolute_difference"}:
        columns = tuple(sorted(columns))
    return operation, columns


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "generated_feature"
