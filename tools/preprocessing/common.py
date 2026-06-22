"""Shared helpers for the dynamic preprocessing tools."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: str | Path, value: Any) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    return str(output)


def extract_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The LLM response did not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The LLM response must be a JSON object")
    return parsed


def stage_paths(output_folder: str | Path, stage: str) -> dict[str, str]:
    folder = Path(output_folder) / ".preprocessing_work"
    folder.mkdir(parents=True, exist_ok=True)
    return {
        "X_train_path": str(folder / f"X_train_{stage}.csv"),
        "X_test_path": str(folder / f"X_test_{stage}.csv"),
        "y_train_path": str(folder / f"y_train_{stage}.csv"),
        "y_test_path": str(folder / f"y_test_{stage}.csv"),
    }


def copy_targets(
    y_train_path: str,
    y_test_path: str,
    output_paths: dict[str, str],
) -> None:
    pd.read_csv(y_train_path).to_csv(output_paths["y_train_path"], index=False)
    pd.read_csv(y_test_path).to_csv(output_paths["y_test_path"], index=False)


def load_split(tool_input: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(tool_input["X_train_path"])
    X_test = pd.read_csv(tool_input["X_test_path"])
    y_train = pd.read_csv(tool_input["y_train_path"]).iloc[:, 0]
    y_test = pd.read_csv(tool_input["y_test_path"]).iloc[:, 0]
    return X_train, X_test, y_train, y_test


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_folder: str | Path,
    stage: str,
) -> dict[str, str]:
    paths = stage_paths(output_folder, stage)
    X_train.to_csv(paths["X_train_path"], index=False)
    X_test.to_csv(paths["X_test_path"], index=False)
    y_train.to_frame(name=y_train.name or "target").to_csv(
        paths["y_train_path"], index=False
    )
    y_test.to_frame(name=y_test.name or "target").to_csv(
        paths["y_test_path"], index=False
    )
    return paths


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def column_lookup(columns: list[str]) -> dict[str, str]:
    return {column.casefold(): column for column in columns}


def mentioned_columns(text: str, columns: list[str], action_words: list[str]) -> list[str]:
    """Find columns in clauses containing an action such as 'do not drop'."""
    lowered = text.casefold()
    found: list[str] = []
    directive_phrases = [
        "do not drop", "don't drop", "never drop", "drop", "keep",
        "do not encode", "don't encode", "never encode", "encode",
        "do not scale", "don't scale", "never scale", "scale",
        "do not impute", "don't impute", "leave missing", "keep missing", "impute",
        "normalize", "balance",
    ]
    directive_pattern = re.compile(
        "|".join(
            sorted(
                (
                    re.escape(phrase).replace(r"\ ", r"\s+")
                    for phrase in directive_phrases
                ),
                key=len,
                reverse=True,
            )
        )
    )
    directive_starts = [match.start() for match in directive_pattern.finditer(lowered)]

    for action in action_words:
        if action.casefold() == "keep":
            action_pattern = re.compile(r"\bkeep\b(?!\s+missing)")
        elif action.casefold() == "drop":
            action_pattern = re.compile(
                r"(?<!do not )(?<!don't )(?<!never )\bdrop\b"
            )
        else:
            action_pattern = re.compile(
                re.escape(action.casefold()).replace(r"\ ", r"\s+")
            )
        for match in action_pattern.finditer(lowered):
            punctuation = [
                index
                for mark in ".;,\n"
                if (index := lowered.find(mark, match.end())) >= 0
            ]
            later_directives = [
                index for index in directive_starts if index > match.start()
            ]
            end = min(
                punctuation + later_directives + [min(len(lowered), match.end() + 120)]
            )
            clause = lowered[match.start() : end]
            for column in sorted(columns, key=len, reverse=True):
                if re.search(rf"\b{re.escape(column.casefold())}\b", clause):
                    found.append(column)
    return sorted(set(found), key=columns.index)
