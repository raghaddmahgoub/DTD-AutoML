"""Categorical encoding tool fitted on training data."""
from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state
from tools.preprocessing_common import load_split, save_split


@tool
def preprocessing_encoding(task, tool_input, prompt, data_path, llm, state=None):
    """Encode categorical columns consistently while honoring no-encode overrides."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        X_train, X_test, y_train, y_test = load_split(tool_input)
        train_frames = []
        test_frames = []
        actions = {}

        for column in X_train.columns:
            decision = plan["columns"].get(column, {})
            method = decision.get("encoding", "none")
            if decision.get("type") == "numeric":
                train_frames.append(X_train[[column]])
                test_frames.append(X_test[[column]])
                actions[column] = {"method": "none", "output_columns": [column]}
                continue

            train = X_train[column].astype("string")
            test = X_test[column].astype("string")
            if method == "onehot":
                categories = sorted(train.dropna().unique().tolist())
                train_encoded = pd.DataFrame(index=X_train.index)
                test_encoded = pd.DataFrame(index=X_test.index)
                for category in categories:
                    name = f"{column}__{category}"
                    train_encoded[name] = (train == category).astype(int)
                    test_encoded[name] = (test == category).astype(int)
            elif method in {"label", "ordinal"}:
                categories = sorted(train.dropna().unique().tolist())
                mapping = {value: index for index, value in enumerate(categories)}
                train_encoded = train.map(mapping).fillna(-1).astype(int).to_frame(column)
                test_encoded = test.map(mapping).fillna(-1).astype(int).to_frame(column)
            elif method == "frequency":
                frequencies = train.value_counts(normalize=True, dropna=False)
                output_name = f"{column}__frequency"
                train_encoded = train.map(frequencies).fillna(0.0).to_frame(output_name)
                test_encoded = test.map(frequencies).fillna(0.0).to_frame(output_name)
            else:
                train_encoded = train.to_frame(column)
                test_encoded = test.to_frame(column)

            train_frames.append(train_encoded)
            test_frames.append(test_encoded)
            actions[column] = {
                "method": method,
                "output_columns": train_encoded.columns.tolist(),
            }

        encoded_train = pd.concat(train_frames, axis=1) if train_frames else pd.DataFrame(index=X_train.index)
        encoded_test = pd.concat(test_frames, axis=1) if test_frames else pd.DataFrame(index=X_test.index)
        paths = save_split(
            encoded_train.reset_index(drop=True),
            encoded_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            tool_input["output_folder"],
            "encoded",
        )
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "encoding_actions": actions,
                "step": "encoding_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "actions": actions}, pipeline_state
    except Exception as exc:
        message = f"Encoding failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "encoding_failed", "status": "error", "error": message}
        )
