"""Training-target balancing tool."""
from __future__ import annotations

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.neighbors import NearestNeighbors

from tools.pipeline_state import ensure_state, merge_state
from tools.preprocessing_common import load_split, save_split


def _smote(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)
    frames = [X.copy()]
    targets = [y.reset_index(drop=True)]
    counts = y.value_counts()
    target_size = int(counts.max())
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1] != X.shape[1] or X.isna().any().any():
        raise ValueError("SMOTE requires fully numeric data with no missing values")

    for label, count in counts.items():
        needed = target_size - int(count)
        if needed <= 0:
            continue
        group = X.loc[y == label].reset_index(drop=True)
        if len(group) < 2:
            raise ValueError(f"SMOTE needs at least two rows for class {label!r}")
        neighbor_count = min(6, len(group))
        model = NearestNeighbors(n_neighbors=neighbor_count).fit(group)
        neighbor_indices = model.kneighbors(return_distance=False)
        synthetic = []
        for _ in range(needed):
            base_index = int(rng.integers(0, len(group)))
            choices = neighbor_indices[base_index][1:]
            neighbor_index = int(rng.choice(choices)) if len(choices) else base_index
            gap = float(rng.random())
            synthetic.append(
                group.iloc[base_index].to_numpy(dtype=float)
                + gap
                * (
                    group.iloc[neighbor_index].to_numpy(dtype=float)
                    - group.iloc[base_index].to_numpy(dtype=float)
                )
            )
        frames.append(pd.DataFrame(synthetic, columns=X.columns))
        targets.append(pd.Series([label] * needed, name=y.name))
    return pd.concat(frames, ignore_index=True), pd.concat(targets, ignore_index=True)


def _adasyn(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    """Small dependency-free ADASYN implementation for numeric binary/multiclass data."""
    rng = np.random.default_rng(random_state)
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1] != X.shape[1] or X.isna().any().any():
        raise ValueError("ADASYN requires fully numeric data with no missing values")

    counts = y.value_counts()
    target_size = int(counts.max())
    generated_frames = [X.reset_index(drop=True)]
    generated_targets = [y.reset_index(drop=True)]
    all_values = X.reset_index(drop=True)
    all_targets = y.reset_index(drop=True)
    full_neighbors = NearestNeighbors(
        n_neighbors=min(6, len(all_values))
    ).fit(all_values)
    full_indices = full_neighbors.kneighbors(return_distance=False)

    for label, count in counts.items():
        needed = target_size - int(count)
        if needed <= 0:
            continue
        class_indices = np.flatnonzero(all_targets.to_numpy() == label)
        if len(class_indices) < 2:
            raise ValueError(f"ADASYN needs at least two rows for class {label!r}")
        difficulty = []
        for index in class_indices:
            neighbors = full_indices[index][1:]
            different = sum(all_targets.iloc[n] != label for n in neighbors)
            difficulty.append(different / max(len(neighbors), 1))
        weights = np.asarray(difficulty, dtype=float)
        weights = (
            weights / weights.sum()
            if weights.sum() > 0
            else np.full(len(class_indices), 1 / len(class_indices))
        )
        allocations = rng.multinomial(needed, weights)
        class_frame = all_values.iloc[class_indices].reset_index(drop=True)
        class_neighbors = NearestNeighbors(
            n_neighbors=min(6, len(class_frame))
        ).fit(class_frame)
        local_indices = class_neighbors.kneighbors(return_distance=False)
        synthetic = []
        for local_index, amount in enumerate(allocations):
            choices = local_indices[local_index][1:]
            for _ in range(int(amount)):
                neighbor_index = (
                    int(rng.choice(choices)) if len(choices) else local_index
                )
                gap = float(rng.random())
                synthetic.append(
                    class_frame.iloc[local_index].to_numpy(dtype=float)
                    + gap
                    * (
                        class_frame.iloc[neighbor_index].to_numpy(dtype=float)
                        - class_frame.iloc[local_index].to_numpy(dtype=float)
                    )
                )
        if synthetic:
            generated_frames.append(pd.DataFrame(synthetic, columns=X.columns))
            generated_targets.append(
                pd.Series([label] * len(synthetic), name=y.name)
            )
    return (
        pd.concat(generated_frames, ignore_index=True),
        pd.concat(generated_targets, ignore_index=True),
    )


@tool
def preprocessing_balancing(task, tool_input, prompt, data_path, llm, state=None):
    """Balance only the training split using the approved method."""
    pipeline_state = ensure_state(state, data_path, prompt)
    try:
        plan = tool_input.get("plan") or pipeline_state["preprocessing_plan"]
        X_train, X_test, y_train, y_test = load_split(tool_input)
        method = plan.get("balancing", {}).get("method", "none")
        random_state = int(tool_input.get("random_state", 42))
        before = {str(key): int(value) for key, value in y_train.value_counts().items()}
        warning = ""
        class_weights = {}

        if pipeline_state.get("problem_type") != "classification":
            method = "none"
            warning = "Balancing skipped because the target is not classification."
        elif method == "class_weight":
            maximum = max(before.values(), default=1)
            class_weights = {
                label: float(maximum / max(count, 1)) for label, count in before.items()
            }
        elif method in {"oversample", "undersample"}:
            combined = X_train.copy()
            combined["__target__"] = y_train.to_numpy()
            groups = [group for _, group in combined.groupby("__target__", dropna=False)]
            size = max(len(group) for group in groups) if method == "oversample" else min(
                len(group) for group in groups
            )
            sampled = [
                group.sample(
                    n=size,
                    replace=method == "oversample" and len(group) < size,
                    random_state=random_state,
                )
                for group in groups
            ]
            combined = pd.concat(sampled).sample(frac=1, random_state=random_state)
            y_train = combined.pop("__target__")
            X_train = combined
        elif method == "smote":
            try:
                X_train, y_train = _smote(X_train, y_train, random_state)
            except Exception as exc:
                warning = f"SMOTE skipped: {exc}"
                method = "none"
        elif method == "adasyn":
            try:
                X_train, y_train = _adasyn(X_train, y_train, random_state)
            except Exception as exc:
                warning = f"ADASYN skipped: {exc}"
                method = "none"

        after = {str(key): int(value) for key, value in y_train.value_counts().items()}
        paths = save_split(
            X_train.reset_index(drop=True),
            X_test.reset_index(drop=True),
            y_train.reset_index(drop=True),
            y_test.reset_index(drop=True),
            tool_input["output_folder"],
            "balanced",
        )
        metadata = {
            "method": method,
            "before_counts": before,
            "after_counts": after,
            "class_weights": class_weights,
            "warning": warning,
        }
        pipeline_state = merge_state(
            pipeline_state,
            {
                **paths,
                "balancing_output": metadata,
                "step": "balancing_complete",
                "status": "success",
            },
        )
        return {"status": "success", **paths, "metadata": metadata}, pipeline_state
    except Exception as exc:
        message = f"Balancing failed: {exc}"
        return {"status": "error", "error": message}, merge_state(
            pipeline_state, {"step": "balancing_failed", "status": "error", "error": message}
        )
