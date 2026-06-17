"""LangGraph node: identify target column and problem type."""
from __future__ import annotations

import numpy as np
import dask.dataframe as dd

from tools.graph_state import TrainingGraphState
from src.utils.logger import Logger

logger = Logger()


def identify_target_node(state: TrainingGraphState) -> TrainingGraphState:
    try:
        target_col = state.get("target_column")
        if not target_col:
            target_col = state["data"].columns[-1]
            logger.info(f"Target not provided, using last column: {target_col}")

        problem_type = state.get("problem_type")
        if not problem_type:
            target_data = state["data"][target_col]
            if isinstance(state["data"], dd.DataFrame):
                n_unique = int(target_data.nunique().compute())
                is_numeric = np.issubdtype(target_data.dtype, np.number)
            else:
                n_unique = int(target_data.nunique())
                is_numeric = np.issubdtype(target_data.dtype, np.number)

            if is_numeric and n_unique > 20:
                problem_type = "regression"
            else:
                problem_type = "classification"

        state["target_column"] = target_col
        state["problem_type"] = problem_type
        state["step"] = "target_identified"
        logger.info(f"Target={target_col}, problem_type={problem_type}")
    except Exception as exc:
        state["error"] = f"Failed to identify target: {exc}"
        state["step"] = "error"
    return state
