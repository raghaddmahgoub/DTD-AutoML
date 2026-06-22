"""
tools/target_suggester.py
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

Tool: Target Suggestion Agent
Responsibility:
    Heuristic fallback that suggests the most likely target column
    and task type when the LLM cannot determine them.

    suggest() priority order:
        1. Column name matches a well-known target keyword
        2. Last column in the DataFrame (standard ML convention)

    suggest_task_type() logic:
        - Non-numeric target         → classification
        - Numeric, ≤ 20 unique vals  → classification
        - Numeric, > 20 unique vals  → regression

Consumers:
    - agents/intent_detector.py  (Agent 0)
    - agents/eda_agent.py        (Agent 1)
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TargetSuggestionAgent:

    COMMON_TARGET_NAMES: list[str] = [
        "target", "label", "class", "output", "y",
        "result", "outcome", "churn", "fraud", "default",
        "survived", "diagnosis", "price", "salary", "revenue",
    ]

    def suggest(self, df: pd.DataFrame) -> Optional[str]:
        """Return the most likely target column name, or None."""
        cols_lower = {c.lower(): c for c in df.columns}

        # Priority 1 — well-known name
        for name in self.COMMON_TARGET_NAMES:
            if name in cols_lower:
                col = cols_lower[name]
                logger.info("[TargetSuggester] Matched known name: '%s'", col)
                return col

        # Priority 2 — last column
        if len(df.columns) > 0:
            col = df.columns[-1]
            logger.info("[TargetSuggester] Fallback to last column: '%s'", col)
            return col

        logger.warning("[TargetSuggester] Could not suggest a target column.")
        return None

    def suggest_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Infer task type from the target column distribution."""
        if target_column not in df.columns:
            return "unknown"

        series   = df[target_column].dropna()
        n_unique = series.nunique()

        if not pd.api.types.is_numeric_dtype(series):
            logger.info("[TargetSuggester] Non-numeric target → classification")
            return "classification"

        if n_unique <= 20:
            logger.info(
                "[TargetSuggester] %d unique values → classification", n_unique
            )
            return "classification"

        logger.info("[TargetSuggester] %d unique values → regression", n_unique)
        return "regression"