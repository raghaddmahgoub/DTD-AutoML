import pandas as pd
from typing import Optional


class TargetSuggestionAgent:
    COMMON_TARGET_NAMES = {
        "target", "label", "class", "output", "y",
        "result", "outcome", "churn", "fraud", "default",
        "survived", "diagnosis", "price", "salary", "revenue",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def suggest(self) -> Optional[str]:
        if self.df.empty:
            return None

        for col in self.df.columns:
            if col.lower() in self.COMMON_TARGET_NAMES:
                return col

        return self.df.columns[-1]

    def suggest_task_type(self, target_column: str) -> str:
        if target_column not in self.df.columns:
            return "unknown"

        series = self.df[target_column]

        if not pd.api.types.is_numeric_dtype(series):
            return "classification"

        return (
            "classification"
            if series.nunique(dropna=True) <= 20
            else "regression"
        )

    def run(self):
        target = self.suggest()

        return {
            "target_column": target,
            "task_type": self.suggest_task_type(target) if target else "unknown",
        }