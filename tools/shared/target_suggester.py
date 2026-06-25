import pandas as pd
from typing import Optional


class TargetSuggestionAgent:
    COMMON_TARGET_NAMES = {
        "target", "label", "class", "output", "y",
        "result", "outcome", "churn", "fraud", "default",
        "survived", "diagnosis", "price", "salary", "revenue",
    }

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def suggest(self, df: Optional[pd.DataFrame] = None) -> Optional[str]:
        data = df if df is not None else self.df
        if data is None or data.empty:
            return None

        for col in data.columns:
            if col.lower() in self.COMMON_TARGET_NAMES:
                return col

        return data.columns[-1]

    def suggest_task_type(self, df_or_target, target_column: Optional[str] = None) -> str:
        if target_column is None:
            data = self.df
            target_column = df_or_target
        else:
            data = df_or_target

        if data is None or target_column not in data.columns:
            return "unknown"

        series = data[target_column]

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