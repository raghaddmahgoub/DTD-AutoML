import pandas as pd

def inspect_dataset(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """ Dataset-level inspection.
    Purely descriptive. No column-level logic. """
    return {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "has_target": target_col in df.columns if target_col else False,
        "target_name": target_col
    }
