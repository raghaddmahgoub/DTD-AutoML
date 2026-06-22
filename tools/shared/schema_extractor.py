"""
tools/schema_extractor.py
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

Tool: Schema Extractor
Responsibility:
    Load a dataset file with pandas and return its schema
    (columns, dtypes, shape, live DataFrame) for the LLM prompt
    and the TargetSuggestionAgent fallback.

Consumers:
    - agents/intent_detector.py  (Agent 0)
    - agents/eda_agent.py        (Agent 1)
"""

import os
import logging
from typing import TypedDict

import pandas as pd

logger = logging.getLogger(__name__)

_SAMPLE_ROWS = 5_000   # cap for large CSV files — schema only needs a sample


class DatasetSchema(TypedDict):
    columns: list[str]
    dtypes:  dict[str, str]     # {"col": "dtype_str"}
    shape:   tuple[int, int]    # (total_rows, n_cols)
    df:      pd.DataFrame       # sampled DataFrame kept for TargetSuggestionAgent


def extract_schema(data_path: str) -> DatasetSchema:
    """
    Load dataset and extract schema.

    Supported: .csv  .xls  .xlsx  .parquet  .json

    For CSV: caps load at _SAMPLE_ROWS rows but reports the TRUE
    total row count in shape[0] via a fast line-count pass.

    Raises:
        FileNotFoundError: path does not exist.
        ValueError:        unsupported file extension.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    ext = os.path.splitext(data_path)[-1].lower()
    logger.info("[SchemaExtractor] Loading '%s' (ext=%s)", data_path, ext)

    if ext == ".csv":
        with open(data_path, "r", encoding="utf-8", errors="ignore") as fh:
            total_rows = sum(1 for _ in fh) - 1      # subtract header line
        df = pd.read_csv(data_path, nrows=_SAMPLE_ROWS)

    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(data_path)
        total_rows = len(df)

    elif ext == ".parquet":
        df = pd.read_parquet(data_path)
        total_rows = len(df)

    elif ext == ".json":
        df = pd.read_json(data_path)
        total_rows = len(df)

    else:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: .csv .xls .xlsx .parquet .json"
        )

    schema: DatasetSchema = {
        "columns": df.columns.tolist(),
        "dtypes":  {col: str(dt) for col, dt in df.dtypes.items()},
        "shape":   (total_rows, len(df.columns)),
        "df":      df,
    }
    logger.info(
        "[SchemaExtractor] Done — %d cols, shape=(%d,%d)",
        len(schema["columns"]), schema["shape"][0], schema["shape"][1],
    )
    return schema