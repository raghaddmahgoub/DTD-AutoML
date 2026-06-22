"""
tools/preprocessing/
────────────────────
All @tool-decorated preprocessing steps for the deterministic pipeline.
Each step is a LangChain @tool.

Exports (one per pipeline step, in execution order):
    preprocessing_inspection    — Step 1: load & profile the dataset
    preprocessing_plan          — Step 2: LLM decides column-level actions
    preprocessing_split         — Step 3: train/test split
    preprocessing_missing_values — Step 4: impute missing data
    preprocessing_outliers      — Step 5: handle outliers
    preprocessing_encoding      — Step 6: encode categoricals
    preprocessing_scaling       — Step 7: scale numerics
    preprocessing_normalization — Step 8: normalize distributions
    preprocessing_balancing     — Step 9: class imbalance handling
    preprocessing_validation    — Step 10: validate processed splits
    preprocessing_execution     — Monolithic execution wrapper

Agent-level tool list (for documentation and future bind_tools support):
    PREPROCESSING_TOOLS         — ordered list of all 10 @tool functions
"""

from .inspection import preprocessing_inspection
from .plan import preprocessing_plan
from .split import preprocessing_split
from .missing_values import preprocessing_missing_values
from .outliers import preprocessing_outliers
from .encoding import preprocessing_encoding
from .scaling import preprocessing_scaling
from .normalization import preprocessing_normalization
from .balancing import preprocessing_balancing
from .validation import preprocessing_validation
from .execution import preprocessing_execution

# Ordered pipeline sequence — useful for introspection and future LLM bind_tools
PREPROCESSING_TOOLS = [
    preprocessing_inspection,
    preprocessing_plan,
    preprocessing_split,
    preprocessing_missing_values,
    preprocessing_outliers,
    preprocessing_encoding,
    preprocessing_scaling,
    preprocessing_normalization,
    preprocessing_balancing,
    preprocessing_validation,
]

__all__ = [
    "preprocessing_inspection",
    "preprocessing_plan",
    "preprocessing_split",
    "preprocessing_missing_values",
    "preprocessing_outliers",
    "preprocessing_encoding",
    "preprocessing_scaling",
    "preprocessing_normalization",
    "preprocessing_balancing",
    "preprocessing_validation",
    "preprocessing_execution",
    "PREPROCESSING_TOOLS",
]
