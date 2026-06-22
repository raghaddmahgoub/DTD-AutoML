"""
tools/eda/
──────────
EDA (Exploratory Data Analysis) tools.
Pure computation functions and plot generators — no @tool decorator.

Exports:
    load_dataframe              — load CSV / Parquet into pandas/dask
    compute_dataset_summary     — overall dataset statistics
    compute_column_profiles     — per-column type/cardinality/nulls
    compute_target_analysis     — target distribution, class balance
    compute_data_quality        — duplicates, missing, anomalies
    compute_relationships       — correlations, mutual info
    compute_warnings            — data quality warnings list
    compute_signal_analysis     — feature importance signals
    build_preprocessing_context — structured context dict for preprocessing
    save_eda_report             — persist report JSON to disk
    generate_all_plots          — batch-generate all EDA plots
    generate_llm_requested_plot — generate a single LLM-requested plot
"""

from .eda_tools import (
    load_dataframe,
    compute_dataset_summary,
    compute_column_profiles,
    compute_target_analysis,
    compute_data_quality,
    compute_relationships,
    compute_warnings,
    compute_signal_analysis,
    build_preprocessing_context,
    save_eda_report,
)
from .eda_plots import (
    generate_all_plots,
    generate_llm_requested_plot,
)
from .data_understanding import data_understanding

__all__ = [
    "load_dataframe",
    "compute_dataset_summary",
    "compute_column_profiles",
    "compute_target_analysis",
    "compute_data_quality",
    "compute_relationships",
    "compute_warnings",
    "compute_signal_analysis",
    "build_preprocessing_context",
    "save_eda_report",
    "generate_all_plots",
    "generate_llm_requested_plot",
    "data_understanding",
]
