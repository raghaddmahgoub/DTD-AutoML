"""
Tool: EDA Plot Generator
Responsibility:
    Generate and save matplotlib / seaborn plots for EDA output.
    Returns a list of plot metadata dicts (path, title, type).
    All plotting logic is isolated here so the EDA agent stays clean.

    Supports both:
        - Deterministic standard plots (always generated)
        - LLM-requested plots (from the visualizations list in LLM response)

Consumers:
    - agents/dynamic/eda_agent/eda_agent.py
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Internal save helper
# ─────────────────────────────────────────────

def _save(fig, path: str, title: str) -> None:
    plt.title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("[EDAPlots] Saved: %s", path)


# ─────────────────────────────────────────────
# Standard deterministic plots
# ─────────────────────────────────────────────

def plot_missing_values(df: pd.DataFrame, output_dir: str) -> Optional[Dict]:
    """Bar chart of missing value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax, color="#2E75B6")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")

    path = os.path.join(output_dir, "missing_values.png")
    _save(fig, path, "Missing Values per Column")

    return {
        "title":      "Missing Values per Column",
        "plot_type":  "missing_values",
        "local_path": path,
        "filename":   "missing_values.png",
    }


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> Optional[Dict]:
    """Pearson correlation heatmap for all numeric columns."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(min(14, numeric.shape[1] + 2), min(12, numeric.shape[1] + 2)))
    sns.heatmap(
        numeric.corr(),
        annot=numeric.shape[1] <= 15,   # only annotate when readable
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )

    path = os.path.join(output_dir, "correlation_heatmap.png")
    _save(fig, path, "Feature Correlation Heatmap")

    return {
        "title":      "Feature Correlation Heatmap",
        "plot_type":  "heatmap",
        "local_path": path,
        "filename":   "correlation_heatmap.png",
    }


def plot_target_distribution(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    output_dir: str,
) -> Optional[Dict]:
    """Distribution plot for the target column."""
    if target_column not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    series = df[target_column].dropna()

    if task_type == "regression":
        sns.histplot(series, kde=True, ax=ax, color="#2E75B6")
    else:
        sns.countplot(x=series, ax=ax, hue=series, legend=False, palette="Blues_d")
        ax.tick_params(axis="x", rotation=30)

    path = os.path.join(output_dir, "target_distribution.png")
    _save(fig, path, f"Target Distribution — {target_column}")

    return {
        "title":      f"Target Distribution — {target_column}",
        "plot_type":  "target_distribution",
        "local_path": path,
        "filename":   "target_distribution.png",
    }


def plot_numeric_distributions(
    df: pd.DataFrame,
    column_profiles: Dict,
    target_column: Optional[str],
    output_dir: str,
    max_cols: int = 12,
) -> Optional[Dict]:
    """Grid of histograms for numeric feature columns."""
    numeric_cols = [
        c for c, s in column_profiles.items()
        if s["data_type"] == "numeric" and c != target_column
    ][:max_cols]

    if not numeric_cols:
        return None

    n    = len(numeric_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = [axes] if n == 1 else axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), ax=axes[i], kde=True, color="#2E75B6")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    path = os.path.join(output_dir, "numeric_distributions.png")
    _save(fig, path, "Numeric Feature Distributions")

    return {
        "title":      "Numeric Feature Distributions",
        "plot_type":  "numeric_distributions",
        "local_path": path,
        "filename":   "numeric_distributions.png",
        "columns":    numeric_cols,
    }


def plot_categorical_distributions(
    df: pd.DataFrame,
    column_profiles: Dict,
    target_column: Optional[str],
    output_dir: str,
    max_cols: int = 8,
) -> Optional[Dict]:
    """Grid of countplots for categorical feature columns."""
    cat_cols = [
        c for c, s in column_profiles.items()
        if s["data_type"] == "categorical"
        and c != target_column
        and not s.get("is_high_cardinality")
    ][:max_cols]

    if not cat_cols:
        return None

    n    = len(cat_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = [axes] if n == 1 else axes.flatten()

    for i, col in enumerate(cat_cols):
        order = df[col].value_counts().head(10).index
        sns.countplot(x=df[col], order=order, ax=axes[i], hue=df[col], legend=False, palette="Blues_d")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    path = os.path.join(output_dir, "categorical_distributions.png")
    _save(fig, path, "Categorical Feature Distributions")

    return {
        "title":      "Categorical Feature Distributions",
        "plot_type":  "categorical_distributions",
        "local_path": path,
        "filename":   "categorical_distributions.png",
        "columns":    cat_cols,
    }


# ─────────────────────────────────────────────
# LLM-requested dynamic plots
# ─────────────────────────────────────────────

def generate_llm_requested_plot(
    df: pd.DataFrame,
    viz: Dict,
    output_dir: str,
    idx: int,
) -> Optional[Dict]:
    """
    Render a single plot requested by the LLM in its analysis response.

    viz dict format:
        {
            "plot_type": "histogram|boxplot|scatterplot|heatmap|countplot|missing_values",
            "columns":   ["col_a", "col_b"],
            "title":     "Plot Title",
            "reason":    "Why this plot was requested"
        }
    """
    plot_type = viz.get("plot_type", "")
    columns   = [c for c in viz.get("columns", []) if c in df.columns]
    title     = viz.get("title", f"Plot {idx}")
    reason    = viz.get("reason", "")
    filename  = f"llm_plot_{idx:02d}.png"
    path      = os.path.join(output_dir, filename)

    try:
        fig, ax = plt.subplots(figsize=(9, 5))

        if plot_type == "missing_values":
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if missing.empty:
                plt.close(fig)
                return None
            missing.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#2E75B6")

        elif plot_type == "heatmap":
            numeric = df.select_dtypes(include=["number"])
            if numeric.shape[1] < 2:
                plt.close(fig)
                return None
            sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)

        elif plot_type == "histogram":
            if columns:
                sns.histplot(df[columns[0]].dropna(), kde=True, ax=ax, color="#2E75B6")

        elif plot_type == "boxplot":
            if len(columns) == 1:
                sns.boxplot(x=df[columns[0]], ax=ax)
            elif len(columns) >= 2:
                sns.boxplot(x=df[columns[0]], y=df[columns[1]], ax=ax)

        elif plot_type == "countplot":
            if columns:
                order = df[columns[0]].value_counts().head(15).index
                hue   = df[columns[1]] if len(columns) >= 2 else df[columns[0]]
                sns.countplot(x=df[columns[0]], order=order, hue=hue, ax=ax, palette="Blues_d", legend=False if len(columns) < 2 else True)
                ax.tick_params(axis="x", rotation=30)

        elif plot_type == "scatterplot":
            if len(columns) >= 2:
                hue = df[columns[2]] if len(columns) >= 3 else None
                sns.scatterplot(x=df[columns[0]], y=df[columns[1]], hue=hue, ax=ax, alpha=0.6)

        else:
            plt.close(fig)
            return None

        _save(fig, path, title)

        return {
            "title":      title,
            "reason":     reason,
            "plot_type":  plot_type,
            "columns":    columns,
            "local_path": path,
            "filename":   filename,
        }

    except Exception as exc:
        logger.warning("[EDAPlots] Failed to render '%s': %s", title, exc)
        plt.close(fig)
        return None


# ─────────────────────────────────────────────
# Main entry point — generate all plots
# ─────────────────────────────────────────────

def generate_all_plots(
    df: pd.DataFrame,
    column_profiles: Dict,
    target_column: Optional[str],
    task_type: str,
    output_dir: str,
    llm_requested_visualizations: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Generate ALL plots — standard deterministic ones + LLM-requested ones.

    Args:
        df:                            The dataset DataFrame.
        column_profiles:               Output of compute_column_profiles().
        target_column:                 Target column name or None.
        task_type:                     "classification" | "regression" | "unknown".
        output_dir:                    Directory to save PNG files.
        llm_requested_visualizations:  List of viz dicts from LLM response.

    Returns:
        List of plot metadata dicts, each with:
            title, plot_type, local_path, filename, [columns], [reason]
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plots: List[Dict] = []

    # ── Standard plots (always generated) ────────────────────────────────────
    for fn in [
        lambda: plot_missing_values(df, output_dir),
        lambda: plot_correlation_heatmap(df, output_dir),
        lambda: plot_numeric_distributions(df, column_profiles, target_column, output_dir),
        lambda: plot_categorical_distributions(df, column_profiles, target_column, output_dir),
    ]:
        result = fn()
        if result:
            plots.append(result)

    if target_column and target_column in df.columns:
        result = plot_target_distribution(df, target_column, task_type, output_dir)
        if result:
            plots.append(result)

    # ── LLM-requested plots ───────────────────────────────────────────────────
    if llm_requested_visualizations:
        for idx, viz in enumerate(llm_requested_visualizations):
            result = generate_llm_requested_plot(df, viz, output_dir, idx)
            if result:
                plots.append(result)

    logger.info("[EDAPlots] Generated %d plots in %s", len(plots), output_dir)
    return plots