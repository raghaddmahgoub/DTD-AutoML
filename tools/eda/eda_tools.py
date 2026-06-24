"""
Tool: EDA Computation Engine
Responsibility:
    All heavy pandas / scipy / numpy EDA computations live here.
    The EDA agent calls these functions; they return plain dicts
    that get written into PipelineState and serialised to JSON.

    No LLM calls here — pure deterministic computation only.
    No plotting here — plots live in tools/eda_plots.py.

Consumers:
    - agents/dynamic/eda_agent/eda_agent.py
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_selection import f_classif

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# File loader
# ─────────────────────────────────────────────

def load_dataframe(data_path: str) -> pd.DataFrame:
    """Load CSV / Excel / Parquet / JSON into a DataFrame."""
    ext = Path(data_path).suffix.lower()
    loaders = {
        ".csv":     lambda p: pd.read_csv(p),
        ".xls":     lambda p: pd.read_excel(p),
        ".xlsx":    lambda p: pd.read_excel(p),
        ".parquet": lambda p: pd.read_parquet(p),
        ".json":    lambda p: pd.read_json(p),
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported format '{ext}'")
    df = loaders[ext](data_path)
    logger.info("[EDATools] Loaded %s — shape=%s", data_path, df.shape)
    return df


# ─────────────────────────────────────────────
# 1. Dataset summary
# ─────────────────────────────────────────────

def compute_dataset_summary(df: pd.DataFrame, target_column: Optional[str]) -> Dict:
    return {
        "n_rows":           int(df.shape[0]),
        "n_columns":        int(df.shape[1]),
        "memory_usage_mb":  round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "duplicate_rows":   int(df.duplicated().sum()),
        "target_column":    target_column if (target_column and target_column in df.columns) else None,
        "column_types": {
            "numerical":   df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime":    df.select_dtypes(include=["datetime"]).columns.tolist(),
            "boolean":     df.select_dtypes(include=["bool"]).columns.tolist(),
        },
    }


# ─────────────────────────────────────────────
# 2. Per-column profiles
# ─────────────────────────────────────────────

def _infer_col_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):     return "boolean"
    if pd.api.types.is_numeric_dtype(series):  return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series): return "datetime"
    return "categorical"


def compute_column_profiles(df: pd.DataFrame, top_k: int = 5) -> Dict:
    profiles = {}
    n_rows = len(df)

    for col in df.columns:
        series   = df[col]
        col_type = _infer_col_type(series)

        profile: Dict[str, Any] = {
            "data_type":       col_type,
            "dtype":           str(series.dtype),
            "missing_count":   int(series.isna().sum()),
            "missing_ratio":   round(float(series.isna().mean()), 4),
            "unique_count":    int(series.nunique(dropna=True)),
            "is_unique_per_row": int(series.nunique(dropna=True)) == n_rows,
        }

        if col_type == "numeric":
            clean = series.dropna()
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr    = q3 - q1
            n_out  = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())

            is_normal = None
            if len(clean) >= 3 and clean.std() > 0:
                sample = clean.sample(min(len(clean), 5000), random_state=42)
                _, p   = scipy_stats.shapiro(sample)
                is_normal = bool(p > 0.05)

            profile.update({
                "mean":              round(float(clean.mean()), 4),
                "std":               round(float(clean.std()), 4),
                "min":               float(clean.min()),
                "max":               float(clean.max()),
                "median":            float(clean.median()),
                "q1":                float(q1),
                "q3":                float(q3),
                "skewness":          round(float(clean.skew()), 4),
                "kurtosis":          round(float(clean.kurtosis()), 4),
                "outlier_count_iqr": n_out,
                "outlier_ratio_iqr": round(n_out / max(len(clean), 1), 4),
                "is_normal":         is_normal,
            })

        elif col_type == "categorical":
            vc = series.value_counts(dropna=True)
            profile.update({
                "top_values":         vc.head(top_k).to_dict(),
                "is_high_cardinality": profile["unique_count"] > 0.5 * n_rows,
            })

        elif col_type == "datetime":
            clean = series.dropna()
            profile.update({
                "min_date": str(clean.min()),
                "max_date": str(clean.max()),
            })

        profiles[col] = profile

    return profiles


# ─────────────────────────────────────────────
# 3. Target analysis
# ─────────────────────────────────────────────

def compute_target_analysis(df: pd.DataFrame, target_column: Optional[str]) -> Optional[Dict]:
    if not target_column or target_column not in df.columns:
        return None

    series       = df[target_column].dropna()
    numeric_test = pd.to_numeric(series, errors="coerce")
    is_numeric   = numeric_test.notna().mean() > 0.8

    if is_numeric:
        series = numeric_test.dropna()

    analysis: Dict[str, Any] = {
        "column":    target_column,
        "dtype":     str(series.dtype),
    }

    if is_numeric and series.nunique() > 20:
        q1, q3 = series.quantile(0.25), series.quantique(0.75) if False else (series.quantile(0.25), series.quantile(0.75))
        iqr    = q3 - q1
        skew   = float(series.skew())
        analysis.update({
            "task_type":        "regression",
            "mean":             round(float(series.mean()), 4),
            "std":              round(float(series.std()), 4),
            "min":              float(series.min()),
            "max":              float(series.max()),
            "skewness":         round(skew, 4),
            "outlier_ratio_iqr":round(float(((series < q1-1.5*iqr)|(series > q3+1.5*iqr)).mean()), 4),
        })
        return analysis

    # Classification
    vc    = series.value_counts()
    probs = vc / vc.sum()
    imb   = round(float(vc.max() / vc.min()), 2) if vc.min() > 0 else None

    analysis.update({
        "task_type":            "classification",
        "n_classes":            len(vc),
        "is_binary":            len(vc) == 2,
        "class_distribution":   probs.round(4).to_dict(),
        "imbalance_ratio":      imb,
        "imbalance_severity":   (
            "none"     if imb is None or imb < 2 else
            "moderate" if imb < 5 else "severe"
        ),
        "minority_class_ratio": round(float(probs.min()), 4),
        "rare_class_risk":      bool(probs.min() < 0.05),
    })
    return analysis


# ─────────────────────────────────────────────
# 4. Data quality
# ─────────────────────────────────────────────

def compute_data_quality(df: pd.DataFrame) -> Dict:
    na_df    = df.isna()
    dup_mask = df.duplicated()
    n_rows   = len(df)

    missing_cols = {
        col: {
            "missing_count": int(na_df[col].sum()),
            "missing_ratio": round(float(na_df[col].mean()), 4),
        }
        for col in df.columns if na_df[col].any()
    }

    constant, near_constant, unique_per_row = [], {}, []
    for col in df.columns:
        nuniq = df[col].nunique(dropna=True)
        if nuniq <= 1:
            constant.append(col)
        elif nuniq == n_rows:
            unique_per_row.append(col)
        else:
            top_freq = df[col].value_counts(dropna=True).iloc[0] / n_rows
            if top_freq > 0.95:
                near_constant[col] = round(float(top_freq), 4)

    return {
        "missing_values": {
            "total_missing_cells":  int(na_df.sum().sum()),
            "columns_with_missing": missing_cols,
            "n_columns_with_missing": len(missing_cols),
        },
        "duplicates": {
            "duplicate_row_count": int(dup_mask.sum()),
            "duplicate_ratio":     round(float(dup_mask.mean()), 4),
        },
        "low_variance_columns": {
            "constant_columns":     constant,
            "near_constant_columns": near_constant,
        },
        "unique_per_row_columns": unique_per_row,
    }


# ─────────────────────────────────────────────
# 5. Relationship insights
# ─────────────────────────────────────────────

def compute_relationships(df: pd.DataFrame, target_column: Optional[str]) -> Dict:
    CORR_THRESHOLD = 0.5
    insights: Dict[str, Any] = {}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    numeric_cols = [c for c in numeric_cols if df[c].std() > 0]

    # Numeric ↔ Numeric
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) >= CORR_THRESHOLD:
                    pairs.append({
                        "feature_1":   numeric_cols[i],
                        "feature_2":   numeric_cols[j],
                        "correlation": round(float(val), 3),
                    })
        insights["numeric_correlations"] = {"threshold": CORR_THRESHOLD, "strong_pairs": pairs}
    else:
        insights["numeric_correlations"] = None

    # Feature ↔ Target
    if not target_column or target_column not in df.columns:
        insights["target_relationships"] = None
        return insights

    target_series = df[target_column]

    if pd.api.types.is_numeric_dtype(target_series):
        target_corr = (
            df[numeric_cols].corrwith(target_series)
            .dropna().round(3).to_dict()
        )
        insights["target_relationships"] = {
            "target_type":          "numeric",
            "feature_correlations": target_corr,
        }
    else:
        group_means = {
            col: df.groupby(target_column)[col].mean().round(3).to_dict()
            for col in numeric_cols
        }
        insights["target_relationships"] = {
            "target_type": "categorical",
            "group_means": group_means,
        }

    return insights


# ─────────────────────────────────────────────
# 6. EDA warnings
# ─────────────────────────────────────────────

def compute_warnings(
    dataset_summary: Dict,
    column_profiles: Dict,
    target_analysis: Optional[Dict],
) -> List[Dict]:
    warnings = []
    n_rows = dataset_summary["n_rows"]
    n_cols = dataset_summary["n_columns"]

    if n_rows < 100:
        warnings.append({"type": "small_dataset",
                         "message": "Fewer than 100 rows — generalisation risk."})
    if n_rows < n_cols:
        warnings.append({"type": "wide_dataset",
                         "message": "More features than rows — overfitting risk."})

    high_missing = [c for c, s in column_profiles.items() if s["missing_ratio"] > 0.5]
    if high_missing:
        warnings.append({"type": "high_missingness", "columns": high_missing,
                         "message": "Columns with >50% missing values."})

    constant = [c for c, s in column_profiles.items() if s["unique_count"] <= 1]
    if constant:
        warnings.append({"type": "constant_columns", "columns": constant,
                         "message": "Constant columns carry no information."})

    high_card = [c for c, s in column_profiles.items()
                 if s["data_type"] == "categorical" and s.get("is_high_cardinality")]
    if high_card:
        warnings.append({"type": "high_cardinality_categoricals", "columns": high_card,
                         "message": "High-cardinality categorical columns."})

    outlier_heavy = [c for c, s in column_profiles.items()
                     if s["data_type"] == "numeric" and s.get("outlier_ratio_iqr", 0) > 0.05]
    if outlier_heavy:
        warnings.append({"type": "high_outlier_ratio", "columns": outlier_heavy,
                         "message": "Numeric columns with >5% outliers (IQR)."})

    if target_analysis and target_analysis.get("task_type") == "classification":
        imb = target_analysis.get("imbalance_ratio")
        if imb and imb >= 3:
            warnings.append({"type": "class_imbalance", "imbalance_ratio": imb,
                              "message": f"Class imbalance ratio {imb} — consider SMOTE."})

    return warnings


# ─────────────────────────────────────────────
# 7. Signal analysis (feature ↔ target)
# ─────────────────────────────────────────────

def compute_signal_analysis(df: pd.DataFrame, target_column: Optional[str], task_type: str) -> Dict:
    if not target_column or target_column not in df.columns:
        return {}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if task_type == "classification":
        f_scores = {}
        for col in numeric_cols:
            tmp = pd.DataFrame({
                "feat": pd.to_numeric(df[col], errors="coerce"),
                "targ": df[target_column],
            }).dropna()
            if len(tmp) < 50:
                continue
            try:
                score, _ = f_classif(tmp[["feat"]], tmp["targ"])
                f_scores[col] = round(float(score[0]), 4)
            except Exception:
                continue
        return {"univariate_class_signal": f_scores}

    elif task_type == "regression":
        target_num = pd.to_numeric(df[target_column], errors="coerce")
        df_num     = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        pearson    = df_num.corrwith(target_num).dropna()
        return {
            "linear_signal_strength":  pearson.abs().round(3).to_dict(),
            "non_linear_candidates":   pearson[pearson.abs() < 0.3].index.tolist(),
        }

    return {}


# ─────────────────────────────────────────────
# 8. Preprocessing context (for Preprocessing Agent)
# ─────────────────────────────────────────────

def build_preprocessing_context(
    df: pd.DataFrame,
    column_profiles: Dict,
    target_column: Optional[str],
    sample_size: int = 5,
) -> List[Dict]:
    """Flat per-column list consumed by the Preprocessing Agent."""
    context = []
    for col, stats in column_profiles.items():
        seen, entry_samples = [], []
        for val in df[col]:
            if pd.isna(val):
                continue
            native = val.item() if hasattr(val, "item") else val
            if native not in seen:
                seen.append(native)
                entry_samples.append(native)
                if len(seen) == sample_size:
                    break

        entry: Dict[str, Any] = {
            "column":        col,
            "dtype":         stats["dtype"],
            "missing_pct":   round(stats["missing_ratio"] * 100, 2),
            "n_unique":      stats["unique_count"],
            "sample_values": entry_samples,
            "is_target":     (col == target_column),
        }
        if stats["data_type"] == "numeric":
            entry.update({
                "mean": stats.get("mean"),
                "std":  stats.get("std"),
                "skew": stats.get("skewness"),
            })
        context.append(entry)

    return context


# ─────────────────────────────────────────────
# 9. Frontend-ready visualization data
# ─────────────────────────────────────────────

def build_visualization_data(
    df: pd.DataFrame,
    column_profiles: Dict,
    target_column: Optional[str],
) -> Dict:
    """Pure-data arrays for the frontend charts — no matplotlib."""

    # Missing values bar
    missing_data = [
        {
            "column": col,
            "count":  info["missing_count"],
            "ratio":  info["missing_ratio"],
        }
        for col, info in
        {c: s for c, s in column_profiles.items() if s["missing_count"] > 0}.items()
    ]

    # Numeric distributions (histogram bins)
    numeric_dists = []
    for col, stats in column_profiles.items():
        if stats["data_type"] != "numeric" or col == target_column:
            continue
        series  = df[col].dropna()
        counts, bins = np.histogram(series, bins=20)
        numeric_dists.append({
            "column":    col,
            "histogram": {"counts": counts.tolist(), "bins": bins.tolist()},
            "raw_sample": series.sample(min(len(series), 50), random_state=42).tolist(),
        })

    # Categorical distributions
    cat_dists = []
    for col, stats in column_profiles.items():
        if stats["data_type"] != "categorical" or col == target_column:
            continue
        cat_dists.append({
            "column": col,
            "top_values": [
                {"label": str(k), "count": v}
                for k, v in stats.get("top_values", {}).items()
            ],
        })

    # Correlation matrix
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    corr_matrix  = None
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        corr_matrix = {"columns": numeric_cols, "values": corr.values.tolist()}

    return {
        "missing_values_chart":      missing_data,
        "numeric_distributions":     numeric_dists,
        "categorical_distributions": cat_dists,
        "correlation_matrix":        corr_matrix,
    }


# ─────────────────────────────────────────────
# 10. JSON persistence
# ─────────────────────────────────────────────

def save_eda_report(report: Dict, output_dir: str, filename: str = "eda_report.json") -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("[EDATools] Report saved -> %s", file_path)
    return str(file_path)