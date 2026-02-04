import json
import base64
import io
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
from pathlib import Path
from scipy import stats as scipy_stats
import gc
from sklearn.feature_selection import f_classif
import numpy as np

matplotlib.use("Agg")  # no display; we render to bytes

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette("husl", 12)
sns.set_theme(style="whitegrid", font_scale=0.9)



def _fig_to_b64(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64 PNG string with memory cleanup."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight") # Reduced DPI from 150 to 100
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    
    # Aggressive Cleanup
    plt.close(fig) # Close the window
    fig.clf() # Clear the figure content
    buf.close() # Close the buffer
    gc.collect() # Force Python to reclaim memory
    return img_str


class EDAAgent:
    """
    Autonomous Exploratory Data Analysis (EDA) Agent.

    Performs descriptive analysis only — no transformations or decisions.

    Dual-output routing based on pipeline stage:
        run_type="raw"   →  generate_preprocessing_context()  (for PreprocessingAgent)
        run_type="clean" →  generate_automl_context()         (for AutoMLAgent)

    Additionally produces a self-contained HTML report with embedded plots
    that the user can open directly in a browser.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.
    target_column : str, optional
        Name of the target/label column.
    df_name : str
        Identifier used in persisted file names.
    top_k : int
        Number of top categorical values kept in column profiles.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        df_name: str = "dataset",
        top_k: int = 5,
    ):
        self.df = df
        self.target = target_column
        self.df_name = df_name
        self.top_k = top_k
        self.report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_column_type(series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        return "categorical"

    # ------------------------------------------------------------------
    # 1. Dataset-level summary
    # ------------------------------------------------------------------

    def _dataset_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "n_rows": int(self.df.shape[0]),
            "n_columns": int(self.df.shape[1]),
            "column_types": {
                "numerical": self.df.select_dtypes(include=["number"]).columns.tolist(),
                "categorical": self.df.select_dtypes(include=["object", "category"]).columns.tolist(),
                "datetime": self.df.select_dtypes(include=["datetime"]).columns.tolist(),
                "boolean": self.df.select_dtypes(include=["bool"]).columns.tolist(),
            },
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "target_column": self.target if (self.target and self.target in self.df.columns) else None,
            "target_dtype": str(self.df[self.target].dtype) if (self.target and self.target in self.df.columns) else None,
        }
        return summary
    
    def _feature_scale_analysis(self) -> Dict[str, Any]:

        numeric = self.df.select_dtypes(include=["number"])
    
        if self.target in numeric.columns:
            numeric = numeric.drop(columns=[self.target])

        stds = numeric.std()

        return {
            "wide_scale_features": stds[stds > stds.median() * 10].index.tolist(),
            "heavy_tailed_features": [
                col for col in numeric.columns
                if abs(numeric[col].kurtosis()) > 3
            ],
            "approximately_standard_scale_features": [
                col for col in numeric.columns
                if 0.5 < numeric[col].std() < 5
            ]
        }

    # ------------------------------------------------------------------
    # 2. Per-column profiling
    # ------------------------------------------------------------------

    def _column_profiles(self) -> Dict[str, Any]:
        profiles: Dict[str, Any] = {}
        n_rows = len(self.df)

        for col in self.df.columns:
            series = self.df[col]
            data_type = self._infer_column_type(series)

            profile: Dict[str, Any] = {
                "data_type": data_type,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_ratio": round(float(series.isna().mean()), 4),
                "unique_count": int(series.nunique(dropna=True)),
                "is_unique_per_row": int(series.nunique(dropna=True)) == n_rows,
            }

            if data_type == "numeric":
                clean = series.dropna()
                q1 = clean.quantile(0.25)
                q3 = clean.quantile(0.75)
                iqr = q3 - q1
                outlier_count = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())

                is_normal: Optional[bool] = None
                if len(clean) >= 3 and clean.std() > 0:
                    sample = clean.sample(min(len(clean), 5000), random_state=42)
                    _, p_value = scipy_stats.shapiro(sample)
                    is_normal = bool(p_value > 0.05)
                elif len(clean) >= 3:
                    is_normal = True  # constant → trivially normal

                profile.update({
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()), 4),
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                    "median": float(clean.median()),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": round(float(iqr), 4),
                    "skewness": round(float(clean.skew()), 4),
                    "kurtosis": round(float(clean.kurtosis()), 4),
                    "zero_count": int((clean == 0).sum()),
                    "outlier_count_iqr": outlier_count,
                    "outlier_ratio_iqr": round(outlier_count / max(len(clean), 1), 4),
                    "is_normal": is_normal,
                })

            elif data_type == "categorical":
                value_counts = series.value_counts(dropna=True)
                profile.update({
                    "top_values": value_counts.head(self.top_k).to_dict(),
                    "is_high_cardinality": profile["unique_count"] > 0.5 * n_rows,
                })

            elif data_type == "datetime":
                clean = series.dropna()
                profile.update({
                    "min_date": str(clean.min()),
                    "max_date": str(clean.max()),
                })

            profiles[col] = profile

        return profiles

    # ------------------------------------------------------------------
    # 3. Target analysis
    # ------------------------------------------------------------------

    def _target_analysis(self) -> Optional[Dict[str, Any]]:
        if self.target is None or self.target not in self.df.columns:
            return None

        series = self.df[self.target].dropna()
        dtype = str(series.dtype)

        analysis: Dict[str, Any] = {
            "column": self.target,
            "dtype": dtype,
        }

        is_numeric = pd.api.types.is_numeric_dtype(series)
        unique_values = series.unique()

        # ─────────────────────────────────────────────
        # REGRESSION TARGET
        # ─────────────────────────────────────────────
        if is_numeric and len(unique_values) > 20:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr))

            skew = float(series.skew())
            kurt = float(series.kurtosis())

            analysis.update({
                "task_type": "regression",
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "variance": round(float(series.var()), 4),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": round(float(series.max() - series.min()), 4),
                "skewness": round(skew, 4),
                "skew_severity": (
                    "low" if abs(skew) < 0.5 else
                    "moderate" if abs(skew) < 1 else
                    "high"
                ),
                "kurtosis": round(kurt, 4),
                "outlier_ratio_iqr": round(float(outliers.mean()), 4),
                "heavy_tailed": bool(kurt > 3),
                "low_variance_target": bool(series.var() < 1e-3),
            })

            return analysis

        # ─────────────────────────────────────────────
        # CLASSIFICATION TARGET
        # ─────────────────────────────────────────────
        value_counts = series.value_counts()
        total = value_counts.sum()
        probs = value_counts / total

        entropy = -np.sum(probs * np.log2(probs))

        imbalance_ratio = (
            round(float(value_counts.max() / value_counts.min()), 2)
            if value_counts.min() > 0 else None
        )

        analysis.update({
            "task_type": "classification",
            "n_classes": len(value_counts),
            "is_binary": len(value_counts) == 2,
            "class_distribution": probs.round(4).to_dict(),
            "minority_class_ratio": round(float(probs.min()), 4),
            "majority_class_ratio": round(float(probs.max()), 4),
            "imbalance_ratio": imbalance_ratio,
            "imbalance_severity": (
                "none" if imbalance_ratio is None or imbalance_ratio < 2 else
                "moderate" if imbalance_ratio < 5 else
                "severe"
            ),
            "target_entropy": round(float(entropy), 4),
            "min_samples_per_class": int(value_counts.min()),
            "requires_stratification": True,
            "rare_class_risk": bool(probs.min() < 0.05),
        })

        return analysis

    # ------------------------------------------------------------------
    # 4. Data quality report
    # ------------------------------------------------------------------

    def _data_quality_report(self) -> Dict[str, Any]:
        n_rows = len(self.df)
        na_df = self.df.isna()
        dup_mask = self.df.duplicated()

        missing_by_column = {
            col: {
                "missing_count": int(na_df[col].sum()),
                "missing_ratio": round(float(na_df[col].mean()), 4),
            }
            for col in self.df.columns
            if na_df[col].any()
        }

        constant_columns: List[str] = []
        near_constant_columns: Dict[str, float] = {}
        unique_per_row_columns: List[str] = []

        for col in self.df.columns:
            nunique = self.df[col].nunique(dropna=True)
            if nunique <= 1:
                constant_columns.append(col)
            elif nunique == n_rows:
                unique_per_row_columns.append(col)
            else:
                top_freq = self.df[col].value_counts(dropna=True).iloc[0] / n_rows
                if top_freq > 0.95:
                    near_constant_columns[col] = round(float(top_freq), 4)

        mixed_type_columns = [
            col
            for col in self.df.select_dtypes(include=["object"]).columns
            if self.df[col].dropna().map(type).nunique() > 1
        ]

        return {
            "missing_values": {
                "total_missing_cells": int(na_df.sum().sum()),
                "columns_with_missing": missing_by_column,
                "n_columns_with_missing": len(missing_by_column),
            },
            "duplicates": {
                "duplicate_row_count": int(dup_mask.sum()),
                "duplicate_ratio": round(float(dup_mask.mean()), 4),
            },
            "low_variance_columns": {
                "constant_columns": constant_columns,
                "near_constant_columns": near_constant_columns,
            },
            "unique_per_row_columns": unique_per_row_columns,
            "type_issues": {
                "mixed_type_columns": mixed_type_columns,
            },
        }

    # ------------------------------------------------------------------
    # 5. Relationship insights
    # ------------------------------------------------------------------

    def _relationship_insights(self) -> Dict[str, Any]:
        insights: Dict[str, Any] = {}
        CORR_THRESHOLD = 0.5

        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)

        # Drop constant columns — correlation undefined when std == 0
        numeric_cols = [col for col in numeric_cols if self.df[col].std() > 0]

        # --- Numeric ↔ Numeric ---
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            strong_pairs: List[Dict[str, Any]] = []

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    val = corr_matrix.iloc[i, j]
                    if pd.notna(val) and abs(val) >= CORR_THRESHOLD:
                        strong_pairs.append({
                            "feature_1": numeric_cols[i],
                            "feature_2": numeric_cols[j],
                            "correlation": round(float(val), 3),
                        })

            insights["numeric_correlations"] = {
                "threshold": CORR_THRESHOLD,
                "strong_pairs": strong_pairs,
            }
        else:
            insights["numeric_correlations"] = None

        # --- Feature ↔ Target ---
        if not (self.target and self.target in self.df.columns):
            insights["target_relationships"] = None
            return insights

        target_series = self.df[self.target]

        if pd.api.types.is_numeric_dtype(target_series):
            target_corr = (
                self.df[numeric_cols]
                .corrwith(target_series)
                .dropna()
                .round(3)
                .to_dict()
            )
            insights["target_relationships"] = {
                "target_type": "numeric",
                "feature_correlations": target_corr,
            }
        else:
            group_means: Dict[str, Any] = {}
            for col in numeric_cols:
                group_means[col] = (
                    self.df.groupby(self.target)[col].mean().round(3).to_dict()
                )

            categorical_cols = [
                col for col in self.df.select_dtypes(include=["object", "category"]).columns
                if col != self.target
            ]
            cramers_v: Dict[str, float] = {}
            for col in categorical_cols:
                if self.df[col].nunique() > 50: 
                    cramers_v[col] = 0.0 # Or skip entirely
                else:
                    cramers_v[col] = round(self._cramers_v(self.df[col], target_series), 3)

            insights["target_relationships"] = {
                "target_type": "categorical",
                "group_means": group_means,
                "cramers_v": cramers_v,
            }

        return insights

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        contingency = pd.crosstab(x, y)
        chi2, _, _, _ = scipy_stats.chi2_contingency(contingency, correction=False)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0], contingency.shape[1]) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    # ------------------------------------------------------------------
    # 6. EDA warnings
    # ------------------------------------------------------------------

    def _generate_eda_warnings(
        self,
        dataset_summary: Dict[str, Any],
        column_profiles: Dict[str, Any],
        target_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        warnings: List[Dict[str, Any]] = []
        n_rows = dataset_summary["n_rows"]
        n_cols = dataset_summary["n_columns"]

        if n_rows < 100:
            warnings.append({"type": "small_dataset", "message": "Dataset contains fewer than 100 rows; model generalisation may be limited."})
        if n_rows < n_cols:
            warnings.append({"type": "wide_dataset", "message": "Number of features exceeds number of rows — overfitting risk is high."})

        high_missing = [col for col, s in column_profiles.items() if s["missing_ratio"] > 0.5]
        if high_missing:
            warnings.append({"type": "high_missingness", "columns": high_missing, "message": "Columns with > 50% missing values detected."})

        constant = [col for col, s in column_profiles.items() if s["unique_count"] <= 1]
        if constant:
            warnings.append({"type": "constant_columns", "columns": constant, "message": "Constant columns carry no information."})

        high_card = [col for col, s in column_profiles.items() if s["data_type"] == "categorical" and s.get("is_high_cardinality", False)]
        if high_card:
            warnings.append({"type": "high_cardinality_categoricals", "columns": high_card, "message": "High-cardinality categoricals detected."})

        id_cols = [col for col, s in column_profiles.items() if s.get("is_unique_per_row", False)]
        if id_cols:
            warnings.append({"type": "unique_per_row_columns", "columns": id_cols, "message": "Likely identifier columns (unique per row)."})

        outlier_heavy = [col for col, s in column_profiles.items() if s["data_type"] == "numeric" and s.get("outlier_ratio_iqr", 0) > 0.05]
        if outlier_heavy:
            warnings.append({"type": "high_outlier_ratio", "columns": outlier_heavy, "message": "Numeric columns with > 5 % outliers (IQR)."})

        non_normal = [col for col, s in column_profiles.items() if s["data_type"] == "numeric" and s.get("is_normal") is False]
        if non_normal:
            warnings.append({"type": "non_normal_columns", "columns": non_normal, "message": "Non-normal numeric columns (Shapiro-Wilk)."})

        if target_analysis and target_analysis.get("task_type") == "classification":
            imbalance_ratio = target_analysis.get("imbalance_ratio")
            if imbalance_ratio is not None and imbalance_ratio >= 3:
                warnings.append({"type": "class_imbalance", "imbalance_ratio": imbalance_ratio, "message": "Target shows class imbalance (majority/minority ≥ 3)."})

        return warnings

    # ------------------------------------------------------------------
    # 7. Run pipeline
    # ------------------------------------------------------------------

    def run(self, run_type: str = "raw") -> Dict[str, Any]:
        """
        Execute the full EDA pipeline.

        Parameters
        ----------
        run_type : {"raw", "clean"}
            "raw"   → after initial ingestion, before preprocessing.
            "clean" → after preprocessing, before model training.
        """
        self.report = {
            "run_type": run_type,
            "dataset_summary": self._dataset_summary(),
            "column_profiles": self._column_profiles(),
            "target_analysis": self._target_analysis(),
            "data_quality_report": self._data_quality_report(),
            "relationship_insights": self._relationship_insights(),
        }
        self.report["eda_warnings"] = self._generate_eda_warnings(
            dataset_summary=self.report["dataset_summary"],
            column_profiles=self.report["column_profiles"],
            target_analysis=self.report["target_analysis"],
        )
        return self.report

    # ==================================================================
    # OUTPUT A — Preprocessing context  (run_type == "raw")
    # ==================================================================

    def _collect_sample_values(self, col: str, n: int = 5) -> List[Any]:
        seen: List[Any] = []
        for val in self.df[col]:
            if pd.isna(val):
                continue
            native = val.item() if hasattr(val, "item") else val
            if native not in seen:
                seen.append(native)
                if len(seen) == n:
                    break
        return seen

    def generate_preprocessing_context(
        self,
        plan_dir: str = "Plan",
        output_dir: str = "Output",
        sample_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Flat per-column JSON consumed by the PreprocessingAgent.
        Called automatically when run_type == "raw".
        """
        if not self.report:
            raise ValueError("Run EDA before generating context.")

        columns = self.report["column_profiles"]
        context: List[Dict[str, Any]] = []

        for col, stats in columns.items():
            entry: Dict[str, Any] = {
                "column": col,
                "dtype": stats["dtype"],
                "missing_pct": round(stats["missing_ratio"] * 100, 2),
                "n_unique": stats["unique_count"],
                "sample_values": self._collect_sample_values(col, sample_size),
                "is_target": (col == self.target),
            }
            if stats["data_type"] == "numeric":
                entry["mean"] = stats.get("mean")
                entry["std"] = stats.get("std")
                entry["skew"] = stats.get("skewness")

            context.append(entry)

        self._persist_json(context, f"{self.df_name}_preprocessing_context.json", plan_dir, output_dir)
        return context

    # ==================================================================
    # OUTPUT B — AutoML context  (run_type == "clean")
    # ==================================================================

    def generate_automl_context(
        self,
        plan_dir: str = "Plan",
        output_dir: str = "Output",
    ) -> Dict[str, Any]:
        """
        Structured JSON consumed by the AutoMLAgent.
        Called automatically when run_type == "clean".

        Contains everything the AutoML agent needs to pick models,
        metrics, and encoding strategies without re-analysing the data.
        """
        if not self.report:
            raise ValueError("Run EDA before generating context.")
        
        target_analysis = self.report.get("target_analysis") or {}
        column_profiles = self.report["column_profiles"]
        relationships = self.report.get("relationship_insights", {})

        task_type = target_analysis.get("task_type", "unknown")

        signal_analysis = {}

        if task_type == "classification":
            signal_analysis = {
                "classification_feature_analysis": self._classification_signal_analysis()
            }

        elif task_type == "regression":
            signal_analysis = {
                "regression_feature_analysis": self._regression_signal_analysis()
            }

        # ── feature lists (exclude target) ──────────────────────────
        numeric_features: List[str] = []
        categorical_features: List[str] = []

        for col, stats in column_profiles.items():
            if col == self.target:
                continue
            if stats["data_type"] == "numeric":
                numeric_features.append(col)
            elif stats["data_type"] == "categorical":
                categorical_features.append(col)

        # ── multicollinearity flags ─────────────────────────────────
        multicollinear_pairs: List[Dict[str, Any]] = []
        num_corr = relationships.get("numeric_correlations")
        if num_corr:
            multicollinear_pairs = [
                p for p in num_corr.get("strong_pairs", [])
                if abs(p["correlation"]) >= 0.7
            ]


        # ── assemble ────────────────────────────────────────────────
        automl_context: Dict[str, Any] = {
            "task_type": task_type,
            "report": self.report,
            "total_feature_count": len(numeric_features) + len(categorical_features),
            "multicollinearity": {
                "threshold": 0.7,
                "pairs": multicollinear_pairs,
            },
            "feature_scale_analysis": self._feature_scale_analysis(),
            # "encoding_hints": self._encoding_hints(column_profiles),
            "signal_analysis": signal_analysis,
        }

        self._persist_json(automl_context, f"{self.df_name}_automl_context.json", plan_dir, output_dir)
        return automl_context

    # ── AutoML helpers ────────────────────────────────────────────────

    def _classification_signal_analysis(self) -> Dict[str, Any]:
        if self.target is None:
            return {}

        df = self.df
        y = df[self.target]

        numeric_features = df.select_dtypes(include=["number"]).columns
        numeric_features = [c for c in numeric_features if c != self.target]

        f_scores = {}

        for feature in numeric_features:
            valid = df[[feature, self.target]].dropna()

            # Skip weak samples
            if valid.shape[0] < 50:
                continue

            X_feat = valid[[feature]].values
            y_valid = valid[self.target].values

            try:
                score, _ = f_classif(X_feat, y_valid)
                f_scores[feature] = round(float(score[0]), 4)
            except Exception:
                continue

        return {
            "univariate_class_signal": f_scores
        }

    def _regression_signal_analysis(self) -> Dict[str, Any]:
        numeric_cols = self.df.select_dtypes(include=["number"]).columns
        numeric_cols = [c for c in numeric_cols if c != self.target]

        target = self.df[self.target]

        pearson = self.df[numeric_cols].corrwith(target).dropna()

        return {
            "linear_signal_strength": pearson.abs().round(3).to_dict(),
            "non_linear_candidates": pearson[pearson.abs() < 0.3].index.tolist()
        }

    # @staticmethod
    # def _encoding_hints(column_profiles: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Per-column encoding suggestions for the AutoML agent.
    #     """
    #     hints: Dict[str, Any] = {}
    #     for col, stats in column_profiles.items():
    #         if stats["data_type"] != "categorical":
    #             continue

    #         n_unique = stats["unique_count"]
    #         if n_unique == 2:
    #             hints[col] = {"strategy": "binary", "reason": "Only 2 unique values."}
    #         elif n_unique <= 15:
    #             hints[col] = {"strategy": "one_hot", "reason": f"{n_unique} categories — safe for one-hot."}
    #         else:
    #             hints[col] = {"strategy": "target_encoding", "reason": f"High cardinality ({n_unique}) — use target encoding to avoid dimensionality blow-up."}

    #     return hints

    # ==================================================================
    # OUTPUT C — User-facing HTML report  (always generated on export)
    # ==================================================================
    
    def generate_report(self, output_dir: str = "Output") -> str:
        """
        Build a self-contained HTML report with embedded plots.
        Returns the path to the saved file.
        """
        if not self.report:
            raise ValueError("Run EDA before generating the report.")

        sections: List[str] = []

        sections.append(self._html_header())
        sections.append(self._section_overview())
        sections.append(self._section_warnings())
        sections.append(self._section_missing_values())
        sections.append(self._section_numeric_distributions())
        sections.append(self._section_categorical_distributions())
        sections.append(self._section_correlation_heatmap())
        sections.append(self._section_target_analysis())
        sections.append(self._html_footer())

        html = "\n".join(sections)

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        report_path = path / f"{self.df_name}_eda_report.html"
        report_path.write_text(html, encoding="utf-8")
        return str(report_path)

    # ── HTML template pieces ──────────────────────────────────────────

    @staticmethod
    def _html_header() -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>EDA Report</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #f0f2f5;
    color: #1e1e2e;
    padding: 32px 16px;
  }
  .container { max-width: 1100px; margin: 0 auto; }
  h1 {
    text-align: center;
    font-size: 2rem;
    margin-bottom: 8px;
    color: #2a2a4a;
  }
  .subtitle {
    text-align: center;
    color: #6b7280;
    margin-bottom: 32px;
    font-size: 0.9rem;
  }
  .card {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    padding: 24px 28px;
    margin-bottom: 28px;
  }
  .card h2 {
    font-size: 1.25rem;
    color: #2a2a4a;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 8px;
    margin-bottom: 18px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }
  th {
    background: #2a2a4a;
    color: #fff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 9px 14px;
    border-bottom: 1px solid #e5e7eb;
  }
  tr:nth-child(even) td { background: #f9fafb; }
  .badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-warn  { background: #fef3c7; color: #92400e; }
  .badge-ok    { background: #d1fae5; color: #065f46; }
  .badge-err   { background: #fee2e2; color: #991b1b; }
  .plots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
  }
  .plot-card {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
  }
  .plot-card p {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 6px;
    font-weight: 600;
  }
  .plot-card img { max-width: 100%; border-radius: 6px; }
  .warning-list { list-style: none; }
  .warning-list li {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #f3f4f6;
  }
  .warning-list li:last-child { border-bottom: none; }
  .warning-icon { font-size: 1.2rem; }
  .warning-text strong { color: #92400e; }
  .warning-cols { color: #6b7280; font-size: 0.8rem; margin-top: 2px; }
  .empty-msg { color: #9ca3af; font-style: italic; font-size: 0.9rem; }
</style>
</head>
<body>
<div class="container">
  <h1>📊 Exploratory Data Analysis Report</h1>
  <p class="subtitle">Auto-generated by EDAAgent</p>
"""

    @staticmethod
    def _html_footer() -> str:
        return """</div><!-- /container -->
</body>
</html>"""

    # ── Section builders ──────────────────────────────────────────────

    def _section_overview(self) -> str:
        s = self.report["dataset_summary"]
        ct = s["column_types"]
        rows = [
            ("Rows", f"{s['n_rows']:,}"),
            ("Columns", s["n_columns"]),
            ("Numeric", len(ct["numerical"])),
            ("Categorical", len(ct["categorical"])),
            ("Datetime", len(ct["datetime"])),
            ("Boolean", len(ct["boolean"])),
            ("Memory", f"{s['memory_usage_mb']} MB"),
            ("Duplicate rows", s["duplicate_rows"]),
            ("Target", s["target_column"] or "—"),
        ]
        table_html = "".join(
            f'<tr><td style="font-weight:600;color:#4b5563;">{k}</td><td>{v}</td></tr>'
            for k, v in rows
        )
        return f"""<div class="card">
  <h2>📋 Dataset Overview</h2>
  <table><tbody>{table_html}</tbody></table>
</div>"""

    def _section_warnings(self) -> str:
        warnings = self.report.get("eda_warnings", [])
        if not warnings:
            return '<div class="card"><h2>✅ Warnings</h2><p class="empty-msg">No issues detected.</p></div>'

        items = ""
        for w in warnings:
            cols_html = ""
            if "columns" in w:
                cols_html = f'<div class="warning-cols">Affected: {", ".join(w["columns"])}</div>'
            items += f"""<li>
      <span class="warning-icon">⚠️</span>
      <div>
        <strong>{w['type'].replace('_', ' ').title()}</strong> — {w['message']}
        {cols_html}
      </div>
    </li>"""

        return f'<div class="card"><h2>⚠️ Warnings</h2><ul class="warning-list">{items}</ul></div>'

    def _section_missing_values(self) -> str:
        quality = self.report["data_quality_report"]
        missing = quality["missing_values"]["columns_with_missing"]

        if not missing:
            return '<div class="card"><h2>✔️ Missing Values</h2><p class="empty-msg">No missing values in any column.</p></div>'

        # bar chart
        cols = list(missing.keys())
        pcts = [missing[c]["missing_ratio"] * 100 for c in cols]

        fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.45), 4))
        bars = ax.barh(cols, pcts, color="#f59e0b", edgecolor="#d97706", height=0.55)
        ax.set_xlabel("Missing %", fontsize=9)
        ax.set_title("Missing Values by Column", fontsize=11, fontweight="bold", pad=10)
        ax.axvline(50, color="#ef4444", linestyle="--", linewidth=1.2, label="50 % threshold")
        ax.legend(fontsize=8)
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=8, color="#374151")
        fig.tight_layout()
        img_b64 = _fig_to_b64(fig)

        # summary table
        table_rows = "".join(
            f'<tr><td>{c}</td>'
            f'<td>{missing[c]["missing_count"]:,}</td>'
            f'<td>{missing[c]["missing_ratio"]*100:.1f}%</td>'
            f'<td><span class="badge {"badge-err" if missing[c]["missing_ratio"]>0.5 else "badge-warn"}">'
            f'{"Critical" if missing[c]["missing_ratio"]>0.5 else "Review"}</span></td></tr>'
            for c in cols
        )

        return f"""<div class="card">
  <h2>🔍 Missing Values</h2>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;margin-bottom:18px;"/>
  <table>
    <thead><tr><th>Column</th><th>Missing</th><th>%</th><th>Status</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>"""
    
    def _get_sample(self, series, limit=10000):
        if len(series) > limit:
            return series.sample(limit, random_state=42)
        return series

    def _section_numeric_distributions(self) -> str:
        profiles = self.report["column_profiles"]
        numeric_cols = [c for c, s in profiles.items() if s["data_type"] == "numeric" and c != self.target]

        if not numeric_cols:
            return '<div class="card"><h2>📈 Numeric Distributions</h2><p class="empty-msg">No numeric feature columns.</p></div>'

        # limit plots to 20 columns max for page length
        plot_cols = numeric_cols[:20]
        n = len(plot_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            data = self.df[col].dropna()
            color = PALETTE[i % len(PALETTE)]
            plot_data = self._get_sample(self.df[col].dropna())

            if len(data) > 10000:
                data_sample = data.sample(10000, random_state=42)
            else:
                data_sample = data
            ax.hist(plot_data, bins="auto", color=color, alpha=0.7, edgecolor="white", linewidth=0.8)

            # KDE overlay
            if len(data) > 10 and data.std() > 0:
                kde = scipy_stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax.twinx().plot(x_range, kde(x_range), color=color, linewidth=2)
                ax.twinx().set_ylabel("")  # hide right-axis label

            stats = profiles[col]
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")

            # stats annotation box
            txt = (f"μ={stats['mean']}  σ={stats['std']}\n"
                   f"skew={stats['skewness']}  outliers={stats['outlier_count_iqr']}")
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=7,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6", edgecolor="#d1d5db", alpha=0.9))

        # hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Feature Distributions", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        img_b64 = _fig_to_b64(fig)

        # stats table
        table_rows = "".join(
            f'<tr><td>{c}</td><td>{profiles[c]["dtype"]}</td>'
            f'<td>{profiles[c]["mean"]}</td><td>{profiles[c]["std"]}</td>'
            f'<td>{profiles[c]["skewness"]}</td>'
            f'<td>{profiles[c]["outlier_count_iqr"]} ({profiles[c]["outlier_ratio_iqr"]*100:.1f}%)</td>'
            f'<td><span class="badge {"badge-ok" if profiles[c].get("is_normal") else "badge-warn"}">'
            f'{"Normal" if profiles[c].get("is_normal") else "Non-normal"}</span></td></tr>'
            for c in numeric_cols
        )

        return f"""<div class="card">
  <h2>📈 Numeric Feature Distributions</h2>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;margin-bottom:20px;"/>
  <table>
    <thead><tr><th>Column</th><th>Type</th><th>Mean</th><th>Std</th><th>Skew</th><th>Outliers</th><th>Normal?</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>"""

    def _section_categorical_distributions(self) -> str:
        profiles = self.report["column_profiles"]
        cat_cols = [c for c, s in profiles.items() if s["data_type"] == "categorical" and c != self.target]

        if not cat_cols:
            return '<div class="card"><h2>🏷️ Categorical Distributions</h2><p class="empty-msg">No categorical feature columns.</p></div>'

        plot_cols = cat_cols[:12]
        n = len(plot_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.0))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            top = profiles[col]["top_values"]
            labels = [str(k) for k in top.keys()]
            values = list(top.values())
            color = PALETTE[(i + 4) % len(PALETTE)]

            bars = ax.barh(labels[::-1], values[::-1], color=color, edgecolor="white", height=0.6)
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.set_xlabel("Count", fontsize=8)
            for bar in bars:
                ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{int(bar.get_width()):,}", va="center", fontsize=7, color="#374151")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Categorical Feature Distributions (top values)", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        img_b64 = _fig_to_b64(fig)

        table_rows = "".join(
            f'<tr><td>{c}</td><td>{profiles[c]["unique_count"]}</td>'
            f'<td>{profiles[c]["missing_ratio"]*100:.1f}%</td>'
            f'<td><span class="badge {"badge-err" if profiles[c].get("is_high_cardinality") else "badge-ok"}">'
            f'{"High" if profiles[c].get("is_high_cardinality") else "Normal"}</span></td></tr>'
            for c in cat_cols
        )

        return f"""<div class="card">
  <h2>🏷️ Categorical Feature Distributions</h2>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;margin-bottom:20px;"/>
  <table>
    <thead><tr><th>Column</th><th>Unique</th><th>Missing %</th><th>Cardinality</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>"""

    def _section_correlation_heatmap(self) -> str:
        numeric_cols = [
            c for c in self.df.select_dtypes(include=["number"]).columns
            if self.df[c].std() > 0
        ]

        if len(numeric_cols) < 2:
            return '<div class="card"><h2>🔗 Correlations</h2><p class="empty-msg">Fewer than 2 numeric columns with variance — no heatmap.</p></div>'

        corr = self.df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        # fig, ax = plt.subplots(figsize=(max(7, len(numeric_cols) * 0.55), max(5.5, len(numeric_cols) * 0.5)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, linewidths=0.5,
                    annot_kws={"size": 7}, ax=ax, square=True,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold", pad=12)
        fig.tight_layout()
        img_b64 = _fig_to_b64(fig)

        # strong-pairs table
        relationships = self.report.get("relationship_insights", {})
        strong = (relationships.get("numeric_correlations") or {}).get("strong_pairs", [])

        if strong:
            pair_rows = "".join(
                f'<tr><td>{p["feature_1"]}</td><td>{p["feature_2"]}</td>'
                f'<td style="color:{"#dc2626" if p["correlation"]<0 else "#16a34a"};font-weight:600;">{p["correlation"]}</td></tr>'
                for p in strong
            )
            pair_table = f"""<table style="margin-top:14px;max-width:500px;">
      <thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th></tr></thead>
      <tbody>{pair_rows}</tbody>
    </table>"""
        else:
            pair_table = '<p class="empty-msg" style="margin-top:12px;">No strongly correlated pairs (threshold |r| ≥ 0.5).</p>'

        return f"""<div class="card">
  <h2>🔗 Correlation Heatmap</h2>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;margin-bottom:16px;"/>
  <h3 style="font-size:0.95rem;color:#374151;margin-bottom:8px;">Strong Pairs (|r| ≥ 0.5)</h3>
  {pair_table}
</div>"""

    def _section_target_analysis(self) -> str:
        target = self.report.get("target_analysis")
        if not target:
            return '<div class="card"><h2>🎯 Target Analysis</h2><p class="empty-msg">No target column specified.</p></div>'

        task = target.get("task_type", "unknown")
        series = self.df[self.target].dropna()

        if task == "classification":
            dist = target.get("class_distribution", {})
            labels = list(dist.keys())
            values = list(dist.values())

            fig, ax = plt.subplots(figsize=(7, 3.8))
            colors = PALETTE[:len(labels)]
            bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
            ax.set_ylabel("Proportion", fontsize=9)
            ax.set_title(f"Target Distribution — {self.target} (Classification)", fontsize=11, fontweight="bold")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.1%}", ha="center", fontsize=9, fontweight="bold", color="#374151")

            if target.get("imbalance_ratio") and target["imbalance_ratio"] >= 3:
                ax.text(0.98, 0.95, f"⚠ Imbalance ratio: {target['imbalance_ratio']}x",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8,
                        color="#991b1b", bbox=dict(boxstyle="round", facecolor="#fee2e2", edgecolor="#fca5a5"))
            fig.tight_layout()
            img_b64 = _fig_to_b64(fig)

            info_rows = [
                ("Task Type", '<span class="badge badge-ok">Classification</span>'),
                ("Binary?", "Yes" if target.get("is_binary") else "No"),
                ("Classes", target.get("n_classes")),
                ("Imbalance Ratio", target.get("imbalance_ratio") or "—"),
                ("Majority Class %", f"{target.get('majority_class_ratio', 0)*100:.1f}%"),
            ]

        else:  # regression
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.hist(series, bins="auto", color=PALETTE[2], alpha=0.75, edgecolor="white")
            kde = scipy_stats.gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 200)
            ax2 = ax.twinx()
            ax2.plot(x_range, kde(x_range), color=PALETTE[2], linewidth=2.2)
            ax2.set_ylabel("")
            ax.set_title(f"Target Distribution — {self.target} (Regression)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Count", fontsize=9)
            fig.tight_layout()
            img_b64 = _fig_to_b64(fig)

            info_rows = [
                ("Task Type", '<span class="badge badge-ok">Regression</span>'),
                ("Mean", target.get("mean")),
                ("Std", target.get("std")),
                ("Skewness", target.get("skewness")),
                ("Range", f"{target.get('min')} → {target.get('max')}"),
            ]

        table_html = "".join(
            f'<tr><td style="font-weight:600;color:#4b5563;">{k}</td><td>{v}</td></tr>'
            for k, v in info_rows
        )

        return f"""<div class="card">
  <h2>🎯 Target Analysis</h2>
  <img src="data:image/png;base64,{img_b64}" style="max-width:100%;border-radius:8px;margin-bottom:18px;"/>
  <table style="max-width:450px;"><tbody>{table_html}</tbody></table>
</div>"""

    # ==================================================================
    # Unified export  (single call — routes automatically)
    # ==================================================================

    def export(
        self,
        plan_dir: str = "Plan",
        output_dir: str = "Output",
    ) -> Dict[str, Any]:
        """
        Single entry-point after run().

        Routes based on run_type:
            "raw"   → preprocessing_context.json  +  HTML report
            "clean" → automl_context.json         +  HTML report

        Returns a dict with keys pointing to every generated artefact.
        """
        if not self.report:
            raise ValueError("Call run() first.")

        result: Dict[str, Any] = {}
        run_type = self.report["run_type"]

        if run_type == "raw":
            result["preprocessing_context"] = self.generate_preprocessing_context(plan_dir, output_dir)
        elif run_type == "clean":
            result["automl_context"] = self.generate_automl_context(plan_dir, output_dir)
        else:
            raise ValueError(f"Unknown run_type '{run_type}'. Use 'raw' or 'clean'.")

        result["report_path"] = self.generate_report(output_dir)
        return result

    # ==================================================================
    # Shared persistence helper
    # ==================================================================

    @staticmethod
    def _persist_json(data: Any, filename: str, plan_dir: str, output_dir: str) -> None:
        payload = json.dumps(data, indent=2)
        for dir_path in (plan_dir, output_dir):
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)
            (path / filename).write_text(payload, encoding="utf-8")