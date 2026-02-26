import json
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Optional, Dict, Any, List
from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.feature_selection import f_classif
import numpy as np
from src.utils.logger import Logger

logger = Logger()
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
        logger.info(f"[EDA Agent] Starting {run_type.upper()} data analysis...")

        self.report = {
            "run_type": run_type,
            "dataset_summary": self._dataset_summary(),
            "feature_scale_analysis": self._feature_scale_analysis(),
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

        self.report["total_feature_count"] = len(numeric_features) + len(categorical_features)
        self.report["multicollinearity"] = {
            "threshold": 0.7,
            "pairs": multicollinear_pairs,
            }
        self.report["signal_analysis"] = signal_analysis

        logger.info(f"[EDA Agent] {run_type.upper()} analysis complete. "
                    f"Found {len(self.report['eda_warnings'])} warnings.")
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
        metrics
        """
        if not self.report:
            logger.error("[EDA Agent] Attempted to generate AutoML context before running analysis.")
            raise ValueError("Run EDA before generating context.")

        logger.info("[EDA Agent] Generating Meaningful JSON directives for AutoML Agent...")
        
        
        # ── assemble ────────────────────────────────────────────────
        automl_context: Dict[str, Any] = {
            "report": self.report,
        }

        self._persist_json(automl_context, f"{self.df_name}_automl_context.json", plan_dir, output_dir)
        logger.info(f"[EDA Agent] AutoML context saved to {output_dir}.")
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

    # ==================================================================
    # OUTPUT C — Frontend-ready JSON
    # ==================================================================
    
    def generate_frontend_json(self, output_dir: str = "Output") -> str:
        """
        Builds a comprehensive JSON containing raw stats and chart-ready data.
        """
        if not self.report:
            raise ValueError("Run EDA before generating frontend data.")

        # Prepare Chart Data for Frontend
        visualizations = {
            "missing_values_chart": self._get_missing_values_data(),
            "numeric_distributions": self._get_numeric_distribution_data(),
            "categorical_distributions": self._get_categorical_distribution_data(),
            "correlation_matrix": self._get_correlation_matrix_data(),
        }

        frontend_payload = {
            "metadata": {
                "df_name": self.df_name,
                "timestamp": pd.Timestamp.now().isoformat(),
                "run_type": self.report["run_type"]
            },
            "report": self.report,
            "visualizations": visualizations
        }

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / f"{self.df_name}_frontend_data.json"
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(frontend_payload, f, indent=2, default=str)
            
        return str(json_path)

    # --- Chart Data Extractors (Frontend ready) ---

    def _get_missing_values_data(self):
        quality = self.report["data_quality_report"]
        missing = quality["missing_values"]["columns_with_missing"]
        return [
            {"column": col, "count": info["missing_count"], "ratio": info["missing_ratio"]}
            for col, info in missing.items()
        ]

    def _get_numeric_distribution_data(self):
        profiles = self.report["column_profiles"]
        numeric_data = {}
        for col, stats in profiles.items():
            if stats["data_type"] == "numeric" and col != self.target:
                series = self.df[col].dropna()
                # Sample for frontend performance
                sample = series.sample(min(len(series), 50), random_state=42).tolist()
                # Generate histogram bins on backend to save frontend CPU
                counts, bin_edges = np.histogram(series, bins=20)
                numeric_data[col] = {
                    "histogram": {"counts": counts.tolist(), "bins": bin_edges.tolist()},
                    "raw_sample": sample 
                }
        return numeric_data

    def _get_categorical_distribution_data(self):
        profiles = self.report["column_profiles"]
        return {
            col: stats["top_values"]
            for col, stats in profiles.items()
            if stats["data_type"] == "categorical" and col != self.target
        }

    def _get_correlation_matrix_data(self):
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) < 2: return None
        
        corr = self.df[numeric_cols].corr().round(3)
        return {
            "columns": numeric_cols,
            "values": corr.values.tolist()
        }
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
            "raw"   → preprocessing_context.json  +  frontend json report
            "clean" → automl_context.json         +  frontend json report

        Returns a dict with keys pointing to every generated artefact.
        """
        if not self.report:
            logger.error("[EDA Agent] Export failed: No report data found.")
            raise ValueError("Call run() first.")

        result: Dict[str, Any] = {}
        run_type = self.report["run_type"]
        logger.info(f"[EDA Agent] Exporting artifacts for {run_type} stage...")

        if run_type == "raw":
            result["preprocessing_context"] = self.generate_preprocessing_context(plan_dir, output_dir)
        elif run_type == "clean":
            result["automl_context"] = self.generate_automl_context(plan_dir, output_dir)
        else:
            raise ValueError(f"Unknown run_type '{run_type}'. Use 'raw' or 'clean'.")

        # Generate the frontend JSON for Node.js
        result["frontend_json_path"] = self.generate_frontend_json(output_dir)
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


class TargetInferenceAgent:
    """
    Infers the most likely target column using structural, semantic,
    and distributional heuristics.
    """

    ID_KEYWORDS = {"id", "uuid", "vin", "index"}
    TARGET_KEYWORDS = {"target", "label", "class", "price", "score", "rating", "outcome"}

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> Dict[str, Any]:
        scores: Dict[str, float] = {}
    
        n_rows = len(self.df)
    
        for col in self.df.columns:
            series = self.df[col]
            score = 0.0
            name = col.lower()
    
            # --- Hard exclusions ---
            if any(k in name for k in self.ID_KEYWORDS):
                continue
            
            nunique = series.nunique(dropna=True)
            missing_ratio = series.isna().mean()
    
            # --- Missingness (targets are usually observed) ---
            if missing_ratio < 0.05:
                score += 1.0
            elif missing_ratio > 0.3:
                score -= 1.0
    
            # --- Cardinality signal ---
            if nunique == 2:
                score += 3.0  # VERY strong signal (Survived)
            elif 2 < nunique <= 10:
                score += 1.5
            elif nunique < n_rows:
                score += 0.3
    
            # --- Distribution signal ---
            if nunique > 1:
                value_counts = series.value_counts(normalize=True, dropna=True)
                majority_ratio = value_counts.iloc[0]
    
                if 0.5 <= majority_ratio <= 0.9:
                    score += 1.0  # good classification target
                elif majority_ratio > 0.95:
                    score -= 1.0  # near-constant
    
            # --- Type signal ---
            if pd.api.types.is_numeric_dtype(series):
                score += 0.5  # reduced (was too dominant)
            elif nunique <= 20:
                score += 0.3
    
            # --- Semantic signals ---
            POSITIVE_KEYWORDS = {"target", "label", "price", "score", "rating", "outcome"}
            NEGATIVE_KEYWORDS = {"class", "level", "rank", "group"}
    
            if any(k in name for k in POSITIVE_KEYWORDS):
                score += 2.5
    
            if any(k in name for k in NEGATIVE_KEYWORDS):
                score -= 1.5  # penalize Pclass-style features
    
            scores[col] = score
    
        if not scores:
            return {
                "inferred_target": None,
                "confidence": 0.0,
                "alternatives": [],
            }
    
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_col, best_score = ranked[0]
    
        total = sum(abs(s) for _, s in ranked[:3]) or 1.0
        confidence = round(min(0.95, best_score / total), 3)
    
        return {
            "inferred_target": best_col,
            "confidence": confidence,
            "alternatives": [c for c, _ in ranked[1:3]],
        }