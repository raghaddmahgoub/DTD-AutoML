import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
from scipy import stats as scipy_stats


class EDAAgent:
    """
    Autonomous Exploratory Data Analysis (EDA) Agent.

    Performs descriptive analysis only — no transformations or decisions.
    Produces a structured report consumed by downstream preprocessing agents.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset to analyze.
    target_column : str, optional
        Name of the target/label column, if supervised.
    df_name : str
        Identifier used when persisting outputs to disk.
    top_k : int
        Number of top categorical values to retain in column profiles.
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
        """Map a pandas Series to one of four canonical type labels."""
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
        """
        Global characterization: shape, types, memory, duplicates, target metadata.
        Purely descriptive — no filtering or transformation.
        """
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

    # ------------------------------------------------------------------
    # 2. Per-column profiling
    # ------------------------------------------------------------------

    def _column_profiles(self) -> Dict[str, Any]:
        """
        Per-column descriptive statistics, grouped by inferred type.
        Numeric columns also include outlier counts and a normality flag.
        """
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

                # Shapiro-Wilk on a sample (max 5 000 rows, needs ≥ 3 points)
                is_normal: Optional[bool] = None
                if len(clean) >= 3:
                    sample = clean.sample(min(len(clean), 5000), random_state=42)
                    _, p_value = scipy_stats.shapiro(sample)
                    is_normal = bool(p_value > 0.05)

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
        """
        Descriptive breakdown of the target column.
        Signals task type, class distribution, and imbalance for
        both binary and multiclass targets.
        """
        if self.target is None or self.target not in self.df.columns:
            return None

        series = self.df[self.target]
        unique_values = series.dropna().unique()

        analysis: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": round(float(series.isna().mean()), 4),
            "unique_values": int(series.nunique(dropna=True)),
        }

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_discrete = is_numeric and len(unique_values) <= 20  # treat low-cardinality numerics as classification

        if is_numeric and not is_discrete:
            # --- Regression ---
            clean = series.dropna()
            analysis["task_type"] = "regression"
            analysis.update({
                "mean": round(float(clean.mean()), 4),
                "std": round(float(clean.std()), 4),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "median": float(clean.median()),
                "skewness": round(float(clean.skew()), 4),
            })
        else:
            # --- Classification (binary or multiclass) ---
            value_counts = series.value_counts(dropna=True)
            total = value_counts.sum()
            n_classes = len(value_counts)

            analysis["task_type"] = "classification"
            analysis["is_binary"] = (n_classes == 2)
            analysis["n_classes"] = n_classes
            analysis["class_distribution"] = {
                str(k): round(float(v / total), 4) for k, v in value_counts.items()
            }
            analysis["majority_class_ratio"] = round(float(value_counts.max() / total), 4)

            # Imbalance ratio is meaningful for any number of classes ≥ 2
            if value_counts.min() > 0:
                analysis["imbalance_ratio"] = round(float(value_counts.max() / value_counts.min()), 2)
            else:
                analysis["imbalance_ratio"] = None  # guard against zero-count edge case

        return analysis

    # ------------------------------------------------------------------
    # 4. Data quality report
    # ------------------------------------------------------------------

    def _data_quality_report(self) -> Dict[str, Any]:
        """
        Flags missing values, duplicates, constant/near-constant columns,
        unique-per-row identifiers, and mixed-type object columns.
        """
        n_rows = len(self.df)
        na_df = self.df.isna()
        dup_mask = self.df.duplicated()

        # --- Missing values ---
        missing_by_column = {
            col: {
                "missing_count": int(na_df[col].sum()),
                "missing_ratio": round(float(na_df[col].mean()), 4),
            }
            for col in self.df.columns
            if na_df[col].any()
        }

        # --- Constant / near-constant / unique-per-row ---
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

        # --- Mixed-type object columns ---
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
        """
        Descriptive pairwise relationships.
        Covers:
          - Numeric ↔ Numeric   : Pearson correlation (threshold 0.5)
          - Numeric  ↔ Target   : correlation (numeric target) or group means (categorical target)
          - Categorical ↔ Target: Cramér's V (categorical target only)
        """
        insights: Dict[str, Any] = {}
        CORR_THRESHOLD = 0.5

        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)

        # --- Numeric ↔ Numeric correlations ---
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

        # --- Feature ↔ Target relationships ---
        if not (self.target and self.target in self.df.columns):
            insights["target_relationships"] = None
            return insights

        target_series = self.df[self.target]

        if pd.api.types.is_numeric_dtype(target_series):
            # Numeric target → Pearson correlation with each numeric feature
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
            # Categorical target → group means + Cramér's V for categoricals
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
                cramers_v[col] = round(self._cramers_v(self.df[col], target_series), 3)

            insights["target_relationships"] = {
                "target_type": "categorical",
                "group_means": group_means,
                "cramers_v": cramers_v,
            }

        return insights

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """
        Compute Cramér's V between two categorical Series.
        Measures association strength in [0, 1] regardless of class count.
        """
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
        """
        Emit descriptive signals for downstream agents.
        Warnings are never acted on here — they inform the preprocessing plan.
        """
        warnings: List[Dict[str, Any]] = []
        n_rows = dataset_summary["n_rows"]
        n_cols = dataset_summary["n_columns"]

        # --- Dataset scale ---
        if n_rows < 100:
            warnings.append({
                "type": "small_dataset",
                "message": "Dataset contains fewer than 100 rows; model generalization may be limited.",
            })
        if n_rows < n_cols:
            warnings.append({
                "type": "wide_dataset",
                "message": "Number of features exceeds number of rows, increasing overfitting risk.",
            })

        # --- Missing data ---
        high_missing = [
            col for col, s in column_profiles.items() if s["missing_ratio"] > 0.5
        ]
        if high_missing:
            warnings.append({
                "type": "high_missingness",
                "columns": high_missing,
                "message": "Some columns have more than 50% missing values.",
            })

        # --- Constant columns ---
        constant = [
            col for col, s in column_profiles.items() if s["unique_count"] <= 1
        ]
        if constant:
            warnings.append({
                "type": "constant_columns",
                "columns": constant,
                "message": "Some columns contain a single unique value and carry no information.",
            })

        # --- High-cardinality categoricals ---
        high_card = [
            col for col, s in column_profiles.items()
            if s["data_type"] == "categorical" and s.get("is_high_cardinality", False)
        ]
        if high_card:
            warnings.append({
                "type": "high_cardinality_categoricals",
                "columns": high_card,
                "message": "Some categorical columns have very high cardinality.",
            })

        # --- Unique-per-row identifiers ---
        id_cols = [
            col for col, s in column_profiles.items() if s.get("is_unique_per_row", False)
        ]
        if id_cols:
            warnings.append({
                "type": "unique_per_row_columns",
                "columns": id_cols,
                "message": "Some columns have a unique value per row and likely represent identifiers.",
            })

        # --- Outlier-heavy numeric columns ---
        outlier_heavy = [
            col for col, s in column_profiles.items()
            if s["data_type"] == "numeric" and s.get("outlier_ratio_iqr", 0) > 0.05
        ]
        if outlier_heavy:
            warnings.append({
                "type": "high_outlier_ratio",
                "columns": outlier_heavy,
                "message": "Some numeric columns have more than 5% outliers (IQR method).",
            })

        # --- Non-normal numeric columns ---
        non_normal = [
            col for col, s in column_profiles.items()
            if s["data_type"] == "numeric" and s.get("is_normal") is False
        ]
        if non_normal:
            warnings.append({
                "type": "non_normal_columns",
                "columns": non_normal,
                "message": "Some numeric columns are not normally distributed (Shapiro-Wilk).",
            })

        # --- Class imbalance (binary AND multiclass) ---
        if target_analysis and target_analysis.get("task_type") == "classification":
            imbalance_ratio = target_analysis.get("imbalance_ratio")
            if imbalance_ratio is not None and imbalance_ratio >= 3:
                warnings.append({
                    "type": "class_imbalance",
                    "imbalance_ratio": imbalance_ratio,
                    "message": "Target variable shows class imbalance (majority/minority ≥ 3).",
                })

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
            Pipeline stage indicator persisted in the report.

        Returns
        -------
        dict
            Complete structured EDA report.
        """
        self.report = {
            "run_type": run_type,
            "dataset_summary": self._dataset_summary(),
            "column_profiles": self._column_profiles(),
            "target_analysis": self._target_analysis(),
            "data_quality_report": self._data_quality_report(),
            "relationship_insights": self._relationship_insights(),
        }
        # Warnings depend on earlier sections, so computed last
        self.report["eda_warnings"] = self._generate_eda_warnings(
            dataset_summary=self.report["dataset_summary"],
            column_profiles=self.report["column_profiles"],
            target_analysis=self.report["target_analysis"],
        )
        return self.report

    # ------------------------------------------------------------------
    # 8. Preprocessing context export
    # ------------------------------------------------------------------
    def _collect_sample_values(self, col: str, n: int = 5) -> List[Any]:
        """
        Return up to *n* representative non-null values from the column,
        preserving original insertion order (i.e. the first *n* distinct
        values that appear in the DataFrame).

        Native numpy/pandas scalar types are cast to plain Python so the
        list is always JSON-serialisable.
        """
        seen: List[Any] = []
        for val in self.df[col]:
            if pd.isna(val):
                continue
            # cast numpy scalars → Python natives
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
            Produce and persist a flat, per-column preprocessing context.

            Output is a JSON array where every element describes exactly one
            column:

                [
                    {
                        "column":        <str>   – column name,
                        "dtype":         <str>   – pandas dtype string,
                        "missing_pct":   <float> – percentage of missing values (0-100),
                        "n_unique":      <int>   – number of distinct non-null values,
                        "sample_values": <list>  – first *sample_size* distinct values,
                        "is_target":     <bool>  – whether this is the target column,

                        // numeric columns only:
                        "mean":          <float>,
                        "std":           <float>,
                        "skew":          <float>
                    },
                    ...
                ]

            Parameters
            ----------
            plan_dir : str
                Directory for the plan-stage copy of the JSON.
            output_dir : str
                Directory for the output-stage copy of the JSON.
            sample_size : int
                How many distinct sample values to include per column.

            Returns
            -------
            list[dict]
                The assembled context (also written to disk).

            Raises
            ------
            ValueError
                If `run()` has not been called first.
            """
            if not self.report:
                raise ValueError("EDA must be run before generating preprocessing context.")

            columns = self.report["column_profiles"]
            preprocessing_context: List[Dict[str, Any]] = []

            for col, stats in columns.items():
                entry: Dict[str, Any] = {
                    "column": col,
                    "dtype": stats["dtype"],
                    "missing_pct": round(stats["missing_ratio"] * 100, 2),
                    "n_unique": stats["unique_count"],
                    "sample_values": self._collect_sample_values(col, sample_size),
                    "is_target": (col == self.target),
                }

                # Append numeric stats only for numeric columns
                if stats["data_type"] == "numeric":
                    entry["mean"] = stats.get("mean")
                    entry["std"] = stats.get("std")
                    entry["skew"] = stats.get("skewness")

                preprocessing_context.append(entry)

            # --- Persist to disk ---
            for dir_path in (plan_dir, output_dir):
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{self.df_name}_preprocessing_context.json").write_text(
                    json.dumps(preprocessing_context, indent=2), encoding="utf-8"
                )

            return preprocessing_context
    # def generate_preprocessing_context(
    #     self,
    #     plan_dir: str = "Plan",
    #     output_dir: str = "Output",
    # ) -> Dict[str, Any]:
    #     """
    #     Produce and persist a read-only contract for the preprocessing agent.
    #     Raises ValueError if `run()` has not been called first.
    #     """
    #     if not self.report:
    #         raise ValueError("EDA must be run before generating preprocessing context.")

    #     summary = self.report["dataset_summary"]
    #     quality = self.report["data_quality_report"]
    #     columns = self.report["column_profiles"]
    #     target = self.report.get("target_analysis")
    #     relationships = self.report.get("relationship_insights")

    #     # --- Per-column context (flattened for the preprocessing agent) ---
    #     column_context: Dict[str, Any] = {}
    #     for col, stats in columns.items():
    #         entry: Dict[str, Any] = {
    #             "data_type": stats["data_type"],
    #             "dtype": stats["dtype"],
    #             "missing_ratio": stats["missing_ratio"],
    #             "missing_count": stats["missing_count"],
    #             "unique_count": stats["unique_count"],
    #         }

    #         if stats["data_type"] == "numeric":
    #             entry.update({
    #                 "mean": stats.get("mean"),
    #                 "std": stats.get("std"),
    #                 "min": stats.get("min"),
    #                 "max": stats.get("max"),
    #                 "skewness": stats.get("skewness"),
    #                 "zero_count": stats.get("zero_count"),
    #                 "outlier_count_iqr": stats.get("outlier_count_iqr"),
    #                 "outlier_ratio_iqr": stats.get("outlier_ratio_iqr"),
    #                 "is_normal": stats.get("is_normal"),
    #             })
    #         elif stats["data_type"] == "categorical":
    #             entry.update({
    #                 "top_values": stats.get("top_values"),
    #                 "is_high_cardinality": stats.get("is_high_cardinality", False),
    #             })
    #         elif stats["data_type"] == "datetime":
    #             entry.update({
    #                 "min_date": stats.get("min_date"),
    #                 "max_date": stats.get("max_date"),
    #             })

    #         column_context[col] = entry

    #     # --- Assembled context ---
    #     preprocessing_context: Dict[str, Any] = {
    #         "meta": {
    #             "run_type": self.report.get("run_type"),
    #             "n_rows": summary["n_rows"],
    #             "n_columns": summary["n_columns"],
    #             "memory_usage_mb": summary["memory_usage_mb"],
    #             "duplicate_rows": summary["duplicate_rows"],
    #         },
    #         "target": target,
    #         "columns": column_context,
    #         "data_quality": {
    #             "unique_per_row_columns": quality.get("unique_per_row_columns", []),
    #             "columns_with_missing": list(quality["missing_values"]["columns_with_missing"].keys()),
    #             "constant_columns": quality["low_variance_columns"]["constant_columns"],
    #             "near_constant_columns": list(quality["low_variance_columns"]["near_constant_columns"].keys()),
    #             "mixed_type_columns": quality["type_issues"]["mixed_type_columns"],
    #         },
    #         "relationship_insights": relationships,
    #         "eda_warnings": self.report.get("eda_warnings", []),
    #     }

    #     # --- Persist to disk ---
    #     for dir_path in (plan_dir, output_dir):
    #         path = Path(dir_path)
    #         path.mkdir(parents=True, exist_ok=True)
    #         (path / f"{self.df_name}_preprocessing_context.json").write_text(
    #             json.dumps(preprocessing_context, indent=2), encoding="utf-8"
    #         )

    #     return preprocessing_context

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
            POSITIVE_KEYWORDS = {"target", "label", "price", "score", "rating", "outcome", "survived"}
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

