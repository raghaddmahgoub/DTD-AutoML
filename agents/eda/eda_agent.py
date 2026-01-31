import json
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
import json


class EDAAgent:
    """
    Autonomous Data Analysis (EDA) Agent.
    This agent performs descriptive analysis only and produces
    a structured EDA report for downstream agents.
    """
    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df
        self.target = target_column
        self.report: Dict[str, Any] = {}
    
    def _dataset_summary(self) -> Dict[str, Any]:
        """
        Descriptive only — no decisions or transformations.
        “The dataset summary provides a global characterization of the dataset, 
        allowing downstream agents to reason about scale, data types, memory constraints,
          and target formulation before any transformation is applied.”
        """
        summary = {}

        summary["n_rows"] = int(self.df.shape[0])
        summary["n_columns"] = int(self.df.shape[1])

        summary["column_types"] = {
            "numerical": self.df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical": self.df.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime": self.df.select_dtypes(include=["datetime"]).columns.tolist(),
            "boolean": self.df.select_dtypes(include=["bool"]).columns.tolist(),
        }

        summary["memory_usage_mb"] = round(
            self.df.memory_usage(deep=True).sum() / (1024 ** 2), 2
        )

        summary["duplicate_rows"] = int(self.df.duplicated().sum())

        if self.target and self.target in self.df.columns:
            summary["target_column"] = self.target
            summary["target_dtype"] = str(self.df[self.target].dtype)
        else:
            summary["target_column"] = None
            summary["target_dtype"] = None

        return summary
    
    def _infer_column_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "categorical"

    def column_info(self,k=5) -> dict:
        profiles = {}
        n_rows = len(self.df)

        for col in self.df.columns:
            series = self.df[col]
            data_type = self._infer_column_type(series)

            profile = {
                "data_type": data_type,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "is_unique_per_row": int(series.nunique(dropna=True)) == n_rows,
            }

            # numerical columns
            if data_type == "numeric":
                profile.update({
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "skewness": float(series.skew()),
                    "zero_count": int((series == 0).sum()),
                })

            #categorical columns
            elif data_type == "categorical":
                value_counts = series.value_counts(dropna=True)
                top_k = value_counts.head(k).to_dict()

                profile.update({
                    "top_values": top_k,
                    "is_high_cardinality": profile["unique_count"] > 0.5 * n_rows,
                })

            # datetime columns
            elif data_type == "datetime":
                profile.update({
                    "min_date": str(series.min()),
                    "max_date": str(series.max()),
                })

            profiles[col] = profile

        return profiles
        
    def _target_analysis(self) -> Optional[Dict[str, Any]]:
      """
      Performs descriptive analysis on the target column. It allows downstream agents to infer
      task type (classification vs regression), class imbalance, and
      distributional properties.
      """
      if self.target is None or self.target not in self.df.columns:
          return None

      series = self.df[self.target]
      analysis: Dict[str, Any] = {}

      analysis["dtype"] = str(series.dtype)
      analysis["missing_count"] = int(series.isna().sum())
      analysis["missing_ratio"] = float(series.isna().mean())
      analysis["unique_values"] = int(series.nunique(dropna=True))

      # NUMERIC TARGET (Regression)
      if pd.api.types.is_numeric_dtype(series):
          analysis["task_type"] = "regression"
          analysis.update({
              "mean": float(series.mean()),
              "std": float(series.std()),
              "min": float(series.min()),
              "max": float(series.max()),
              "median": float(series.median()),
              "skewness": float(series.skew()),
          })

      # CATEGORICAL / BOOLEAN TARGET (Classification)
      else:
          analysis["task_type"] = "classification"

          value_counts = series.value_counts(dropna=True)
          total = value_counts.sum()

        # k = the class label (the value itself)
        # v = count of that class
        
          class_distribution = {
              str(k): float(v / total)
              for k, v in value_counts.items()
          }

          analysis["n_classes"] = len(value_counts)
          analysis["class_distribution"] = class_distribution
          analysis["majority_class_ratio"] = float(value_counts.max() / total)

          if len(value_counts) == 2:
              analysis["is_binary"] = True
              analysis["imbalance_ratio"] = float(
                  value_counts.max() / value_counts.min()
              )
          else:
              analysis["is_binary"] = False

      return analysis
    
    def _data_quality_report(self) -> Dict[str, Any]:
     """
    This report highlights potential data quality issues
     """
     report: Dict[str, Any] = {}
     n_rows = len(self.df)
     unique_per_row_columns = []
     na_df = self.df.isna()
     dup_mask = self.df.duplicated()


     # Missing Values
     missing_by_column = {
         col: {
             "missing_count": int(na_df[col].sum()),
             "missing_ratio": float(na_df[col].mean())
         }
         for col in self.df.columns
         if na_df[col].any()
     }

     report["missing_values"] = {
         "total_missing_cells": int(na_df.sum().sum()),
         "columns_with_missing": missing_by_column,
         "n_columns_with_missing": len(missing_by_column)
     }

     # Duplicate Rows
     report["duplicates"] = {
         "duplicate_row_count": int(dup_mask.sum()),
         "duplicate_ratio": float(dup_mask.mean())
     }

     # Constant / Near-Constant Columns
     constant_columns = []
     near_constant_columns = {}

     for col in self.df.columns:
         series = self.df[col]
         nunique = series.nunique(dropna=True)

         if nunique == 1:
             constant_columns.append(col)
         elif nunique == n_rows:
            unique_per_row_columns.append(col)
         elif nunique > 1:
             counts = series.value_counts(dropna=True)
             top_freq = counts.iloc[0] / n_rows
             if top_freq > 0.95:
                 near_constant_columns[col] = float(top_freq)


     report["low_variance_columns"] = {
         "constant_columns": constant_columns, # Completely useless for ML
         "near_constant_columns": near_constant_columns
     }

     # -----------------------
     # Data Type Inconsistencies
     # -----------------------
     mixed_type_columns = []

     for col in self.df.select_dtypes(include=["object"]).columns:
         inferred_types = self.df[col].dropna().map(type).nunique()
         if inferred_types > 1:
             mixed_type_columns.append(col)

     report["type_issues"] = {
         "mixed_type_columns": mixed_type_columns
     }
     report["unique_per_row_columns"] = unique_per_row_columns

     return report

    def _relationship_insights(self) -> Dict[str, Any]:
        """
        Computes descriptive relationships between features and target.
        NO feature selection, NO decisions, NO transformations.
        """
        insights = {}

        # Identify numeric columns    
        numeric_cols = self.df.select_dtypes(include=["number"]).columns.tolist()

        if self.target in numeric_cols:
            numeric_cols.remove(self.target)

        # 1. Numeric ↔ Numeric Correlation        
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()

            strong_pairs = []
            CORR_THRESHOLD = 0.5  # descriptive only

            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_value = corr_matrix.iloc[i, j]

                    if pd.notna(corr_value) and abs(corr_value) >= CORR_THRESHOLD:
                        strong_pairs.append({
                            "feature_1": numeric_cols[i],
                            "feature_2": numeric_cols[j],
                            "correlation": round(float(corr_value), 3)
                        })

            insights["numeric_correlations"] = {
                "threshold": CORR_THRESHOLD,
                "strong_pairs": strong_pairs
            }
        else:
            insights["numeric_correlations"] = None

        # 2. Feature ↔ Target Relationship
        if self.target and self.target in self.df.columns:
            target_series = self.df[self.target]

            # Case 1: Numeric target → correlation
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
                    "feature_correlations": target_corr
                }

            # Case 2: Categorical target → group statistics
            else:
                group_stats = {}

                for col in numeric_cols:
                    group_stats[col] = (
                        self.df.groupby(self.target)[col]
                        .mean()
                        .round(3)
                        .to_dict()
                    )

                insights["target_relationships"] = {
                    "target_type": "categorical",
                    "group_means": group_stats
                }
        else:
            insights["target_relationships"] = None

        return insights

    def _generate_eda_warnings(
        self,
        dataset_summary: dict,
        column_profiles: dict,
        unique_id_cols = None,
        target_analysis: Optional[dict] = None) -> list:
        
        if unique_id_cols is None:
            unique_id_cols = [
                col for col, stats in column_profiles.items()
                if stats.get("is_unique_per_row", False)
            ]
        """
        Generates high-level EDA warnings.
        Warnings are descriptive signals for downstream agents — not actions.
        """
        warnings = []

        n_rows = dataset_summary["n_rows"]
        n_cols = dataset_summary["n_columns"]

        # Dataset size warnings
        if n_rows < 100:
            warnings.append({
                "type": "small_dataset",
                "message": "Dataset contains fewer than 100 rows; model generalization may be limited."
            })

        if n_rows < n_cols:
            warnings.append({
                "type": "wide_dataset",
                "message": "Number of features exceeds number of rows, increasing overfitting risk."
            })

        # Missing data warnings
        high_missing_cols = [
            col for col, stats in column_profiles.items()
            if stats["missing_ratio"] > 0.5
        ]

        if high_missing_cols:
            warnings.append({
                "type": "high_missingness",
                "columns": high_missing_cols,
                "message": "Some columns have more than 50% missing values."
            })

        # Constant / near-constant columns
        constant_cols = [
            col for col, stats in column_profiles.items()
            if stats["unique_count"] <= 1
        ]

        if constant_cols:
            warnings.append({
                "type": "constant_columns",
                "columns": constant_cols,
                "message": "Some columns contain a single unique value and carry no information."
            })

        # High-cardinality categoricals
        high_cardinality_cols = [
            col for col, stats in column_profiles.items()
            if stats["data_type"] == "categorical" and stats.get("is_high_cardinality", False)
        ]

        if high_cardinality_cols:
            warnings.append({
                "type": "high_cardinality_categoricals",
                "columns": high_cardinality_cols,
                "message": "Some categorical columns have very high cardinality, which may affect encoding and memory usage."
            })

        # Target-related warnings
        if target_analysis:
            if target_analysis.get("task_type") == "classification":
                imbalance_ratio = target_analysis.get("imbalance_ratio")

                if imbalance_ratio and imbalance_ratio >= 3:
                    warnings.append({
                        "type": "class_imbalance",
                        "message": "Target variable shows strong class imbalance."
                    })
        if unique_id_cols:
            warnings.append({
                "type": "unique_per_row_columns",
                "columns": unique_id_cols,
                "message": "Some columns have a unique value per row and likely represent identifiers rather than predictive features."
            })

        return warnings

    def run(self, run_type: str = "raw") -> Dict[str, Any]:
        """
        Executes the EDA process.
        Parameters:
        - run_type: "raw" or "clean" to indicate pipeline stage
        Returns:
        - Structured EDA report (dict)
        """
        self.report["run_type"] = run_type
        self.report["dataset_summary"] = self._dataset_summary()
        self.report["column_profiles"] = self.column_info()
        self.report["target_analysis"] = self._target_analysis()
        self.report["data_quality_report"] = self._data_quality_report()
        self.report["relationship_insights"] = self._relationship_insights()
        self.report["eda_warnings"] = self._generate_eda_warnings(
        dataset_summary=self.report["dataset_summary"],
        column_profiles=self.report["column_profiles"],
        target_analysis=self.report["target_analysis"] )

        return self.report
    
    def generate_preprocessing_context(
        self,
        plan_dir: str = "Plan",
        output_dir: str = "Output"
    ) -> dict:
        """
        Generates and persists a structured preprocessing context.
        This context is a READ-ONLY contract for the preprocessing agent.
        """

        if not self.report:
            raise ValueError("EDA must be run before generating preprocessing context.")

        summary = self.report["dataset_summary"]
        quality = self.report["data_quality_report"]
        columns = self.report["column_profiles"]
        target = self.report.get("target_analysis")
        relationships = self.report.get("relationship_insights")

        # Column-level signals
        column_context = {}

        for col, stats in columns.items():
            col_entry = {
                "data_type": stats["data_type"],
                "dtype": stats["dtype"],
                "missing_ratio": round(stats["missing_ratio"], 4),
                "missing_count": stats["missing_count"],
                "unique_count": stats["unique_count"],
            }

            if stats["data_type"] == "numeric":
                col_entry.update({
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "min": stats.get("min"),
                    "max": stats.get("max"),
                    "skewness": stats.get("skewness"),
                    "zero_count": stats.get("zero_count"),
                })

            elif stats["data_type"] == "categorical":
                col_entry.update({
                    "top_values": stats.get("top_values"),
                    "is_high_cardinality": stats.get("is_high_cardinality", False),
                })

            elif stats["data_type"] == "datetime":
                col_entry.update({
                    "min_date": stats.get("min_date"),
                    "max_date": stats.get("max_date"),
                })

            column_context[col] = col_entry

        # Final preprocessing context
        preprocessing_context = {
            "meta": {
                "run_type": self.report.get("run_type"),
                "n_rows": summary["n_rows"],
                "n_columns": summary["n_columns"],
                "memory_usage_mb": summary["memory_usage_mb"],
                "duplicate_rows": summary["duplicate_rows"],
            },
            "target": target,
            "columns": column_context,
            "data_quality": {
                "unique_per_row_columns": quality.get("unique_per_row_columns", []),
                "columns_with_missing": list(
                    quality["missing_values"]["columns_with_missing"].keys()
                ),
                "constant_columns": quality["low_variance_columns"]["constant_columns"],
                "near_constant_columns": list(
                    quality["low_variance_columns"]["near_constant_columns"].keys()
                ),
                "mixed_type_columns": quality["type_issues"]["mixed_type_columns"],
            },
            "relationship_insights": relationships,
            "eda_warnings": self.report.get("eda_warnings", []),
        }

        # to disk (Plan + Output)
        plan_path = Path(plan_dir)
        output_path = Path(output_dir)

        plan_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        plan_file = plan_path / "preprocessing_context.json"
        output_file = output_path / "preprocessing_context.json"

        plan_file.write_text(
            json.dumps(preprocessing_context, indent=2),
            encoding="utf-8"
        )
        output_file.write_text(
            json.dumps(preprocessing_context, indent=2),
            encoding="utf-8"
        )

        return preprocessing_context
    
# with open("Plan/preprocessing_context.json") as f:
#     context = json.load(f)

# prep = PreprocessingAgent(context)
# plan = prep.build_plan()



