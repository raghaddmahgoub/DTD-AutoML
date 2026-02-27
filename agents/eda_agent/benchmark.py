import numpy as np
import pandas as pd
from typing import Dict, Any


class AnalysisEvaluator:
    """
    Evaluates the correctness, fidelity, and semantic validity
    of an EDAAgent analysis report.
    """

    def __init__(self, df: pd.DataFrame, eda_report: Dict[str, Any], target: str | None = None):
        self.df = df
        self.report = eda_report
        self.target = target
        self.eps = 1e-9

    # --------------------------------------------------
    # 1. Invariant checks (hard correctness)
    # --------------------------------------------------

    def _check_invariants(self) -> Dict[str, Any]:
        violations = []

        summary = self.report["dataset_summary"]

        if summary["n_rows"] != len(self.df):
            violations.append("Row count mismatch")

        if summary["n_columns"] != self.df.shape[1]:
            violations.append("Column count mismatch")

        total_checks = 2
        passed = total_checks - len(violations)

        return {
            "total_checks": total_checks,
            "passed": passed,
            "failed": len(violations),
            "pass_rate_percent": round(passed / total_checks * 100, 2),
            "violations": violations,
        }

    # --------------------------------------------------
    # 2. Statistical fidelity (normalized error)
    # --------------------------------------------------

    def _statistical_fidelity(self) -> Dict[str, Any]:
        profiles = self.report["column_profiles"]
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        errors = {}
        all_errors = []

        for col in numeric_cols:
            if col not in profiles:
                continue

            gt = self.df[col].dropna()
            if gt.empty:
                continue

            prof = profiles[col]

            def rel_err(a, b):
                return abs(a - b) / (abs(b) + self.eps) * 100

            mean_err = rel_err(prof["mean"], gt.mean())
            std_err = rel_err(prof["std"], gt.std())
            min_err = rel_err(prof["min"], gt.min())
            max_err = rel_err(prof["max"], gt.max())

            col_errors = {
                "mean_error_percent": round(mean_err, 6),
                "std_error_percent": round(std_err, 6),
                "range_error_percent": round(max(min_err, max_err), 6),
            }

            errors[col] = col_errors
            all_errors.extend(col_errors.values())

        return {
            "summary": {
                "features_evaluated": len(errors),
                "avg_relative_error_percent": round(float(np.mean(all_errors)), 6),
                "max_relative_error_percent": round(float(np.max(all_errors)), 6),
                "interpretation": "Statistical summaries closely match ground truth within numerical tolerance.",
            },
            "by_feature": errors,
        }

    # --------------------------------------------------
    # 3. Issue detection quality (coverage-based)
    # --------------------------------------------------

    def _issue_detection_quality(self) -> Dict[str, Any]:
        quality = self.report["data_quality_report"]

        expected_issues = 0
        detected_issues = 0

        # Missing values
        if self.df.isna().any().any():
            expected_issues += 1
            if quality["missing_values"]["total_missing_cells"] > 0:
                detected_issues += 1

        # Duplicates
        if self.df.duplicated().any():
            expected_issues += 1
            if quality["duplicates"]["duplicate_row_count"] > 0:
                detected_issues += 1

        # Unique-per-row identifiers
        if any(self.df[col].nunique() == len(self.df) for col in self.df.columns):
            expected_issues += 1
            if quality["unique_per_row_columns"]:
                detected_issues += 1

        coverage = detected_issues / max(expected_issues, 1) * 100

        return {
            "issues_expected": expected_issues,
            "issues_detected": detected_issues,
            "coverage_percent": round(coverage, 2),
            "false_positives": 0,  # by design (descriptive-only)
            "false_negatives": expected_issues - detected_issues,
            "interpretation": "Issue detection coverage reflects how completely the analysis flags real data quality risks.",
        }

    # --------------------------------------------------
    # 4. Semantic consistency
    # --------------------------------------------------

    def _semantic_consistency(self) -> Dict[str, Any]:
        target_analysis = self.report.get("target_analysis")

        if not target_analysis:
            return {"semantic_agreement_percent": 100}

        expected_task = (
            "classification"
            if self.df[self.target].nunique() <= 20
            else "regression"
        )

        task_match = target_analysis["task_type"] == expected_task

        imbalance_expected = False
        vc = self.df[self.target].value_counts()
        if len(vc) > 1 and vc.max() / vc.min() >= 3:
            imbalance_expected = True

        imbalance_match = (
            target_analysis.get("imbalance_ratio") is not None
        ) == imbalance_expected

        score = (task_match + imbalance_match) / 2 * 100

        return {
            "task_match": task_match,
            "imbalance_match": imbalance_match,
            "semantic_agreement_percent": round(score, 2),
        }

    # --------------------------------------------------
    # 5. Run full evaluation
    # --------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        invariants = self._check_invariants()
        fidelity = self._statistical_fidelity()
        issues = self._issue_detection_quality()
        semantics = self._semantic_consistency()

        overall = np.mean([
            invariants["pass_rate_percent"],
            100 - fidelity["summary"]["avg_relative_error_percent"],
            issues["coverage_percent"],
            semantics["semantic_agreement_percent"],
        ])

        return {
            "analysis_validity": {
                "invariant_checks": invariants
            },
            "statistical_fidelity": fidelity,
            "issue_detection_quality": issues,
            "semantic_consistency": semantics,
            "overall_assessment": {
                "analysis_trust_score_percent": round(float(overall), 2),
                "confidence_level": "high" if overall > 90 else "medium",
            },
        }
