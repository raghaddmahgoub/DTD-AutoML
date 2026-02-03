# import pandas as pd
# from eda_agent import EDAAgent
# import os

# filename = "Titanic-Dataset.csv" 
# df_name = os.path.splitext(os.path.basename(filename))[0] 
# df = pd.read_csv(filename)

# eda = EDAAgent(
#     df=df,
#     target_column="Survived",
#     df_name=df_name
# )

# report = eda.run(run_type="raw")
# eda.generate_preprocessing_context()


# # Inspect outputs
# print("EDA REPORT KEYS:")
# print(report.keys())

# print("\nDATASET SUMMARY:")
# print(report["dataset_summary"])

# print("\nTARGET ANALYSIS:")
# print(report["target_analysis"])

# print("\nCOLUMN PROFILES (year):")
# print(report["column_profiles"])

# print("\nDATA QUALITY:")
# print(report["data_quality_report"])

# print("\nRELATIONSHIP INSIGHTS:")
# print(report["relationship_insights"])

# print("\nEDA WARNINGS:")
# print(report["eda_warnings"])

from typing import Any, Dict, Optional
import pandas as pd
from eda_agent import EDAAgent
from eda_agent import TargetInferenceAgent
import os

def run_eda_with_target_resolution(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    df_name: str = "dataset",
    run_type: str = "raw",
) -> Dict[str, Any]:
    """
    Resolve target column (user-provided or inferred) and execute EDA.
    """

    target_metadata = {
        "source": "user",
        "confidence": 1.0,
        "alternatives": [],
    }

    resolved_target = target_column

    # --- Infer target if not provided ---
    if target_column is None:
        inference = TargetInferenceAgent(df).run()
        resolved_target = inference["inferred_target"]
        target_metadata = {
            "source": "inferred",
            "confidence": inference["confidence"],
            "alternatives": inference["alternatives"],
        }

    # --- Run EDA ---
    eda = EDAAgent(
        df=df,
        target_column=resolved_target,
        df_name=df_name,
    )
    report = eda.run(run_type=run_type)

    # --- Attach target provenance (CRITICAL) ---
    report["target_metadata"] = {
        "target_column": resolved_target,
        **target_metadata,
    }

    return report


filename = 'assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv'
df_name = os.path.splitext(os.path.basename(filename))[0]
df = pd.read_csv(filename)
report = run_eda_with_target_resolution(
    df=df,
    target_column=None,  # ← simulate user not knowing target
    df_name=df_name,
    run_type="raw",
)
# Inspect outputs
print("EDA REPORT KEYS:")
print(report.keys())

print("\nDATASET SUMMARY:")
print(report["dataset_summary"])

print("\nTARGET ANALYSIS:")
print(report["target_analysis"])

print("\nCOLUMN PROFILES (year):")
print(report["column_profiles"])

print("\nDATA QUALITY:")
print(report["data_quality_report"])

print("\nRELATIONSHIP INSIGHTS:")
print(report["relationship_insights"])

print("\nEDA WARNINGS:")
print(report["eda_warnings"])

print("\nTARGET METADATA:")
print(report["target_metadata"])
