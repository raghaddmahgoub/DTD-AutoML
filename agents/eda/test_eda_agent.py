import pandas as pd
from eda_agent import EDAAgent


df = pd.read_csv("Titanic-Dataset.csv")


eda = EDAAgent(
    df=df,
    target_column="Survived"
)

report = eda.run(run_type="raw")
eda.generate_preprocessing_context()


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
