import pandas as pd
from eda_agent import EDAAgent
import os

filename = "Titanic-Dataset.csv" 
df_name = os.path.splitext(os.path.basename(filename))[0] 
df = pd.read_csv(filename)

eda = EDAAgent(
    df=df,
    target_column="Survived",
    df_name=df_name
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
