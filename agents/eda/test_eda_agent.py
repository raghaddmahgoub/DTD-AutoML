import pandas as pd
from eda_agent import EDAAgent


df = pd.read_csv("car_prices.csv")


eda = EDAAgent(
    df=df,
    target_column="sellingprice"
)

report = eda.run(run_type="raw")
eda.generate_preprocessing_context()


# Inspect outputs
print("EDA REPORT KEYS:")
print(report.keys())

print("\nDATASET SUMMARY:")
print(report["dataset_summary"])

print("\nCOLUMN PROFILES (year):")
print(report["column_profiles"]["year"])

print("\nDATA QUALITY:")
print(report["data_quality_report"])

print("\nRELATIONSHIP INSIGHTS:")
print(report["relationship_insights"])

print("\nEDA WARNINGS:")
print(report["eda_warnings"])
