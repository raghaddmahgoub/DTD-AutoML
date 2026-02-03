"""
Pipeline the agent lives in:
    AnalysisAgent → PreprocessingAgent → EDAAgent (clean) → AutoMLAgent
                                    ↑
                        EDAAgent (raw) ─┘

    • raw  dataset  →  EDA (run_type="raw")   →  preprocessing_context.json
    • clean dataset →  EDA (run_type="clean") →  automl_context.json

Both runs also produce a self-contained HTML report with plots.


All outputs land in ./Output/  (JSON + HTML).  Plan-stage copies go
to ./Plan/.
──────────────────────────────────────────────────────────────────────
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
from eda_agent2 import EDAAgent  # local import — same directory

# ===========================================================================
# CONFIG  
# ===========================================================================
# TARGET_COLUMN = "Species"
TARGET_COLUMN = "Survived"
# TARGET_COLUMN = "Life expectancy "
PLAN_DIR      = "Plan"
OUTPUT_DIR    = "Output"

RAW_CSV   = '../../assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv'
# CLEAN_CSV = '../../assets/data/Datasets/Classification Datasets/Iris.csv'
CLEAN_CSV   = '../../assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv'
# CLEAN_CSV = '../../assets/data/Datasets/Regression Datasets/Life Expectancy Data.csv'
# df_name = os.path.splitext(os.path.basename(RAW_CSV))[0] 
df_name = os.path.splitext(os.path.basename(CLEAN_CSV))[0] 


# ===========================================================================
# Shared pipeline runner
# ===========================================================================


def run_pipeline(
    df: pd.DataFrame,
    run_type: str,
    df_name: str,
) -> dict:
    """
    Instantiate the agent, run analysis, export all artefacts.
    Returns the export result dict so the caller can log paths.
    """
    separator = "=" * 60

    print(f"\n{separator}")
    print(f"  EDA PIPELINE  —  run_type = '{run_type}'  |  name = '{df_name}'")
    print(separator)

    # ── 1. Show what we're working with ──────────────────────────────
    print(f"\n  📂  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  📂  Columns        : {list(df.columns)}")
    print(f"  📂  Target         : {TARGET_COLUMN}")
    print(f"  📂  Missing values : {int(df.isna().sum().sum()):,} cells\n")

    # ── 2. Instantiate & run ─────────────────────────────────────────
    print("  ⏳  Running EDA analysis …")
    agent = EDAAgent(
        df=df,
        target_column=TARGET_COLUMN,
        df_name=df_name,
    )
    report = agent.run(run_type=run_type)

    # ── 3. Print key findings inline ─────────────────────────────────
    target_info = report.get("target_analysis") or {}
    print(f"  ✅  Task type      : {target_info.get('task_type', 'unknown')}")
    if target_info.get("task_type") == "classification":
        print(f"  ✅  Classes        : {target_info.get('n_classes')}")
        print(f"  ✅  Imbalance ratio: {target_info.get('imbalance_ratio')}")
    else:
        print(f"  ✅  Target mean    : {target_info.get('mean')}")
        print(f"  ✅  Target std     : {target_info.get('std')}")

    warnings = report.get("eda_warnings", [])
    if warnings:
        print(f"\n  ⚠️   {len(warnings)} warning(s) detected:")
        for w in warnings:
            cols_note = ""
            if "columns" in w:
                cols_note = f"  → {w['columns']}"
            print(f"       • {w['message']}{cols_note}")
    else:
        print("\n  ✅  No warnings.")

    # ── 4. Export (routes automatically based on run_type) ───────────
    print(f"\n  ⏳  Exporting artefacts …")
    result = agent.export(plan_dir=PLAN_DIR, output_dir=OUTPUT_DIR)
    print("  ✅  Export complete.\n")

    return result


# ===========================================================================
# Main
# ===========================================================================

import sys

def main():
    # Usage: python test.py raw OR python test.py clean
    # mode = "raw"
    mode = "clean"

    if mode == "raw":
        print("🚀 [STAGE 1] Running RAW Data Analysis...")
        raw_df = pd.read_csv(RAW_CSV)
        # Generates preprocessing_context.json for the next agent
        run_pipeline(raw_df, run_type="raw", df_name=f"{df_name}_raw")
        
    elif mode == "clean":
        print("🚀 [STAGE 2] Running CLEAN Data Analysis...")
        # This assumes the Preprocessing Agent has already saved the new file
        if not Path(CLEAN_CSV).exists():
            print(f"❌ Error: {CLEAN_CSV} not found. Run preprocessing first!")
            return
        clean_df = pd.read_csv(CLEAN_CSV)
        # Generates automl_context.json for the AutoML agent
        run_pipeline(clean_df, run_type="clean", df_name=f"{df_name}_clean")

if __name__ == "__main__":
    main()