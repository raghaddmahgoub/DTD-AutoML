"""
run_datasets.py
===============
Runs the full DTDPipeline on 5 benchmark datasets in sequence:
  1. Iris                    (classification)
  2. Loan Prediction         (classification)
  3. Wine                    (classification)
  4. Diabetes                (regression)
  5. California Housing Prices (regression)

All datasets are found under assets/data/Datasets/.

Usage:
    python run_datasets.py
"""

import os
import sys
import time
import traceback
import tempfile
import shutil

# ── Make sure project root is on the import path ─────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestrator import DTDPipeline  # noqa: E402  (after path fix)

# ── Dataset registry ──────────────────────────────────────────────────────────
CLS = os.path.join("assets", "data", "Datasets", "Classification Datasets")
REG = os.path.join("assets", "data", "Datasets", "Regression Datasets")

DATASETS = [
    {
        "name":          "Iris",
        "raw_path":      os.path.join(CLS, "Iris.csv"),
        "target_column": "Species",
        "task_type":     "classification",
    },
    {
        "name":          "Loan Prediction",
        "raw_path":      os.path.join(CLS, "Loan Prediction.csv"),
        "target_column": "Loan_Status",
        "task_type":     "classification",
    },
    {
        "name":          "Wine",
        "raw_path":      os.path.join(CLS, "wine.csv"),
        "target_column": "target",
        "task_type":     "classification",
    },
    {
        "name":          "Diabetes",
        "raw_path":      os.path.join(REG, "diabetes.csv"),
        "target_column": "target",
        "task_type":     "regression",
    },
    {
        "name":          "California Housing Prices",
        "raw_path":      os.path.join(REG, "California Housing Prices.csv"),
        "target_column": "median_house_value",
        "task_type":     "regression",
    },
]

# ── Helper: prepare a dataset path ────────────────────────────────────────────

def prepare_dataset_path(ds: dict, tmp_dir: str) -> str:
    """Return the path to feed to the pipeline (no special handling needed)."""
    return ds["raw_path"]


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    summary_rows = []
    tmp_dir = tempfile.mkdtemp(prefix="dtd_run_")

    print("\n" + "=" * 70)
    print("  DTD Pipeline — Multi-Dataset Run")
    print("=" * 70)
    print(f"  Datasets to run : {len(DATASETS)}")
    print(f"  Temp workspace  : {tmp_dir}")
    print("=" * 70 + "\n")

    for idx, ds in enumerate(DATASETS, start=1):
        print(f"\n{'─' * 70}")
        print(f"  [{idx}/{len(DATASETS)}]  {ds['name'].upper()}")
        print(f"{'─' * 70}")

        t0 = time.time()
        try:
            effective_path = prepare_dataset_path(ds, tmp_dir)

            pipeline = DTDPipeline()
            inputs = {
                "data_path":     effective_path,
                "target_column": ds["target_column"],
                # Pre-set task_type so preprocessing summary is cross-checked.
                # The preprocessing stage will confirm / override this.
                "task_type":     ds["task_type"],
            }

            result = pipeline.workflow.invoke(inputs)

            elapsed = time.time() - t0
            error   = result.get("error")

            if error:
                status      = "⚠️  PARTIAL"
                best_model  = "N/A"
                best_score  = "N/A"
                print(f"\n  ⚠️  Pipeline completed with error: {error}")
            else:
                status      = "✅  OK"
                fm          = result.get("final_metrics") or {}
                best_model  = fm.get("best_model", "N/A")
                raw_score   = fm.get("best_score")
                best_score  = f"{raw_score:.4f}" if isinstance(raw_score, float) else str(raw_score)

            summary_rows.append({
                "dataset":     ds["name"],
                "task":        ds["task_type"],
                "status":      status,
                "best_model":  best_model,
                "best_score":  best_score,
                "elapsed_s":   f"{elapsed:.1f}",
            })

        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            print(f"\n  ❌  Fatal exception for {ds['name']}:")
            traceback.print_exc()
            summary_rows.append({
                "dataset":     ds["name"],
                "task":        ds["task_type"],
                "status":      "❌  FATAL",
                "best_model":  "N/A",
                "best_score":  "N/A",
                "elapsed_s":   f"{elapsed:.1f}",
            })

    # ── Clean up temp dir ─────────────────────────────────────────────────────
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    # ── Print final summary table ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    header = f"{'Dataset':<22} {'Task':<16} {'Status':<14} {'Best Model':<30} {'Score':<10} {'Time(s)'}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['dataset']:<22} {row['task']:<16} {row['status']:<14} "
            f"{row['best_model']:<30} {row['best_score']:<10} {row['elapsed_s']}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
