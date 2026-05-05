import time
import json
import os
from orchestrator import DTDPipeline

# Define datasets to benchmark
DATASETS = [
    {
        "name": "titanic",
        "path": "assets/data/Classification Datasets/Titanic-Dataset.csv",
        "target": "Survived",
    },
    # Add more datasets later
]

OUTPUT_PATH = "benchmark/results/benchmark_results.json"


def run_pipeline(data_path, target):
    pipeline = DTDPipeline()

    state = {
        "data_path": data_path,
        "target_column": target
    }

    result = pipeline.workflow.invoke(state)
    return result


def run_benchmark():
    results = []

    for dataset in DATASETS:
        print(f"\n🚀 Running benchmark on {dataset['name']}...")

        start_time = time.time()

        try:
            output = run_pipeline(dataset["path"], dataset["target"])
            end_time = time.time()

            metrics = output.get("final_metrics", {}) or {}

            results.append({
                "dataset": dataset["name"],
                "pipeline_metrics": metrics,
                "execution_time_sec": round(end_time - start_time, 2),
                "status": "success" if not output.get("error") else "failed",
                "error": output.get("error")
            })

        except Exception as e:
            results.append({
                "dataset": dataset["name"],
                "status": "failed",
                "error": str(e)
            })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ Benchmark completed. Results saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    run_benchmark()