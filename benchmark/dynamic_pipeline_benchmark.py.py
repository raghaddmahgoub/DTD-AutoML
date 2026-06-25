import time
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from agents.dynamic.controller_agent.controller_agent import ControllerAgent
load_dotenv()
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DTD-Dynamic-Benchmark"
print("LANGCHAIN_TRACING_V2 =", os.getenv("LANGCHAIN_TRACING_V2"))
print("LANGCHAIN_PROJECT =", os.getenv("LANGCHAIN_PROJECT"))
print("LANGCHAIN_API_KEY exists =", bool(os.getenv("LANGCHAIN_API_KEY")))

# "Run full AutoML pipeline including preprocessing, model training and evaluation. Prioritize interpretability.",
# "Run full AutoML pipeline including preprocessing, model training and evaluation. Use lightweight models suitable for deployment.",
# "Run full AutoML pipeline including preprocessing, model training and evaluation. Avoid feature selection."

REG_PROMPTS = [
    "Run full AutoML pipeline including preprocessing, model training and evaluation, the target column is price",
    "Run full AutoML pipeline including preprocessing, model training and evaluation. Prioritize predictive accuracy, the target column is price.",
]

CLS_PROMPTS = [
    "Run full AutoML pipeline including preprocessing, model training and evaluation, the target column is diabetesMed",
    # "Run full AutoML pipeline including preprocessing, model training and evaluation. Prioritize predictive accuracy, the target column is diabetesMed.",
]

BENCHMARKS = [
    # {
    #     "dataset": {
    #         "path": "benchmark/datasets/Housing.csv",
    #         "target": "price",
    #     },
    #     "prompts": REG_PROMPTS,
    #     "type": "regression",
    # },
    {
        "dataset": {
            "path": "benchmark/datasets/diabetes130/diabetic_data.csv",
            "target": "diabetesMed",
        },
        "prompts": CLS_PROMPTS,
        "type": "classification",
    },
]
def benchmark():
    regression_results = []
    classification_results = []
    agent = ControllerAgent()
    
    for benchmark_cfg in BENCHMARKS:

        ds = benchmark_cfg["dataset"]
        prompts = benchmark_cfg["prompts"]
        benchmark_type = benchmark_cfg["type"]

        print("\n" + "#" * 80)
        print(f"BENCHMARK TYPE: {benchmark_type.upper()}")
        print("#" * 80)

        for prompt in prompts:

            print("\n" + "=" * 80)
            print(f"Dataset: {ds['path']}")
            print(f"Prompt : {prompt}")
            print("=" * 80)

            start = time.time()

            try:
                run_id = (f"{Path(ds['path']).stem}_"f"{prompt[:20].replace(' ','_')}")
                state = agent.run(
                    {
                        "data_path": ds["path"],
                        "target_column": ds["target"],
                        "prompt": prompt,
                        "run_id": run_id,
                    }
                )
                while state.get("__interrupted__"):
                    state = agent.resume(
                        run_id=state["__run_id__"],
                        decision="accept")
                    
                runtime = round(time.time() - start, 2)

                model_metrics = state.get("model_metrics", {})

                result = {
                    "benchmark_type": benchmark_type,
                    "dataset": ds["path"],
                    "target_column": ds["target"],
                    "prompt": prompt,
                    # "raw_state": state,
                    "success": ("__error__" not in state and state.get("task_type") is not None),
                    "error": state.get("__error__"),
                    "runtime_seconds": runtime,
                    "task_type": state.get("task_type"),
                    "selected_model": state.get("trained_model_path"),
                    "model_metrics": model_metrics,
                    "endpoint_url": state.get("endpoint_url"),
                    "intent_flags": state.get("intent_flags"),
                    "kg_generated": bool(
                        state.get("knowledge_graph")
                        or state.get("knowledge_graph_path")
                    ),
                }
                if result["success"] is False:
                    debug_dir = Path("benchmark/debug")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    current_count = len(regression_results) + len(classification_results)
                    debug_file = (
                        debug_dir /
                        f"{benchmark_type}_{Path(ds['path']).stem}_{current_count}.json"
                    )
                    with open(debug_file, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2, default=str)

                if benchmark_type == "regression":
                    regression_results.append(result)
                else:
                    classification_results.append(result)
                print(f"✓ Completed in {runtime}s")

            except Exception as e:
                runtime = round(time.time() - start, 2)
                print(f"✗ Failed: {e}")
                error_result = {
                    "benchmark_type": benchmark_type,
                    "dataset": ds["path"],
                    "target_column": ds["target"],
                    "prompt": prompt,
                    "success": False,
                    "runtime_seconds": runtime,
                    "error": str(e),
                }
                if benchmark_type == "regression":
                    regression_results.append(error_result)
                else:
                    classification_results.append(error_result)
                    
    output_dir = Path("benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "regression_results.json", "w") as f:
        json.dump(regression_results, f, indent=2, default=str)

    with open(output_dir / "classification_results.json", "w") as f:
        json.dump(classification_results, f, indent=2, default=str)

    all_results = regression_results + classification_results

    with open(output_dir / "dynamic_pipeline_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n")
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Regression runs: {len(regression_results)}")
    print(f"Classification runs: {len(classification_results)}")
    print(f"Total runs: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    
if __name__ == "__main__":
    benchmark()