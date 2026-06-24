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

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DTD-Dynamic-Benchmark"

PROMPTS = [
    "Run full AutoML pipeline including preprocessing, model training and evaluation",

    # "Run full AutoML pipeline including preprocessing, model training and evaluation. Prioritize interpretability.",

    "Run full AutoML pipeline including preprocessing, model training and evaluation. Prioritize predictive accuracy.",

    "Run full AutoML pipeline including preprocessing, model training and evaluation. Use lightweight models suitable for deployment.",

    # "Run full AutoML pipeline including preprocessing, model training and evaluation. Avoid feature selection."
]

DATASETS = [
    {
        "path": "benchmark/datasets/Housing.csv",
        "target": "price",
    },
    {
        "path": "benchmark/datasets/diabetes130/diabetic_data.csv",
        "target": "diabetesMed",
    },
]


def benchmark():
    results = []
    agent = ControllerAgent()
    
    for ds in DATASETS:
        for prompt in PROMPTS:

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
                    debug_file = (debug_dir /f"{Path(ds['path']).stem}_{len(results)}.json")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2, default=str)

                results.append(result)
                print(f"✓ Completed in {runtime}s")

            except Exception as e:
                runtime = round(time.time() - start, 2)
                print(f"✗ Failed: {e}")
                results.append(
                    {
                        "dataset": ds["path"],
                        "target_column": ds["target"],
                        "prompt": prompt,
                        "success": False,
                        "runtime_seconds": runtime,
                        "error": str(e),
                    }
                )
    output_dir = Path("benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dynamic_pipeline_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n")
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"Total runs: {len(results)}")

if __name__ == "__main__":
    benchmark()