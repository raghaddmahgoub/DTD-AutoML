"""
Test the dynamic tools pipeline (preprocess → plan_training → train_* → evaluate).

Examples (from repo root):

  # Full pipeline: PreprocessingAgent → ModelAgent
  python src/test_tools_pipeline.py --mode model_agent --data iris --no-prompts

  # Skip preprocessing if splits already exist under Output/Preprocessing/
  python src/test_tools_pipeline.py --mode model_agent --data iris --skip-preprocess --no-prompts

Requires: GOOGLE_API_KEY in .env for LLM steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.dynamic.controller_agent.controller_agent import ControllerAgent
from agents.dynamic.model_agent import ModelAgent
from agents.dynamic.preprocessing_agent.preprocessing_agent import PreprocessingAgent
from src.utils.logger import Logger
from tools.registry import ToolRegistry
from tools.preprocessing_execution import preprocessing_execution
from tools.plan_training import plan_training
from tools.train_simple import train_simple
from tools.train_simple_optuna import train_simple_optuna
from tools.train_autogluon import train_autogluon
from tools.evaluate import evaluate
from tools.pipeline_state import empty_state, merge_state
from tools.training_common import resolve_problem_type


DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "assets/data/Datasets/Classification Datasets/Iris.csv",
    PROJECT_ROOT / "assets/data/Datasets/Classification Datasets/wine.csv",
    PROJECT_ROOT / "assets/data/Datasets/Classification Datasets/Loan Prediction.csv",
    PROJECT_ROOT / "assets/data/Classification Datasets/Titanic-Dataset.csv",
    PROJECT_ROOT / "assets/data/Classification Datasets/Iris.csv",
    PROJECT_ROOT / "Datasets/Titanic-Dataset.csv",
]

IRIS_FALLBACK = PROJECT_ROOT / "output" / "test_pipeline" / "iris_sample.csv"


def ensure_iris_fallback() -> str:
    IRIS_FALLBACK.parent.mkdir(parents=True, exist_ok=True)
    if not IRIS_FALLBACK.exists():
        from sklearn.datasets import load_iris
        import pandas as pd

        iris = load_iris(as_frame=True)
        df = iris.frame.rename(columns={"target": "species"})
        df.to_csv(IRIS_FALLBACK, index=False)
        print(f"[test] Created fallback dataset → {IRIS_FALLBACK}")
    return str(IRIS_FALLBACK)


def _alternate_dataset_paths(path: Path) -> list[Path]:
    """Try common repo layouts when the user path does not exist."""
    name = path.name
    bases = [
        PROJECT_ROOT / "assets/data/Datasets/Classification Datasets" / name,
        PROJECT_ROOT / "assets/data/Datasets/Regression Datasets" / name,
        PROJECT_ROOT / "assets/data/Classification Datasets" / name,
        PROJECT_ROOT / "assets/data" / name,
    ]
    return [p for p in bases if p != path and p.exists()]


def resolve_data_path(user_path: str | None) -> str:
    if user_path:
        token = user_path.strip().lower()
        if token in {"iris", "iris.csv", "sklearn:iris"}:
            return ensure_iris_fallback()

        p = Path(user_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return str(p)

        for alt in _alternate_dataset_paths(p):
            print(f"[test] Dataset not found at {p}")
            print(f"[test] Using alternate path → {alt}")
            return str(alt)

        if "iris" in p.stem.lower():
            print(f"[test] Dataset not found at {p}")
            print("[test] Iris CSV is not in this repo — using built-in sklearn Iris sample.")
            return ensure_iris_fallback()

        raise FileNotFoundError(
            f"Dataset not found: {p}\n"
            "Tips:\n"
            "  • Run without --data to use the built-in Iris sample\n"
            "  • Or use: --data iris\n"
            "  • Or download datasets to assets/data/Datasets/Classification Datasets/"
        )

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    print("[test] No bundled CSV found in assets/ — using built-in sklearn Iris sample.")
    return ensure_iris_fallback()


def build_registry(*, include_controller_stubs: bool = False) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("preprocessing_execution", preprocessing_execution)
    reg.register("plan_training", plan_training)
    reg.register("train_simple", train_simple)
    reg.register("train_simple_optuna", train_simple_optuna)
    reg.register("train_autogluon", train_autogluon)
    reg.register("evaluate", evaluate)
    if include_controller_stubs:
        from tools.data_understanding import data_understanding

        reg.register("data_understanding", data_understanding)
    return reg


def make_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Add it to .env")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.3,
    )


def invoke_tool(tool, *, task: str, tool_input: dict, prompt: str, data_path: str, llm, state):
    return tool.invoke(
        {
            "task": task,
            "tool_input": tool_input,
            "prompt": prompt,
            "data_path": data_path,
            "llm": llm,
            "state": state,
        }
    )


def print_step(name: str, result: dict, state: dict) -> None:
    print("\n" + "=" * 70)
    print(f"STEP: {name}")
    print("-" * 70)
    print(json.dumps(result, indent=2, default=str)[:4000])
    print("-" * 70)
    plan = state.get("training_plan") or {}
    print(
        f"pipeline step={state.get('step')} | "
        f"plan.approved={plan.get('approved')} | "
        f"train_tool={plan.get('train_tool')}"
    )


def run_model_agent(
    data_path: str,
    prompt: str,
    llm,
    registry: ToolRegistry,
    *,
    approach: str | None,
    target: str | None,
    no_prompts: bool,
    skip_preprocess: bool = False,
) -> dict:
    """PreprocessingAgent then ModelAgent LangGraph: plan → train → evaluate."""
    logger = Logger()
    state = empty_state(data_path, prompt)

    if not skip_preprocess:
        prep_agent = PreprocessingAgent(logger, llm, registry)
        prep_kwargs: dict = {}
        if target:
            prep_kwargs["target_column"] = target
        state = prep_agent.run(data_path, prompt, pipeline_state=state, **prep_kwargs)
        print_step(
            "preprocessing_agent",
            {
                "status": state.get("status"),
                "step": state.get("step"),
                "X_train_path": state.get("X_train_engineered_path") or state.get("X_train_path"),
            },
            state,
        )
        if state.get("error") or state.get("step") in ("preprocessing_failed",):
            return state
    else:
        stem = Path(data_path).stem
        prep_dir = PROJECT_ROOT / "Output" / "Preprocessing" / stem
        if prep_dir.exists():
            paths = {
                "X_train_path": str(prep_dir / "X_train.csv"),
                "X_test_path": str(prep_dir / "X_test.csv"),
                "y_train_path": str(prep_dir / "y_train.csv"),
                "y_test_path": str(prep_dir / "y_test.csv"),
            }
            eng_train = prep_dir / "X_train_engineered.csv"
            eng_test = prep_dir / "X_test_engineered.csv"
            if eng_train.exists() and eng_test.exists():
                paths["X_train_engineered_path"] = str(eng_train)
                paths["X_test_engineered_path"] = str(eng_test)
            state = merge_state(state, paths)

    problem_type = resolve_problem_type(state)
    if problem_type:
        state = merge_state(state, {"problem_type": problem_type})

    agent = ModelAgent(logger, llm, registry)
    kwargs = {
        "ask_before_training": not no_prompts,
        "auto_approve_plan": no_prompts,
    }
    if approach:
        kwargs["training_approach"] = approach
    if target:
        kwargs["target_column"] = target
    if state.get("problem_type"):
        kwargs["problem_type"] = state["problem_type"]

    state = agent.run(data_path, prompt, pipeline_state=state, **kwargs)

    plan = state.get("training_plan") or {}
    print_step(
        "model_agent",
        {
            "status": state.get("status"),
            "step": state.get("step"),
            "train_tool": plan.get("train_tool"),
            "approved": plan.get("approved"),
            "metrics": state.get("model_metrics"),
        },
        state,
    )

    out = PROJECT_ROOT / "output" / "test_pipeline" / "final_state.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\n[test] Final state saved → {out}")
    return state


def run_manual(
    data_path: str,
    prompt: str,
    llm,
    registry: ToolRegistry,
    *,
    approach: str | None,
    target: str | None,
    no_prompts: bool,
) -> dict:
    """Alias for run_model_agent (EDA + ModelAgent LangGraph)."""
    return run_model_agent(
        data_path,
        prompt,
        llm,
        registry,
        approach=approach,
        target=target,
        no_prompts=no_prompts,
    )


def run_single_tool(
    tool_name: str,
    data_path: str,
    prompt: str,
    llm,
    registry: ToolRegistry,
    *,
    approach: str | None,
    no_prompts: bool,
) -> dict:
    tool = registry.get(tool_name)
    if tool is None:
        raise RuntimeError(f"Unknown tool: {tool_name}. Available: {registry.list_tools()}")

    state = empty_state(data_path, prompt)
    tool_input: dict = {}
    if tool_name == "plan_training":
        tool_input = {
            "ask_before_training": not no_prompts,
            "auto_approve_plan": no_prompts,
        }
        if approach:
            tool_input["training_approach"] = approach

    result, state = invoke_tool(
        tool,
        task=f"Test {tool_name}",
        tool_input=tool_input,
        prompt=prompt,
        data_path=data_path,
        llm=llm,
        state=state,
    )
    print_step(tool_name, result, state)
    return state


def run_controller(data_path: str, prompt: str, llm, registry: ToolRegistry) -> dict:
    logger = Logger()
    controller = ControllerAgent(logger, llm, registry)
    return controller.run(data_path, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test dynamic AutoML tools pipeline")
    parser.add_argument(
        "--mode",
        choices=["manual", "model_agent", "controller", "tool"],
        default="model_agent",
        help="model_agent=LangGraph training agent; manual=same; controller=LLM tool loop; tool=single tool",
    )
    parser.add_argument("--data", dest="data_path", default=None, help="Path to CSV dataset")
    parser.add_argument(
        "--approach",
        choices=["1", "2", "3", "simple", "simple_optuna", "autogluon"],
        default=None,
        help="Force training approach (omit to let LLM decide)",
    )
    parser.add_argument("--target", default=None, help="Target column (optional)")
    parser.add_argument(
        "--tool",
        default="plan_training",
        help="Tool name when --mode tool (e.g. plan_training, train_simple_optuna)",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip PreprocessingAgent (use existing splits in pipeline_state / Output/Preprocessing/)",
    )
    parser.add_argument(
        "--no-prompts",
        action="store_true",
        help="Skip interactive input; auto-approve the LLM training plan",
    )
    parser.add_argument(
        "--prompt",
        default="Analyze the dataset and train a model to predict the target.",
    )
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_path)
    llm = make_llm()
    registry = build_registry(include_controller_stubs=args.mode == "controller")

    print(f"[test] mode={args.mode}")
    print(f"[test] data={data_path}")
    print(f"[test] approach={'LLM decides' if not args.approach else args.approach}")
    print(f"[test] prompts={'off (auto-approve)' if args.no_prompts else 'on (interactive)'}")

    if args.mode in ("manual", "model_agent"):
        run_model_agent(
            data_path,
            args.prompt,
            llm,
            registry,
            approach=args.approach,
            target=args.target,
            no_prompts=args.no_prompts,
            skip_preprocess=args.skip_preprocess,
        )
    elif args.mode == "tool":
        run_single_tool(
            args.tool,
            data_path,
            args.prompt,
            llm,
            registry,
            approach=args.approach,
            no_prompts=args.no_prompts,
        )
    else:
        if not args.no_prompts:
            print(
                "[test] controller mode uses plan_training prompts inside the loop. "
                "Use --no-prompts for non-interactive runs."
            )
        run_controller(data_path, args.prompt, llm, registry)


if __name__ == "__main__":
    main()
