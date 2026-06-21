"""Interactive, tool-driven standalone preprocessing agent."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agents.dynamic.preprocessing_agent.graph import build_preprocessing_graph
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from agents.dynamic.preprocessing_agent.tool_runner import invoke_tool
from tools.pipeline_state import empty_state, merge_state


TOOL_IMPORTS = {
    "preprocessing_inspection": (
        "tools.preprocessing_inspection",
        "preprocessing_inspection",
    ),
    "preprocessing_plan": ("tools.preprocessing_plan", "preprocessing_plan"),
    "preprocessing_split": ("tools.preprocessing_split", "preprocessing_split"),
    "preprocessing_missing_values": (
        "tools.preprocessing_missing_values",
        "preprocessing_missing_values",
    ),
    "preprocessing_outliers": (
        "tools.preprocessing_outliers",
        "preprocessing_outliers",
    ),
    "preprocessing_encoding": (
        "tools.preprocessing_encoding",
        "preprocessing_encoding",
    ),
    "preprocessing_scaling": (
        "tools.preprocessing_scaling",
        "preprocessing_scaling",
    ),
    "preprocessing_normalization": (
        "tools.preprocessing_normalization",
        "preprocessing_normalization",
    ),
    "preprocessing_balancing": (
        "tools.preprocessing_balancing",
        "preprocessing_balancing",
    ),
    "preprocessing_validation": (
        "tools.preprocessing_validation",
        "preprocessing_validation",
    ),
    "feature_engineering_execution": (
        "tools.feature_engineering_execution",
        "feature_engineering_execution",
    ),
}


class PreprocessingAgent:
    """Plan and execute preprocessing through separate registered tools."""

    def __init__(self, logger: Any, llm: Any, registry: Any):
        self.logger = logger
        self.llm = llm
        self.registry = registry
        self._register_tools()

    def _register_tools(self) -> None:
        import importlib

        for tool_name, (module_name, attribute_name) in TOOL_IMPORTS.items():
            if self.registry.get(tool_name) is None:
                module = importlib.import_module(module_name)
                self.registry.register(tool_name, getattr(module, attribute_name))

    def prepare_plan(
        self,
        *,
        data_path: str,
        prompt: str,
        target_column: str,
        feature_top_k: int = 4,
        output_folder: str | None = None,
        use_llm: bool = True,
        pipeline_state: dict | None = None,
    ) -> dict:
        """Inspect the dataset and generate a preprocessing plan."""
        output_folder = output_folder or str(
            Path("Output") / "Preprocessing" / Path(data_path).stem
        )
        state = pipeline_state or empty_state(data_path, prompt)
        state = merge_state(
            state,
            {
                "data_path": data_path,
                "prompt": prompt,
                "target_column": target_column,
                "status": "running",
            },
        )

        inspect_result, state = invoke_tool(
            self.registry.get("preprocessing_inspection"),
            task="Inspect dataset for preprocessing",
            tool_input={
                "target_column": target_column,
                "output_folder": output_folder,
            },
            prompt=prompt,
            data_path=data_path,
            llm=self.llm,
            pipeline_state=state,
        )
        if inspect_result.get("status") == "error":
            return state

        plan_result, state = invoke_tool(
            self.registry.get("preprocessing_plan"),
            task="Build a preprocessing plan",
            tool_input={
                "evidence": state.get("preprocessing_evidence"),
                "feature_top_k": feature_top_k,
                "output_folder": output_folder,
                "use_llm": use_llm,
            },
            prompt=prompt,
            data_path=data_path,
            llm=self.llm,
            pipeline_state=state,
        )
        if plan_result.get("status") == "error":
            llm_error = plan_result.get(
                "error", "Unknown LLM planning error"
            )
            self.logger.warning(
                f"{llm_error}. Retrying with deterministic planning."
            )
            state.pop("error", None)
            state["status"] = "running"
            plan_result, state = invoke_tool(
                self.registry.get("preprocessing_plan"),
                task="Build a deterministic preprocessing plan",
                tool_input={
                    "evidence": state.get("preprocessing_evidence"),
                    "feature_top_k": feature_top_k,
                    "output_folder": output_folder,
                    "use_llm": False,
                },
                prompt=prompt,
                data_path=data_path,
                llm=self.llm,
                pipeline_state=state,
            )
            if plan_result.get("status") == "success":
                plan = state.get("preprocessing_plan", {})
                warnings = list(plan.get("warnings") or [])
                warnings.append(
                    f"LLM planning failed, so a deterministic plan was used: {llm_error}"
                )
                plan["warnings"] = warnings
                state["preprocessing_plan"] = plan
        return state

    def run(
        self,
        data_path: str,
        prompt: str,
        pipeline_state: dict | None = None,
        *,
        task: str = "Execute preprocessing plan",
        target_column: str = "",
        test_size: float = 0.2,
        random_state: int = 42,
        use_llm: bool = True,
        feature_top_k: int = 4,
        output_folder: str | None = None,
        feature_engineering_input: dict | None = None,
    ) -> dict:
        """Execute only this agent's preprocessing and feature-engineering work."""
        output_folder = output_folder or str(
            Path("Output") / "Preprocessing" / Path(data_path).stem
        )
        state = pipeline_state
        if not state or not state.get("preprocessing_plan"):
            state = self.prepare_plan(
                data_path=data_path,
                prompt=prompt,
                target_column=target_column,
                feature_top_k=feature_top_k,
                output_folder=output_folder,
                use_llm=use_llm,
                pipeline_state=state,
            )
        if state.get("status") == "error":
            return state

        config = {
            "target_column": target_column,
            "test_size": test_size,
            "random_state": random_state,
            "use_llm": use_llm,
            "feature_top_k": feature_top_k,
            "feature_engineering_input": feature_engineering_input or {},
            "output_folder": output_folder,
        }
        graph = build_preprocessing_graph(self.llm, self.registry, config)
        initial: PreprocessingAgentState = {
            "data_path": data_path,
            "prompt": prompt,
            "task": task,
            "pipeline_state": state,
            "step": "preprocessing_agent_start",
        }

        self.logger.info("\n" + "=" * 60)
        self.logger.info("STANDALONE DYNAMIC PREPROCESSING AGENT")
        self.logger.info("=" * 60)
        final_state: PreprocessingAgentState = graph.invoke(initial)
        result = final_state.get("pipeline_state") or state
        if final_state.get("error"):
            self.logger.warning(final_state["error"])
        return result


def _print_columns(columns: list[str]) -> None:
    print("\nAvailable columns:")
    for index, column in enumerate(columns, start=1):
        print(f"  {index:>2}. {column}")


def _choose_target(columns: list[str], default_target: str) -> str:
    _print_columns(columns)
    default_number = (
        columns.index(default_target) + 1 if default_target in columns else 1
    )
    while True:
        answer = input(
            f"\nChoose target by number or name [{default_target or columns[default_number - 1]}]: "
        ).strip()
        if not answer:
            return default_target if default_target in columns else columns[default_number - 1]
        if answer.isdigit() and 1 <= int(answer) <= len(columns):
            return columns[int(answer) - 1]
        if answer in columns:
            return answer
        print("Invalid target. Enter one of the displayed numbers or exact names.")


def _print_plan(plan: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("PROPOSED PREPROCESSING PLAN")
    print("=" * 70)
    print(f"Summary: {plan.get('summary', '')}")
    print(f"Duplicates: {plan.get('duplicates', 'keep')}")
    for column, decision in plan.get("columns", {}).items():
        if decision.get("drop"):
            action = "DROP"
        else:
            action = (
                f"type={decision.get('type')} | missing={decision.get('missing')} | "
                f"outlier={decision.get('outlier')} | encoding={decision.get('encoding')}"
            )
        override = (
            f" | USER OVERRIDE={decision['user_override']}"
            if decision.get("user_override")
            else ""
        )
        print(f"- {column}: {action}{override}")
    print(f"Scaling: {plan.get('scaling', {}).get('method', 'none')}")
    print(f"Normalization: {plan.get('normalization', {}).get('method', 'none')}")
    print(f"Balancing: {plan.get('balancing', {}).get('method', 'none')}")
    feature_plan = plan.get("feature_engineering", {})
    if feature_plan.get("enabled", True):
        print(
            "Feature engineering: run | "
            f"keep top {feature_plan.get('top_k', 4)} from "
            f"{feature_plan.get('max_candidates', 12)} candidates"
        )
    else:
        print("Feature engineering: disabled by user (top_k=0)")
    for warning in plan.get("warnings", []):
        print(f"WARNING: {warning}")
    print("=" * 70)


def _interactive_main() -> int:
    import os

    import pandas as pd
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.utils.logger import Logger
    from tools.registry import ToolRegistry

    load_dotenv(_project_root / ".env")
    default_dataset = _project_root / "uploads" / "Titanic-Dataset.csv"
    default_target = "Survived"

    print("\nStandalone Dynamic Preprocessing Agent")
    print("Press Enter to accept a value shown in brackets.\n")
    chosen_path = input(f"Dataset CSV path [{default_dataset}]: ").strip()
    data_path = Path(chosen_path or default_dataset).expanduser().resolve()
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return 1

    columns = pd.read_csv(data_path, nrows=5).columns.tolist()
    target_column = _choose_target(columns, default_target)
    prompt = input(
        "\nWhat do you want to do in preprocessing?\n> "
    ).strip()
    if not prompt:
        prompt = (
            "Prepare the data safely for modeling. Handle missing values and "
            "outliers, encode categorical columns, scale when useful, preserve "
            "important columns, and balance the target only if needed."
        )
    top_k_raw = input(
        "\nHow many generated feature-engineering columns should be retained? [4]: "
    ).strip()
    try:
        feature_top_k = max(0, min(20, int(top_k_raw or "4")))
    except ValueError:
        feature_top_k = 4
        print("Invalid number; using 4.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY is missing from the environment/.env file.")
        return 1
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )
    agent = PreprocessingAgent(Logger(), llm, ToolRegistry())
    output_folder = str(
        _project_root / "Output" / "Preprocessing" / data_path.stem
    )

    plan_state = agent.prepare_plan(
        data_path=str(data_path),
        prompt=prompt,
        target_column=target_column,
        feature_top_k=feature_top_k,
        output_folder=output_folder,
    )
    if plan_state.get("status") == "error":
        print(plan_state.get("error"))
        return 1

    while True:
        _print_plan(plan_state["preprocessing_plan"])
        answer = input(
            "\nType 'approve', 'cancel', or describe changes to the plan:\n> "
        ).strip()
        if answer.casefold() in {"approve", "approved", "yes", "y"}:
            break
        if answer.casefold() in {"cancel", "no", "n", "quit", "exit"}:
            print("Cancelled. No preprocessing was executed.")
            return 0
        prompt = f"{prompt}\nPlan revision requested by user: {answer}"
        plan_state = agent.prepare_plan(
            data_path=str(data_path),
            prompt=prompt,
            target_column=target_column,
            feature_top_k=feature_top_k,
            output_folder=output_folder,
            pipeline_state=plan_state,
        )
        if plan_state.get("status") == "error":
            print(
                plan_state.get(
                    "error",
                    "The revised plan could not be generated. Please try another revision.",
                )
            )
            continue

    result = agent.run(
        data_path=str(data_path),
        prompt=prompt,
        pipeline_state=plan_state,
        target_column=target_column,
        feature_top_k=feature_top_k,
        output_folder=output_folder,
    )
    print("\n" + "=" * 70)
    print("PREPROCESSING AGENT RESULT")
    print("=" * 70)
    print(f"Status: {result.get('status')}")
    print(f"Modeling ready: {result.get('modeling_ready')}")
    for blocker in result.get("modeling_blockers", []):
        print(f"BLOCKER: {blocker}")
    output = result.get("preprocessing_output", {})
    feature_output = result.get("feature_engineering_output", {})
    print(f"X_train: {output.get('X_train_path')}")
    print(f"X_test: {output.get('X_test_path')}")
    print(f"Readiness report: {output.get('readiness_path')}")
    print(f"Engineered train: {feature_output.get('X_train_engineered_path')}")
    print(f"Feature report: {feature_output.get('feature_report_path')}")
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(_interactive_main())
