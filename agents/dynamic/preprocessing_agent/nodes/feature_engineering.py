"""LangGraph node: append selected LLM-generated feature combinations."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from agents.dynamic.preprocessing_agent.tool_runner import invoke_tool


def make_feature_engineering_node(
    llm: Any,
    registry: Any,
    config: dict,
) -> Callable[[PreprocessingAgentState], PreprocessingAgentState]:
    feature_tool = registry.get("feature_engineering_execution")
    if feature_tool is None:
        raise RuntimeError("feature_engineering_execution tool is not registered")

    def feature_engineering_node(
        state: PreprocessingAgentState,
    ) -> PreprocessingAgentState:
        pipeline_state = state.get("pipeline_state") or {}
        tool_input = dict(config.get("feature_engineering_input") or {})
        feature_plan = (
            pipeline_state.get("preprocessing_plan", {})
            .get("feature_engineering", {})
        )
        output_folder = Path(config["output_folder"])
        if not feature_plan.get("enabled", True) or int(
            feature_plan.get("top_k", config.get("feature_top_k", 4))
        ) <= 0:
            work_folder = output_folder / ".preprocessing_work"
            if work_folder.exists():
                shutil.rmtree(work_folder)
            pipeline_state["feature_engineering_status"] = "skipped_by_user"
            pipeline_state["step"] = "preprocessing_complete"
            return {
                **state,
                "pipeline_state": pipeline_state,
                "last_tool": "feature_engineering_execution",
                "last_result": {
                    "status": "skipped",
                    "message": "Feature engineering disabled because top_k is 0.",
                },
                "step": "preprocessing_complete",
            }
        tool_input.setdefault(
            "top_k",
            feature_plan.get("top_k", config.get("feature_top_k", 4)),
        )
        tool_input.setdefault(
            "max_candidates",
            feature_plan.get("max_candidates", 12),
        )
        tool_input.setdefault("output_folder", config.get("output_folder"))
        tool_input.setdefault("use_llm", config.get("use_llm", True))

        result, pipeline_state = invoke_tool(
            feature_tool,
            task=state.get("task") or "Generate, evaluate, and select new feature combinations",
            tool_input=tool_input,
            prompt=state.get("prompt", ""),
            data_path=state.get("data_path", pipeline_state.get("data_path", "")),
            llm=llm,
            pipeline_state=pipeline_state,
        )

        if result.get("status") == "success":
            feature_output = result.get("feature_engineering_output", {})
            engineered_train = Path(
                feature_output["X_train_engineered_path"]
            )
            engineered_test = Path(
                feature_output["X_test_engineered_path"]
            )
            final_train = output_folder / "X_train.csv"
            final_test = output_folder / "X_test.csv"
            shutil.copyfile(engineered_train, final_train)
            shutil.copyfile(engineered_test, final_test)
            engineered_train.unlink(missing_ok=True)
            engineered_test.unlink(missing_ok=True)

            feature_output["X_train_engineered_path"] = str(final_train)
            feature_output["X_test_engineered_path"] = str(final_test)
            pipeline_state["X_train_path"] = str(final_train)
            pipeline_state["X_test_path"] = str(final_test)
            pipeline_state["X_train_engineered_path"] = str(final_train)
            pipeline_state["X_test_engineered_path"] = str(final_test)
            pipeline_state["feature_engineering_output"] = feature_output

        work_folder = output_folder / ".preprocessing_work"
        if work_folder.exists():
            shutil.rmtree(work_folder)

        out: PreprocessingAgentState = {
            **state,
            "pipeline_state": pipeline_state,
            "last_tool": "feature_engineering_execution",
            "last_result": result,
            "step": pipeline_state.get("step", "feature_engineering_complete"),
        }
        if result.get("status") == "error":
            warning = result.get(
                "error", "feature_engineering_execution could not complete"
            )
            warnings = list(pipeline_state.get("preprocessing_warnings", []))
            warnings.append(warning)
            pipeline_state["preprocessing_warnings"] = warnings
            pipeline_state["feature_engineering_status"] = "failed_non_blocking"
            pipeline_state["status"] = "success"
            pipeline_state["step"] = "preprocessing_complete_with_feature_warning"
            pipeline_state.pop("error", None)
            out["pipeline_state"] = pipeline_state
            out["step"] = pipeline_state["step"]
        return out

    return feature_engineering_node
