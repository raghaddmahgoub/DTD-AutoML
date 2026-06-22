"""Preprocessing Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import json
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Any, Optional
from langgraph.types import interrupt

from state.pipeline_state import PipelineState
from tools.shared import get_llm, build_prompt_preprocessing
from src.utils.logger import Logger

# Import fine-grained preprocessing tools from the new package namespace
from tools.preprocessing import (
    preprocessing_inspection,
    preprocessing_plan,
    preprocessing_split,
    preprocessing_missing_values,
    preprocessing_outliers,
    preprocessing_encoding,
    preprocessing_scaling,
    preprocessing_normalization,
    preprocessing_balancing,
    preprocessing_validation,
    PREPROCESSING_TOOLS,  # ordered list for introspection / future bind_tools
)

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Plan and execute preprocessing sequentially using individual tools.
    Avoids using any sub-graphs or monolithic execution scripts.
    """

    def __init__(self, logger_obj: Any = None, llm: Any = None, registry: Any = None):
        self.logger = logger_obj or Logger()
        self.llm = llm or get_llm()
        # registry parameter is accepted for backward compatibility

    def run(
        self,
        data_path: str,
        prompt: str,
        pipeline_state: dict | None = None,
        *,
        task: str = "Execute preprocessing pipeline",
        target_column: str | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        use_llm: bool = True,
        output_folder: str | None = None,
        feedback_context: str = "",
    ) -> dict:
        """Execute the full preprocessing pipeline sequentially using fine-grained tools."""
        from state.pipeline_state import make_initial_state

        if pipeline_state is None:
            pipeline_state = make_initial_state(data_path, prompt)

        if not target_column:
            target_column = pipeline_state.get("target_column")

        if not target_column:
            raise ValueError("target_column must be provided in pipeline_state or as a run argument.")

        pipeline_state["target_column"] = target_column

        task_type = pipeline_state.get("task_type", "unknown")
        if task_type == "unknown" or not task_type:
            # Determine task type directly from sample data
            df_sample = pd.read_csv(data_path, nrows=5)
            y_sample = df_sample[target_column]
            if pd.api.types.is_numeric_dtype(y_sample) and y_sample.nunique() > 20:
                task_type = "regression"
            else:
                task_type = "classification"
        pipeline_state["task_type"] = task_type

        output_folder = output_folder or str(
            Path("Output") / "Preprocessing" / Path(data_path).stem
        )
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Helper to check for errors in intermediate tool results
        def check_error(result: dict, state: dict) -> bool:
            if result.get("status") == "error":
                state["error"] = result.get("error", "Step failed")
                state["status"] = "error"
                return True
            return False

        # 1. Dataset Inspection
        self.logger.info("[PreprocessingAgent] 1/10 Running dataset inspection...")
        res, pipeline_state = preprocessing_inspection.invoke({
            "task": "Inspect dataset for preprocessing",
            "tool_input": {
                "target_column": target_column,
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 2. Build Preprocessing Plan
        self.logger.info("[PreprocessingAgent] 2/10 Generating preprocessing plan...")
        prompts = build_prompt_preprocessing(
            data_path=data_path,
            target_column=target_column,
            task_type=task_type,
            preprocessing_context=pipeline_state.get("preprocessing_context"),
            test_size=test_size,
            feedback_context=feedback_context,
        )

        res, pipeline_state = preprocessing_plan.invoke({
            "task": "Build a preprocessing plan",
            "tool_input": {
                "evidence": pipeline_state.get("preprocessing_evidence"),
                "feature_top_k": 4,
                "output_folder": output_folder,
                "use_llm": use_llm,
            },
            "prompt": prompts.user,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 3. Split Dataset
        self.logger.info("[PreprocessingAgent] 3/10 Creating dataset train/test splits...")
        res, pipeline_state = preprocessing_split.invoke({
            "task": "Prepare data and create the train/test split",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "target_column": target_column,
                "output_folder": output_folder,
                "test_size": test_size,
                "random_state": random_state,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 4. Handle Missing Values
        self.logger.info("[PreprocessingAgent] 4/10 Imputing missing values...")
        res, pipeline_state = preprocessing_missing_values.invoke({
            "task": "Handle missing values",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 5. Handle Outliers
        self.logger.info("[PreprocessingAgent] 5/10 Handling outliers...")
        res, pipeline_state = preprocessing_outliers.invoke({
            "task": "Handle numerical outliers",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 6. Encode Categorical Features
        self.logger.info("[PreprocessingAgent] 6/10 Encoding categorical features...")
        res, pipeline_state = preprocessing_encoding.invoke({
            "task": "Encode categorical features",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 7. Scale Numerical Features
        self.logger.info("[PreprocessingAgent] 7/10 Scaling numerical features...")
        res, pipeline_state = preprocessing_scaling.invoke({
            "task": "Scale numerical features",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 8. Normalize Feature Rows
        self.logger.info("[PreprocessingAgent] 8/10 Normalizing feature rows...")
        res, pipeline_state = preprocessing_normalization.invoke({
            "task": "Normalize feature rows",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 9. Balance Target Variable
        self.logger.info("[PreprocessingAgent] 9/10 Balancing training target...")
        res, pipeline_state = preprocessing_balancing.invoke({
            "task": "Balance the training target",
            "tool_input": {
                "plan": pipeline_state["preprocessing_plan"],
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # 10. Final Model-Readiness Validation
        self.logger.info("[PreprocessingAgent] 10/10 Validating modeling readiness...")
        res, pipeline_state = preprocessing_validation.invoke({
            "task": "Validate modeling readiness",
            "tool_input": {
                "X_train_path": pipeline_state["X_train_path"],
                "X_test_path": pipeline_state["X_test_path"],
                "y_train_path": pipeline_state["y_train_path"],
                "y_test_path": pipeline_state["y_test_path"],
                "output_folder": output_folder,
            },
            "prompt": prompt,
            "data_path": data_path,
            "llm": self.llm,
            "state": pipeline_state,
        })
        if check_error(res, pipeline_state):
            return pipeline_state

        # Rebuild full combined dataset for clean analysis
        try:
            X_train_path = pipeline_state.get("X_train_path")
            X_test_path = pipeline_state.get("X_test_path")
            y_train_path = pipeline_state.get("y_train_path")
            y_test_path = pipeline_state.get("y_test_path")

            if X_train_path and X_test_path and y_train_path and y_test_path:
                X_train = pd.read_csv(X_train_path)
                X_test = pd.read_csv(X_test_path)
                y_train = pd.read_csv(y_train_path).squeeze("columns")
                y_test = pd.read_csv(y_test_path).squeeze("columns")

                train_df = X_train.copy()
                train_df[target_column] = y_train.reset_index(drop=True)

                test_df = X_test.copy()
                test_df[target_column] = y_test.reset_index(drop=True)

                clean_df = pd.concat([train_df, test_df], ignore_index=True)
                clean_data_path = os.path.join(output_folder, "full_preprocessed.csv")
                clean_df.to_csv(clean_data_path, index=False)
                pipeline_state["clean_data_path"] = clean_data_path
                self.logger.info(f"[PreprocessingAgent] Combined clean dataset written to {clean_data_path}")
        except Exception as e:
            self.logger.warning(f"Failed to build clean dataset from splits: {e}")

        # Populate structured UI column actions
        column_actions_frontend = []
        plan_cols = (pipeline_state.get("preprocessing_plan") or {}).get("columns") or {}
        for col in sorted(plan_cols.keys()):
            dec = plan_cols[col]
            col_actions = {
                "column": col,
                "action": "drop" if dec.get("drop") else "transform",
                "reason": dec.get("reason", "policy_decision"),
                "policy_source": "llm_policy" if use_llm else "default_policy",
                "details": {
                    "type": dec.get("type"),
                    "missing": pipeline_state.get("missing_value_actions", {}).get(col, {}).get("method"),
                    "outlier": pipeline_state.get("outlier_actions", {}).get(col, {}).get("method"),
                    "encoding": pipeline_state.get("encoding_actions", {}).get(col, {}).get("method"),
                }
            }
            column_actions_frontend.append(col_actions)

        # Serialize column_actions_frontend to json file
        column_actions_path = os.path.join(output_folder, "column_actions_frontend.json")
        try:
            with open(column_actions_path, "w", encoding="utf-8") as f:
                json.dump(column_actions_frontend, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to write column actions to file: {e}")

        # Build final agent output for UI panel
        output_info = pipeline_state.get("preprocessing_output") or {}
        output_info["column_actions_frontend_path"] = column_actions_path
        output_info["policy_path"] = pipeline_state.get("preprocessing_plan_path") or ""
        pipeline_state["preprocessing_output"] = output_info

        agent_output = {
            "status": "success",
            "task_type": pipeline_state.get("task_type"),
            "column_actions": column_actions_frontend,
            "preprocessing_plan": pipeline_state.get("preprocessing_plan"),
            "X_train_path": pipeline_state.get("X_train_path"),
            "X_test_path": pipeline_state.get("X_test_path"),
            "y_train_path": pipeline_state.get("y_train_path"),
            "y_test_path": pipeline_state.get("y_test_path"),
        }

        merged_outputs = dict(pipeline_state.get("agent_outputs", {}))
        merged_outputs["preprocessing"] = agent_output
        pipeline_state["agent_outputs"] = merged_outputs

        return pipeline_state


def _build_feedback_context(state: PipelineState, agent_name: str = "preprocessing") -> str:
    """Format and pull user feedback for this specific agent node."""
    history = state.get("feedback_history", []) or []
    own = [h["feedback_text"] for h in history if h.get("agent") == agent_name]
    if own:
        return f"\n\nUser Feedback History for {agent_name}:\n" + "\n".join(f"- {f}" for f in own)
    return ""


def preprocessing_node(state: PipelineState) -> dict:
    """LangGraph node representing the Preprocessing Agent."""
    logger.info("[PreprocessingAgent] Executing LangGraph node")
    agent = PreprocessingAgent()

    result = agent.run(
        data_path=state["data_path"],
        prompt=state.get("prompt") or state.get("nl_query") or "preprocess the data",
        pipeline_state=state,
        target_column=state.get("target_column"),
        feedback_context=_build_feedback_context(state),
    )

    return result

def route_after_preprocessing(state: PipelineState) -> str:
    """LangGraph conditional edge router after the preprocessing node."""
    flags = state["intent_flags"]
    if flags.get("run_feature_engineering"):
        return "feature_engineering_agent"
    if flags.get("run_model_selection"):
        return "model_selection_agent"
    if flags.get("run_training"):
        return "training_agent"
    if flags.get("run_evaluation"):
        return "evaluation_agent"
    if flags.get("run_deployment"):
        return "deployment_agent"
    return "pipeline_done"
