"""Preprocessing tool for the dynamic preprocessing agent."""
from langchain_core.tools import tool
from pathlib import Path
import json

from tools.pipeline_state import ensure_state, merge_state


@tool
def preprocessing_execution(task, tool_input, prompt, data_path, llm, state=None):
    """
    Execute data preprocessing pipeline: handle missing values, outliers, encoding, scaling, imbalance handling.

    Inputs (via tool_input):
    - target_column: str - Target column to predict
    - test_size: float (default 0.2)
    - random_state: int (default 42)
    - use_llm: bool (default True) - Use LLM for policy decisions
    - output_folder: str (optional)

    Returns:
    - status: success/error
    - X_train_path, X_test_path, y_train_path, y_test_path
    - preprocessing_summary
    """
    from agents.static.preprocessing_agent.preprocessing_node import PreprocessingNode

    pipeline_state = ensure_state(state, data_path, prompt)

    try:
        # Extract inputs
        target_column = tool_input.get(
            "target_column") or pipeline_state.get("target_column")
        test_size = float(tool_input.get("test_size", 0.2))
        random_state = int(tool_input.get("random_state", 42))
        use_llm = bool(tool_input.get("use_llm", True))
        output_folder = tool_input.get("output_folder") or str(
            Path("Output") / "Preprocessing" / Path(data_path).stem
        )

        # Initialize preprocessing node
        config = {
            "test_size": test_size,
            "random_state": random_state,
            "use_llm": use_llm,
        }

        preprocessing_node = PreprocessingNode(config=config)

        # Run preprocessing
        preprocess_state = {
            "dataset_path": data_path,
            "target_column": target_column,
            "output_folder": output_folder,
            "test_size": test_size,
            "random_state": random_state,
            "use_llm": use_llm,
        }

        result_state = preprocessing_node.run(preprocess_state)

        # Extract paths and metadata
        output_info = {
            "X_train_path": result_state.get("X_train_path"),
            "X_test_path": result_state.get("X_test_path"),
            "y_train_path": result_state.get("y_train_path"),
            "y_test_path": result_state.get("y_test_path"),
            "summary_path": result_state.get("summary_path"),
            "column_actions_frontend_path": result_state.get("column_actions_frontend_path"),
            "policy_path": result_state.get("policy_path"),
            "status": result_state.get("status", "failed"),
            "error": result_state.get("error"),
        }

        # Update pipeline state
        pipeline_state = merge_state(
            pipeline_state,
            {
                "step": "preprocessing_complete",
                "status": output_info["status"],
                "preprocessing_output": output_info,
                "X_train_path": output_info["X_train_path"],
                "X_test_path": output_info["X_test_path"],
                "y_train_path": output_info["y_train_path"],
                "y_test_path": output_info["y_test_path"],
                "target_column": target_column,
            }
        )

        result = {
            "status": output_info["status"],
            "message": f"Preprocessing {'succeeded' if output_info['status'] == 'success' else 'failed'}",
            "preprocessing_output": output_info,
        }

        return result, pipeline_state

    except Exception as e:
        import traceback
        error_msg = f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}"
        result = {
            "status": "error",
            "error": error_msg,
            "message": "Preprocessing execution failed",
        }
        pipeline_state = merge_state(
            pipeline_state,
            {"step": "preprocessing_failed", "status": "error", "error": error_msg}
        )
        return result, pipeline_state
