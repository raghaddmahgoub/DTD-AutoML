from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state


@tool
def evaluate(task, tool_input, prompt, data_path, llm, state=None):
    """Evaluate the performance of a trained machine learning model using appropriate metrics and techniques."""
    print("=========================================================================")
    print(f"[TOOL] Evaluating model: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    pipeline_state = ensure_state(state, data_path, prompt)
    metrics = pipeline_state.get("model_metrics") or {}
    result = {
        "status": "success",
        "metrics": metrics,
        "best_model": metrics.get("best_model"),
        "best_score": metrics.get("best_score"),
    }
    pipeline_state = merge_state(pipeline_state, {"step": "evaluated", "status": "success"})
    return result, pipeline_state
