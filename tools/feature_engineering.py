from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state


@tool
def feature_engineering(task, tool_input, prompt, data_path, llm, state=None):
    """Perform feature engineering on the dataset to create new features or transform existing ones."""
    print("=========================================================================")
    print(f"[TOOL] Feature engineering: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    pipeline_state = merge_state(
        ensure_state(state, data_path, prompt),
        {"step": "features_engineered", "status": "success"},
    )
    return {"status": "success", "message": "features"}, pipeline_state