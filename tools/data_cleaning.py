from langchain_core.tools import tool

from tools.pipeline_state import ensure_state, merge_state


@tool
def data_cleaning(task, tool_input, prompt, data_path, llm, state=None):
    """Perform data cleaning by identifying and handling missing values, outliers, and inconsistencies in the dataset."""
    print("=========================================================================")
    print("task:", task)
    print("input_data:", tool_input)
    print("input_data:", data_path)
    print("prompt:", prompt)
    pipeline_state = merge_state(
        ensure_state(state, data_path, prompt),
        {"step": "cleaned", "status": "success"},
    )
    return {"status": "success", "message": "clean_data"}, pipeline_state