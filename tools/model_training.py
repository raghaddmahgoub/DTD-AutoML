def run(task,tool_input,prompt,data_path):
    print("=========================================================================")
    print(f"[TOOL] Training model: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    return "model","new_model_path"