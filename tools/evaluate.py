def run(task,tool_input,prompt,data_path):
    print("=========================================================================")
    print(f"[TOOL] Evaluating model: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    return "accuracy=0.92" , "evaluation_report_path"