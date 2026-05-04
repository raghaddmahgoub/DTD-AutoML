def run(task,prompt,data_path):
    print(f"[TOOL] Evaluating model: {task}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    return "accuracy=0.92" , "evaluation_report_path"