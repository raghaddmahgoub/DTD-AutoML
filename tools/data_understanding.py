def run(task,prompt,data_path):
    print(f"[TOOL] Understanding data: {task}")
    print(f"[TOOL] Prompt: {prompt}")
    print(f"[TOOL] Data path: {data_path}")
    return "data_summary","new_data_summary_path"