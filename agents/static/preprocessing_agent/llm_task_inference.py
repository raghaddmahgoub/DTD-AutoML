"""
LLM-assisted task inference with safe fallbacks
- Reads plan/dataset_profile.json
- Writes task_definition.json to plan/ and output/
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import requests

# ================== CONFIG ==================
PLAN_DIR = Path("Output/static/Plan")
OUTPUT_DIR = Path("Output/static/Preprocessing")

PROFILE_PATH = PLAN_DIR / "Titanic-Dataset_preprocessing_context (1).json"
PLAN_OUT = PLAN_DIR / "task_definition.json"
OUTPUT_OUT = OUTPUT_DIR / "task_definition.json"

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.1
TIMEOUT = 120
# ===========================================


def load_profile() -> list[dict]:
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"Dataset profile not found at {PROFILE_PATH.resolve()}")
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON found in response:\n{text}")
    return json.loads(text[start:end + 1])


def hf_inference(prompt: str) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("Missing $HF_TOKEN for Hugging Face Inference API")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "return_full_text": False,
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"HF API error {r.status_code}: {r.text}")
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    raise RuntimeError(f"Unexpected HF response: {data}")


def _rule_based_inference(profile: list[dict], target_column: str) -> dict:
    target_info = next((c for c in profile if c["column"] == target_column), None)
    if target_info is None:
        raise KeyError(f"Target column {target_column!r} not present in profile")

    dtype = str(target_info.get("dtype", "")).lower()
    n_unique = int(target_info.get("n_unique", 0))

    # Simple heuristics: small unique -> classification; numeric & many unique -> regression
    if ("float" in dtype or "int" in dtype) and n_unique > 30:
        return {"task_type": "regression", "metric": "r2"}
    else:
        return {"task_type": "classification", "metric": "accuracy"}


def infer_task_type(profile: list[dict], target_column: str) -> dict:
    target_info = next(col for col in profile if col["column"] == target_column)

    prompt = f"""
You are an AutoML system.

Target column profile:
{json.dumps(target_info, indent=2)}

Decide:
1) task_type: classification or regression
2) metric: best default metric for CV model selection

Return ONLY valid JSON:
{{
  "task_type": "classification",
  "metric": "accuracy"
}}
""".strip()

    try:
        out = hf_inference(prompt)
        return extract_json(out)
    except Exception:
        return _rule_based_inference(profile, target_column)


def main():
    for d in (PLAN_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    profile = load_profile()
    # First marked 'is_target' or fallback to last column
    target_column = next((col["column"] for col in profile if col.get("is_target") is True), profile[-1]["column"])

    result = infer_task_type(profile, target_column)

    print("\n=== LLM TASK INFERENCE ===\n")
    print(result)

    # Write both to plan/ and output/
    for p in (PLAN_OUT, OUTPUT_OUT):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"\nTask definition written to:\n- {PLAN_OUT.resolve()}\n- {OUTPUT_OUT.resolve()}")


if __name__ == "__main__":
    main()
