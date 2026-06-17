import os
import json
import hashlib
import pandas as pd
from pathlib import Path

CACHE_DIR = "cache/datasets"
os.makedirs(CACHE_DIR, exist_ok=True)

def compute_dataset_fingerprint(file_path: str) -> str:
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()

def build_dataset_snapshot(file_path: str) -> dict:
    df = pd.read_csv(file_path)

    return {
        "file_name": os.path.basename(file_path),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

def get_cached_dataset(file_path: str):
    data_cache = compute_dataset_fingerprint(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{data_cache}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)

    snapshot = build_dataset_snapshot(file_path)

    cache_contents = {
        "dataset_cache": data_cache,
        "snapshot": snapshot
    }

    with open(cache_file, "w") as f:
        json.dump(cache_contents, f, indent=2)

    return cache_contents