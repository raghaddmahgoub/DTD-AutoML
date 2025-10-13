import os
from datetime import datetime
from src.config.settings import LOG_DIR, LOG_FILE

class Logger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"\n\n=== New Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def log(self, msg: str):
        print(f"[LOG] {msg}")
        with open(LOG_FILE, "a") as f:
            f.write(f"[LOG] {msg}\n")
