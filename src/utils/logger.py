import os,sys
import traceback
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from configs.settings import LOG_DIR, LOG_FILE



class Logger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== New Run — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def _write(self, level: str, msg: str):
        """Internal helper to format and write messages."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level.upper()}] {msg}"
        print(formatted)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    def info(self, msg: str):
        """Log general information."""
        self._write("INFO", msg)

    def warn(self, msg: str):
        """Log a warning message."""
        self._write("WARNING", msg)

    def error(self, msg: str, exc: Exception = None):
        """Log an error message, optionally with exception details."""
        if exc:
            msg += f"\n{traceback.format_exc()}"
        self._write("ERROR", msg)
