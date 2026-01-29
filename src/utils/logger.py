import os
import sys
import traceback
from datetime import datetime

# Get absolute project root (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Define log directory and file paths (assets/logs/)
LOG_DIR = os.path.join(PROJECT_ROOT, 'assets', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Ensure project root is in sys.path for other imports if needed
sys.path.append(PROJECT_ROOT)


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
