import json
from pathlib import Path
from datetime import datetime

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "requests.jsonl"


def write_log(data: dict):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **data
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
