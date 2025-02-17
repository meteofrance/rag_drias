import json
import threading
import time
from pathlib import Path
from typing import Dict

# ----- Threading Lock -----

lock = threading.Lock()


def add_json_with_lock(src_path: Path, dict_add: Dict):
    """Add dict to json without conflict with reader"""
    with lock:
        try:
            with open(src_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        data.append(dict_add)
        with open(src_path, "w") as f:
            json.dump(data, f)
