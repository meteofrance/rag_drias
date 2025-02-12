import json
import time
from pathlib import Path
from typing import Dict


def add_json_with_lock(src_path: Path, dict_add: Dict):
    """Add dict to json without conflict with reader"""
    lock = src_path.parent / f"lock_file_{src_path.stem}.txt"
    ok = False
    for _ in range(5):
        try:
            # Attempt to acquire a lock
            with open(lock, "x"):
                try:
                    with open(src_path, "r") as f:
                        data = json.load(f)
                except FileNotFoundError:
                    data = []
                data.append(dict_add)
                with open(src_path, "w") as f:
                    json.dump(data, f)
                lock.unlink()  # Release the lock
                ok = True
                break
        except FileExistsError:
            print("Another process is reading the image. Retrying in a moment...")
            time.sleep(1)
    if not ok:
        print(f"ERROR: Could not copy file {src_path} to {src_path}")
