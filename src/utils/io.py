from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def load_npy(path: str | Path) -> np.ndarray:
    p = Path(path)
    return np.load(p)

def load_npy_dict(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    obj = np.load(p, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError(f"{p} is not a dict-like npy.")
    return obj

