from __future__ import annotations
from datetime import datetime

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")

