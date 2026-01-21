from pathlib import Path

def repo_root() -> Path:
    # Assumes this file is at <repo>/src/utils/paths.py
    return Path(__file__).resolve().parents[2]

def data_dir() -> Path:
    return repo_root() / "data"

def results_dir() -> Path:
    return repo_root() / "results"

def tables_dir() -> Path:
    return results_dir() / "tables"

def figures_dir() -> Path:
    return results_dir() / "figures"

