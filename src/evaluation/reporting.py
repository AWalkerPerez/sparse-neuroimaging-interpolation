from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List

from src.utils.io import save_csv
from src.utils.paths import tables_dir

def summarize_folds(fold_rows: List[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Adds optional extra fields (e.g., method name, variogram) to the single summary row.
    """
    if not fold_rows:
        return {}
    # fold_rows already produced a summary elsewhere sometimes; this helper is for building a row
    # kept for convenience if you want to roll your own summaries.
    out = dict(extra or {})
    return out

def save_cv_outputs(
    fold_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    tag: str,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """
    Saves fold-level and summary csv files to results/tables by default.
    """
    out = out_dir or tables_dir()
    folds_path = out / f"{tag}_cv_folds.csv"
    summary_path = out / f"{tag}_cv_summary.csv"

    save_csv(fold_rows, folds_path)
    save_csv(summary_rows, summary_path)
    return summary_path, folds_path

