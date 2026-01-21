from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from src.utils.io import ensure_dir
from src.utils.paths import figures_dir

def bar_plot_from_summary(
    summary_rows: List[Dict],
    x_key: str,
    y_key: str,
    title: str,
    filename: str,
    out_dir: Path | None = None,
) -> Path:
    """
    Makes a simple bar chart (e.g., y_key=MSE_mean by variogram/method).
    """
    out = out_dir or figures_dir()
    ensure_dir(out)

    xs = [r[x_key] for r in summary_rows]
    ys = [r[y_key] for r in summary_rows]

    plt.figure()
    plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)

    path = out / filename
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path

