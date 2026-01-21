from __future__ import annotations
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from .metrics import regression_metrics

PredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

def cross_validate(
    coords: np.ndarray,
    values: np.ndarray,
    predict_fn: PredictFn,
    n_splits: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generic KFold CV for any interpolation method.

    predict_fn signature:
        preds = predict_fn(train_coords, train_values, test_coords)

    Returns:
      fold_rows: list of per-fold dicts
      summary_rows: list with one dict summary (mean across folds)
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    # Clean invalid rows
    mask = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    coords, values = coords[mask], values[mask]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows: List[Dict] = []
    for fold_idx, (tr, te) in enumerate(kf.split(coords), start=1):
        tc, tv = coords[tr], values[tr]
        qc, qv = coords[te], values[te]

        t0 = time.time()
        preds = predict_fn(tc, tv, qc)
        dt = time.time() - t0

        m = regression_metrics(qv, preds)
        fold_rows.append({
            "fold": fold_idx,
            "n_train": len(tr),
            "n_test": len(te),
            "time_s": float(dt),
            **m
        })

    # Summary (mean across folds)
    def nanmean(key: str) -> float:
        arr = np.array([r[key] for r in fold_rows], dtype=float)
        return float(np.nanmean(arr))

    summary = {
        "folds": len(fold_rows),
        "MSE_mean": nanmean("MSE"),
        "MAE_mean": nanmean("MAE"),
        "RMSE_mean": nanmean("RMSE"),
        "R2_mean": nanmean("R2"),
        "PearsonR_mean": nanmean("PearsonR"),
        "time_mean_s": nanmean("time_s"),
        "time_total_s": float(np.nansum([r["time_s"] for r in fold_rows])),
    }

    return fold_rows, [summary]

