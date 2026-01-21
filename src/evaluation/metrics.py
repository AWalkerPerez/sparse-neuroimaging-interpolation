from __future__ import annotations
import numpy as np
from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def _pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return float("nan")

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Returns: MSE, MAE, RMSE, R2, PearsonR
    Handles NaNs in predictions (drops them).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"MSE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "PearsonR": np.nan}

    yt, yp = y_true[mask], y_pred[mask]
    mse = float(mean_squared_error(yt, yp))
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(sqrt(mse))
    r2 = float(r2_score(yt, yp))
    r = _pearson_safe(yt, yp)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "PearsonR": r}

