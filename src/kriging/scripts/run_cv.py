# src/kriging/scripts/run_cv.py
"""
Kriging-only cross-validation runner.

What it does
- Loads sparse points (coords + values)
- Runs K-Fold CV for 3 variuogram models: exponential, gaussian and, spherical
- Computes: MSE, MAE, RMSE, R2, PearsonR
- Saves:
  - results/tables/kriging_cv_summary_<tag>.csv
  - results/tables/kriging_cv_folds_<tag>.csv

Example (generic):
  python src/kriging/scripts/run_cv.py --coords_file data/coords.npy --values_file data/values.npy --tag seeg_delta
  
Example (dict-like values file, e.g. band powers saved as npy dict):
  python src/kriging/scripts/run_cv.py --coords_file data/coords.npy --values_file data/band_powers.npy --values_key Delta --tag seeg_Delta

Notes
- coords must be (N, 3): x,y,z
- values must be (N,) or a dict saved in .npy where you select a key via --values_key
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# sklearn + scipy are common in these projects; used for clean metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

try:
    from pykrige.ok3d import OrdinaryKriging3D
except ImportError as e:
    raise ImportError(
        "PyKrige is required. Install with: pip install pykrige"
    ) from e


VARIOGRAMS = ("exponential", "gaussian", "spherical")


@dataclass
class FoldResult:
    variogram: str
    fold: int
    n_train: int
    n_test: int
    mse: float
    mae: float
    rmse: float
    r2: float
    pearson_r: float
    fit_predict_time_s: float


def _repo_root_from_this_file() -> Path:
    """
    Assumes this script lives at: <repo>/src/kriging/scripts/run_cv.py
    """
    return Path(__file__).resolve().parents[3]


def _load_coords(path: Path) -> np.ndarray:
    coords = np.load(path)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be shape (N,3). Got {coords.shape} from {path}")
    return coords


def _load_values(path: Path, values_key: Optional[str] = None) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    # If user passed a key, treat as dict-like
    if values_key is not None:
        if isinstance(arr, np.ndarray) and arr.shape == ():
            arr = arr.item()  # unwrap 0-d object array
        if not isinstance(arr, dict):
            raise ValueError(
                f"--values_key was provided but {path} is not a dict-like npy."
            )
        if values_key not in arr:
            raise KeyError(f"Key '{values_key}' not found in {path}. Available: {list(arr.keys())[:20]}")
        values = np.asarray(arr[values_key], dtype=float)
    else:
        # regular numeric array
        values = np.asarray(arr, dtype=float)

    values = values.reshape(-1)
    return values


def _basic_checks(coords: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(coords) != len(values):
        raise ValueError(f"coords and values length mismatch: {len(coords)} vs {len(values)}")

    # remove NaNs (common in some pipelines)
    mask = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    if mask.sum() < len(values):
        coords = coords[mask]
        values = values[mask]

    if len(values) < 10:
        raise ValueError(f"Not enough samples after cleaning: {len(values)}")

    return coords, values


def _pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # pearsonr fails if constant arrays
    try:
        r = pearsonr(y_true, y_pred)[0]
        return float(r)
    except Exception:
        return float("nan")


def kriging_predict(
    train_coords: np.ndarray,
    train_values: np.ndarray,
    query_coords: np.ndarray,
    variogram_model: str,
) -> np.ndarray:
    """
    Ordinary Kriging in 3D using PyKrige.
    """
    ok = OrdinaryKriging3D(
        train_coords[:, 0], train_coords[:, 1], train_coords[:, 2],
        train_values,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )
    preds, _ = ok.execute(
        "points",
        query_coords[:, 0], query_coords[:, 1], query_coords[:, 2],
    )
    return np.asarray(preds, dtype=float).reshape(-1)


def run_cv(
    coords: np.ndarray,
    values: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[FoldResult]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: List[FoldResult] = []

    for variogram in VARIOGRAMS:
        for fold_idx, (tr, te) in enumerate(kf.split(coords), start=1):
            tc, tv = coords[tr], values[tr]
            qc, qv = coords[te], values[te]

            t0 = time.time()
            preds = kriging_predict(tc, tv, qc, variogram_model=variogram)
            dt = time.time() - t0

            mse = float(mean_squared_error(qv, preds))
            mae = float(mean_absolute_error(qv, preds))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(qv, preds))
            r = _pearson_safe(qv, preds)

            results.append(
                FoldResult(
                    variogram=variogram,
                    fold=fold_idx,
                    n_train=len(tr),
                    n_test=len(te),
                    mse=mse,
                    mae=mae,
                    rmse=rmse,
                    r2=r2,
                    pearson_r=r,
                    fit_predict_time_s=float(dt),
                )
            )

    return results


def summarize(results: List[FoldResult]) -> List[Dict[str, Any]]:
    rows = []
    for variogram in VARIOGRAMS:
        sub = [r for r in results if r.variogram == variogram]
        if not sub:
            continue

        def avg(x: List[float]) -> float:
            arr = np.asarray(x, dtype=float)
            return float(np.nanmean(arr))

        row = {
            "variogram": variogram,
            "folds": len(sub),
            "mse_mean": avg([r.mse for r in sub]),
            "mae_mean": avg([r.mae for r in sub]),
            "rmse_mean": avg([r.rmse for r in sub]),
            "r2_mean": avg([r.r2 for r in sub]),
            "pearson_r_mean": avg([r.pearson_r for r in sub]),
            "time_mean_s": avg([r.fit_predict_time_s for r in sub]),
            "time_total_s": float(np.nansum([r.fit_predict_time_s for r in sub])),
        }
        rows.append(row)

    # sort by mse ascending (best first)
    rows.sort(key=lambda d: d["mse_mean"])
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No results.")
        return

    cols = list(rows[0].keys())
    # simple aligned print
    col_widths = {c: max(len(c), max(len(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols}

    header = " | ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        def fmt(c):
            v = r[c]
            if isinstance(v, float):
                return f"{v:.6g}".ljust(col_widths[c])
            return str(v).ljust(col_widths[c])

        print(" | ".join(fmt(c) for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Kriging CV (3 variogram models).")
    parser.add_argument("--coords_file", type=str, required=True, help="Path to coords .npy (shape Nx3).")
    parser.add_argument("--values_file", type=str, required=True, help="Path to values .npy (shape N) or dict-like npy.")
    parser.add_argument("--values_key", type=str, default=None, help="Key for dict-like values npy (e.g., Delta, Theta, ...).")
    parser.add_argument("--splits", type=int, default=10, help="KFold splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tag", type=str, default="kriging", help="Tag used in output filenames.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir (defaults to <repo>/results/tables).",
    )
    args = parser.parse_args()

    coords_path = Path(args.coords_file).expanduser().resolve()
    values_path = Path(args.values_file).expanduser().resolve()

    coords = _load_coords(coords_path)
    values = _load_values(values_path, values_key=args.values_key)
    coords, values = _basic_checks(coords, values)

    results = run_cv(coords, values, n_splits=args.splits, seed=args.seed)
    summary_rows = summarize(results)

    # Output locations
    if args.out_dir is None:
        repo_root = _repo_root_from_this_file()
        out_dir = repo_root / "results" / "tables"
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()

    folds_csv = out_dir / f"kriging_cv_folds_{args.tag}.csv"
    summary_csv = out_dir / f"kriging_cv_summary_{args.tag}.csv"

    # Save fold-level
    fold_rows = [asdict(r) for r in results]
    _write_csv(folds_csv, fold_rows)

    # Save summary
    _write_csv(summary_csv, summary_rows)

    print("\nKriging CV summary (sorted by MSE):\n")
    _print_summary_table(summary_rows)
    print(f"\nSaved:\n- {summary_csv}\n- {folds_csv}\n")


if __name__ == "__main__":
    main()
  
