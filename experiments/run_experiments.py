# experiments/run_experiments.py
"""
Unified experiment runner (config-driven).

What it does
- Reads a YAML config describing:
  - dataset files (coords + values + optional values_key)
  - CV settings (splits, seed)
  - methods to run (kriging/mlp/unet + hyperparams)
- Runs each method and aggregates a single comparison table:
  - results/tables/compare_<tag>.csv
- Also saves method-specific fold/summary CSVs via existing save_cv_outputs()

Assumptions about your repo (based on what we built together)
- src/utils/io.py provides: load_npy, load_npy_dict
- src/utils/paths.py provides: tables_dir
- src/evaluation/cv.py provides: cross_validate (generic KFold)
- src/evaluation/reporting.py provides: save_cv_outputs
- src/kriging/predict.py provides: kriging_predict OR you can edit import below
- src/mlp/predict.py provides: mlp_predict + TrainConfig in src/mlp/train.py
- src/unet has scripts/run_cv.py but CV is heavier; we call a small internal runner here

Install dependency:
- pip install pyyaml

Example:
python experiments/run_experiments.py --config experiments/configs/seeg_delta_full.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.utils.io import load_npy, load_npy_dict
from src.utils.paths import tables_dir
from src.utils.seed import seed_everything

from src.evaluation.cv import cross_validate
from src.evaluation.reporting import save_cv_outputs


# ---------- Helpers ----------

def _as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def _load_dataset(cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    Expected config keys:
      dataset:
        coords_file: ...
        values_file: ...
        values_key: (optional)
    """
    ds = cfg["dataset"]
    coords = np.asarray(load_npy(ds["coords_file"]), dtype=float)

    values_key = ds.get("values_key", None)
    if values_key is None:
        values = np.asarray(load_npy(ds["values_file"]), dtype=float).reshape(-1)
    else:
        d = load_npy_dict(ds["values_file"])
        if values_key not in d:
            raise KeyError(f"values_key '{values_key}' not found. Available keys: {list(d.keys())[:30]}")
        values = np.asarray(d[values_key], dtype=float).reshape(-1)

    # clean NaNs
    mask = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    coords = coords[mask]
    values = values[mask]
    return coords, values, values_key

def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    out.update(b)
    return out

def _write_compare_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # stable header: union of keys in insertion order
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def _print_top(rows: List[Dict[str, Any]], key: str = "MSE_mean", n: int = 10) -> None:
    rows2 = [r for r in rows if key in r and r[key] is not None]
    rows2.sort(key=lambda r: float(r.get(key, np.inf)))
    print(f"\nTop results by {key}:")
    for r in rows2[:n]:
        print(f"  {r.get('method'):>10} | {r.get('variant',''):>12} | {key}={r.get(key):.6g} | R2={r.get('R2_mean'):.4g}")

# ---------- Method runners ----------

def run_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    cv_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
    base_tag: str,
    values_key: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Config example:
      - name: kriging
        variograms: [exponential, gaussian, spherical]
    """
    # Prefer using your shared kriging_predict if present.
    # If your kriging_predict signature differs, adjust here.
    try:
        from src.kriging.predict import kriging_predict
    except Exception:
        # fallback: use PyKrige directly here
        from pykrige.ok3d import OrdinaryKriging3D

        def kriging_predict(train_coords, train_values, query_coords, variogram_model="exponential"):
            ok = OrdinaryKriging3D(
                train_coords[:, 0], train_coords[:, 1], train_coords[:, 2],
                train_values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
            )
            preds, _ = ok.execute("points",
                                 query_coords[:, 0], query_coords[:, 1], query_coords[:, 2])
            return np.asarray(preds, dtype=float).reshape(-1)

    variograms = method_cfg.get("variograms", ["exponential", "gaussian", "spherical"])

    rows: List[Dict[str, Any]] = []
    for v in variograms:
        def predict_fn(tc, tv, qc):
            return kriging_predict(tc, tv, qc, variogram_model=v)

        fold_rows, summary_rows = cross_validate(
            coords, values,
            predict_fn=predict_fn,
            n_splits=int(cv_cfg.get("splits", 10)),
            seed=int(cv_cfg.get("seed", 42)),
        )

        # enrich summary
        summary_rows[0].update({
            "method": "kriging",
            "variant": v,
            "values_key": values_key,
        })

        save_cv_outputs(fold_rows, summary_rows, tag=f"kriging_{base_tag}_{v}")
        rows.append(summary_rows[0])

    return rows


def run_mlp(
    coords: np.ndarray,
    values: np.ndarray,
    cv_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
    base_tag: str,
    values_key: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Config example:
      - name: mlp
        variants:
          - {aug: none}
          - {aug: jitter, jitter_std: 0.5}
    or just single settings at top-level.
    """
    from src.mlp.predict import mlp_predict
    from src.mlp.train import TrainConfig

    # allow multiple variants
    variants = method_cfg.get("variants", [method_cfg])

    rows: List[Dict[str, Any]] = []

    for i, var_cfg in enumerate(variants, start=1):
        cfg = TrainConfig(
            hidden=int(var_cfg.get("hidden", method_cfg.get("hidden", 128))),
            depth=int(var_cfg.get("depth", method_cfg.get("depth", 4))),
            dropout=float(var_cfg.get("dropout", method_cfg.get("dropout", 0.0))),
            lr=float(var_cfg.get("lr", method_cfg.get("lr", 1e-3))),
            epochs=int(var_cfg.get("epochs", method_cfg.get("epochs", 300))),
            batch_size=int(var_cfg.get("batch_size", method_cfg.get("batch_size", 128))),
            weight_decay=float(var_cfg.get("weight_decay", method_cfg.get("weight_decay", 0.0))),
            aug=str(var_cfg.get("aug", method_cfg.get("aug", "none"))),
            jitter_std=float(var_cfg.get("jitter_std", method_cfg.get("jitter_std", 0.5))),
            noise_std=float(var_cfg.get("noise_std", method_cfg.get("noise_std", 0.01))),
            device=str(var_cfg.get("device", method_cfg.get("device", "cpu"))),
            seed=int(var_cfg.get("seed", method_cfg.get("seed", cv_cfg.get("seed", 42)))),
        )

        variant_name = var_cfg.get("name", cfg.aug)
        if variant_name is None or variant_name == "":
            variant_name = f"variant{i}"

        def predict_fn(tc, tv, qc):
            return mlp_predict(tc, tv, qc, cfg)

        fold_rows, summary_rows = cross_validate(
            coords, values,
            predict_fn=predict_fn,
            n_splits=int(cv_cfg.get("splits", 10)),
            seed=int(cv_cfg.get("seed", 42)),
        )

        summary_rows[0].update({
            "method": "mlp",
            "variant": variant_name,
            "values_key": values_key,
            "hidden": cfg.hidden,
            "depth": cfg.depth,
            "dropout": cfg.dropout,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "aug": cfg.aug,
            "jitter_std": cfg.jitter_std,
            "noise_std": cfg.noise_std,
            "device": cfg.device,
        })

        save_cv_outputs(fold_rows, summary_rows, tag=f"mlp_{base_tag}_{variant_name}")
        rows.append(summary_rows[0])

    return rows


def run_unet(
    coords: np.ndarray,
    values: np.ndarray,
    cv_cfg: Dict[str, Any],
    method_cfg: Dict[str, Any],
    base_tag: str,
    values_key: Optional[str],
) -> List[Dict[str, Any]]:
    """
    U-Net CV is different: it operates on a voxelised volume, and CV splits known voxels.
    This uses the unet modules we set up.

    Config example:
      - name: unet
        step: 4
        splits: 5
        base: 16
        epochs: 200
        hide_frac: 0.2
    """
    import time
    from sklearn.model_selection import KFold

    from src.evaluation.metrics import regression_metrics
    from src.unet.volume import build_volume_from_points, make_input_from_available_mask
    from src.unet.train import TrainConfig, fit_unet

    step = int(method_cfg.get("step", 4))
    jitter_std = float(method_cfg.get("jitter_std", 0.0))

    # build voxel volume once
    pack = build_volume_from_points(coords, values, step=step, jitter_std=jitter_std, seed=int(cv_cfg.get("seed", 42)))
    vol_std = pack.volume_std
    known_mask = pack.known_mask.astype(bool)
    known_idx = np.argwhere(known_mask)

    splits = int(method_cfg.get("splits", 5))
    seed = int(cv_cfg.get("seed", 42))

    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)

    fold_rows: List[Dict[str, Any]] = []
    t0_all = time.time()

    for fold, (tr, va) in enumerate(kf.split(known_idx), start=1):
        train_known = np.zeros_like(known_mask, dtype=np.float32)
        val_known = np.zeros_like(known_mask, dtype=np.float32)

        tr_idx = known_idx[tr]
        va_idx = known_idx[va]
        train_known[tr_idx[:, 0], tr_idx[:, 1], tr_idx[:, 2]] = 1.0
        val_known[va_idx[:, 0], va_idx[:, 1], va_idx[:, 2]] = 1.0

        cfg = TrainConfig(
            base=int(method_cfg.get("base", 16)),
            lr=float(method_cfg.get("lr", 1e-3)),
            weight_decay=float(method_cfg.get("weight_decay", 1e-5)),
            epochs=int(method_cfg.get("epochs", 200)),
            device=str(method_cfg.get("device", "cpu")),
            seed=seed + fold,
            hide_frac=float(method_cfg.get("hide_frac", 0.2)),
            tv_alpha=float(method_cfg.get("tv_alpha", 0.0)),
            patience=int(method_cfg.get("patience", 25)),
        )

        t0 = time.time()
        model = fit_unet(vol_std, train_known, val_known, cfg)
        train_time = time.time() - t0

        # predict with only train-known visible
        import torch
        x_np = make_input_from_available_mask(vol_std, train_known)
        x = torch.tensor(x_np, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy().squeeze()

        y_true = vol_std[val_known.astype(bool)].reshape(-1)
        y_pred = pred[val_known.astype(bool)].reshape(-1)
        m = regression_metrics(y_true, y_pred)

        fold_rows.append(m | {
            "fold": fold,
            "n_train_vox": int(train_known.sum()),
            "n_val_vox": int(val_known.sum()),
            "train_time_s": float(train_time),
        })

    def mean_key(k):
        arr = np.array([r[k] for r in fold_rows], dtype=float)
        return float(np.nanmean(arr))

    summary_rows = [{
        "method": "unet3d",
        "variant": method_cfg.get("name", "default"),
        "values_key": values_key,
        "splits": splits,
        "MSE_mean": mean_key("MSE"),
        "MAE_mean": mean_key("MAE"),
        "RMSE_mean": mean_key("RMSE"),
        "R2_mean": mean_key("R2"),
        "PearsonR_mean": mean_key("PearsonR"),
        "train_time_mean_s": mean_key("train_time_s"),
        "train_time_total_s": float(np.nansum([r["train_time_s"] for r in fold_rows])),
        "wall_time_total_s": float(time.time() - t0_all),
        "step": step,
        "jitter_std": jitter_std,
        "base": int(method_cfg.get("base", 16)),
        "epochs": int(method_cfg.get("epochs", 200)),
        "hide_frac": float(method_cfg.get("hide_frac", 0.2)),
        "device": str(method_cfg.get("device", "cpu")),
        "seed": seed,
    }]

    save_cv_outputs(fold_rows, summary_rows, tag=f"unet_{base_tag}")
    return summary_rows


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg_path = _as_path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Global seed
    seed = int(cfg.get("cv", {}).get("seed", 42))
    seed_everything(seed)

    coords, values, values_key = _load_dataset(cfg)

    cv_cfg = cfg.get("cv", {})
    base_tag = cfg.get("output", {}).get("tag", cfg_path.stem)

    methods = cfg.get("methods", [])
    if not methods:
        raise ValueError("No methods specified in config under 'methods:'")

    all_rows: List[Dict[str, Any]] = []

    for m in methods:
        name = m.get("name")
        if name is None:
            raise ValueError("Each method must have a 'name' field (kriging/mlp/unet).")

        print(f"\nRunning method: {name}")
        if name.lower() == "kriging":
            all_rows.extend(run_kriging(coords, values, cv_cfg, m, base_tag, values_key))
        elif name.lower() == "mlp":
            all_rows.extend(run_mlp(coords, values, cv_cfg, m, base_tag, values_key))
        elif name.lower() == "unet":
            all_rows.extend(run_unet(coords, values, cv_cfg, m, base_tag, values_key))
        else:
            raise ValueError(f"Unknown method name '{name}'. Use kriging/mlp/unet.")

    # Save combined comparison table
    out_path = tables_dir() / f"compare_{base_tag}.csv"
    _write_compare_csv(all_rows, out_path)

    print(f"\nSaved combined comparison table:\n  {out_path}")
    _print_top(all_rows, key="MSE_mean", n=10)


if __name__ == "__main__":
    main()

