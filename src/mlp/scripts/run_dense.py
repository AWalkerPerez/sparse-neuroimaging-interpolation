# src/mlp/scripts/run_dense.py
"""
MLP dense prediction + brain slice figures (convex hull grid).

What it does
- Loads sparse coords + values
- Loads a trained MLP checkpoint (model + scaler)
- Builds a regular 3D grid and masks to points inside the convex hull
- Predicts values on those grid points
- Converts predictions back to a 3D volume
- Saves slice plots to results/figures/

Requires:
- You have already created: src/evaluation/spatial_viz.py (from earlier)
- You trained a model with: src/mlp/scripts/train_full.py

Example:
python src/mlp/scripts/run_dense.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --model_path results/models/mlp/seeg_delta.pt \
  --scaler_path results/models/mlp/seeg_delta_scaler.npz \
  --tag mlp_seeg_delta \
  --res 40 \
  --axis z \
  --n_slices 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from src.utils.io import load_npy, load_npy_dict, ensure_dir
from src.utils.seed import seed_everything
from src.utils.paths import figures_dir

from src.mlp.train import TrainConfig, load_mlp
from src.evaluation.spatial_viz import (
    build_grid_within_convex_hull,
    points_inside_to_volume,
    plot_volume_slices,
    plot_sparse_points_3d,
)


def main():
    p = argparse.ArgumentParser(description="Dense MLP prediction on convex-hull grid + slice plots.")
    p.add_argument("--coords_file", required=True)
    p.add_argument("--values_file", required=True)
    p.add_argument("--values_key", default=None)

    p.add_argument("--model_path", required=True, help="Path to .pt saved by train_full.py")
    p.add_argument("--scaler_path", required=True, help="Path to _scaler.npz saved by train_full.py")

    p.add_argument("--tag", default="mlp_dense")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)

    # Must match the model you trained
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # Grid / plots
    p.add_argument("--res", type=int, default=30, help="Grid resolution per axis")
    p.add_argument("--margin", type=float, default=0.0, help="Expand bounds by +/- margin")
    p.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"])
    p.add_argument("--n_slices", type=int, default=6)

    args = p.parse_args()

    seed_everything(args.seed)

    coords = np.asarray(load_npy(args.coords_file), dtype=float)
    if args.values_key is None:
        values = np.asarray(load_npy(args.values_file), dtype=float).reshape(-1)
    else:
        d = load_npy_dict(args.values_file)
        values = np.asarray(d[args.values_key], dtype=float).reshape(-1)

    # Clean
    mask = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    coords = coords[mask]
    values = values[mask]

    # Load trained model + scaler (must match architecture)
    cfg = TrainConfig(hidden=args.hidden, depth=args.depth, dropout=args.dropout, device=args.device, seed=args.seed)
    model, scaler = load_mlp(args.model_path, args.scaler_path, cfg)

    # Build convex-hull grid
    hg = build_grid_within_convex_hull(coords, res=args.res, margin=args.margin)

    # Predict on inside-hull grid points
    Xq = scaler.transform(hg.grid_points_inside)
    Xq_t = torch.tensor(Xq, dtype=torch.float32, device=cfg.device)

    model.eval()
    with torch.no_grad():
        preds_inside = model(Xq_t).detach().cpu().numpy().reshape(-1)

    # Back to volume (outside hull -> NaN)
    V_pred = points_inside_to_volume(preds_inside, hg, fill_value=np.nan)

    # Save figures
    out_dir = figures_dir()
    ensure_dir(out_dir)

    # 3D scatter of sparse points
    scatter_path = out_dir / f"{args.tag}_sparse_points.png"
    plot_sparse_points_3d(coords, values, title=f"Sparse points ({args.tag})", save_path=scatter_path)

    # Slice plot
    slice_path = out_dir / f"{args.tag}_pred_slices_{args.axis}.png"
    plot_volume_slices(
        V_pred,
        axis=args.axis,
        n_slices=args.n_slices,
        title=f"MLP dense prediction slices ({args.tag})",
        save_path=slice_path,
    )

    print("Saved figures:")
    print(f"- {scatter_path}")
    print(f"- {slice_path}")


if __name__ == "__main__":
    main()
