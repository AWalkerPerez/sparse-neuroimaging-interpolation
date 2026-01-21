from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import joblib

from src.utils.io import load_npy, load_npy_dict, ensure_dir
from src.utils.seed import seed_everything
from src.utils.paths import figures_dir

from src.unet.volume import build_volume_from_points, make_input_from_available_mask
from src.unet.train import TrainConfig, load_unet

from src.evaluation.spatial_viz import plot_volume_slices


def main():
    p = argparse.ArgumentParser(description="Dense U-Net prediction + slice plots.")
    p.add_argument("--coords_file", required=True)
    p.add_argument("--values_file", required=True)
    p.add_argument("--values_key", default=None)

    p.add_argument("--model_path", required=True)
    p.add_argument("--scaler_path", required=True)  # joblib scaler

    p.add_argument("--tag", default="unet_dense")

    p.add_argument("--step", type=int, default=4)
    p.add_argument("--jitter_std", type=float, default=0.0)

    p.add_argument("--device", default="cpu")
    p.add_argument("--base", type=int, default=16)

    p.add_argument("--axis", default="z", choices=["x", "y", "z"])
    p.add_argument("--n_slices", type=int, default=6)

    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)

    coords = np.asarray(load_npy(args.coords_file), dtype=float)
    if args.values_key is None:
        values = np.asarray(load_npy(args.values_file), dtype=float).reshape(-1)
    else:
        d = load_npy_dict(args.values_file)
        values = np.asarray(d[args.values_key], dtype=float).reshape(-1)

    pack = build_volume_from_points(coords, values, step=args.step, jitter_std=args.jitter_std, seed=args.seed)
    train_known = pack.known_mask.astype(np.float32)

    cfg = TrainConfig(base=args.base, device=args.device, seed=args.seed)
    model = load_unet(args.model_path, cfg)

    # load scaler (optional, for inverse transform)
    scaler = joblib.load(args.scaler_path)

    x_np = make_input_from_available_mask(pack.volume_std, train_known)
    x = torch.tensor(x_np, dtype=torch.float32, device=cfg.device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred_std = model(x).detach().cpu().numpy().squeeze()  # (D,H,W)

    # Inverse transform for nicer interpretation (optional)
    pred = pred_std.copy()
    known = train_known.astype(bool)
    # scaler expects shape (-1,1)
    pred[known] = scaler.inverse_transform(pred_std[known].reshape(-1,1)).reshape(-1)

    out = figures_dir()
    ensure_dir(out)

    # Plot slices of the *standardised* prediction (usually cleaner for visuals)
    slice_path = out / f"{args.tag}_pred_slices_{args.axis}.png"
    plot_volume_slices(pred_std, axis=args.axis, n_slices=args.n_slices,
                       title=f"U-Net dense prediction (std) — {args.tag}", save_path=slice_path)

    print("Saved:", slice_path)


if __name__ == "__main__":
    main()

