from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import joblib

from src.utils.io import load_npy, load_npy_dict, ensure_dir
from src.utils.seed import seed_everything

from src.unet.volume import build_volume_from_points
from src.unet.train import TrainConfig, fit_unet, save_unet


def main():
    p = argparse.ArgumentParser(description="Train U-Net 3D on full data (self-supervised masking) and save checkpoint.")
    p.add_argument("--coords_file", required=True)
    p.add_argument("--values_file", required=True)
    p.add_argument("--values_key", default=None)
    p.add_argument("--tag", default="unet_model")

    p.add_argument("--step", type=int, default=4)
    p.add_argument("--jitter_std", type=float, default=0.0)

    p.add_argument("--device", default="cpu")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hide_frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--tv_alpha", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_dir", default="results/models/unet")
    args = p.parse_args()

    seed_everything(args.seed)

    coords = np.asarray(load_npy(args.coords_file), dtype=float)
    if args.values_key is None:
        values = np.asarray(load_npy(args.values_file), dtype=float).reshape(-1)
    else:
        d = load_npy_dict(args.values_file)
        values = np.asarray(d[args.values_key], dtype=float).reshape(-1)

    pack = build_volume_from_points(coords, values, step=args.step, jitter_std=args.jitter_std, seed=args.seed)
    train_known = pack.known_mask.astype(np.float32)   # all known are "train"
    val_known = None

    cfg = TrainConfig(
        base=args.base,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        hide_frac=args.hide_frac,
        tv_alpha=args.tv_alpha,
        patience=args.patience,
    )

    model = fit_unet(pack.volume_std, train_known, val_known, cfg)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    model_path = out_dir / f"{args.tag}.pt"
    scaler_path = out_dir / f"{args.tag}_scaler.joblib"
    meta_path = out_dir / f"{args.tag}_meta.txt"

    save_unet(model, model_path)
    joblib.dump(pack.scaler, scaler_path)

    meta_path.write_text("\n".join([
        f"coords_file={args.coords_file}",
        f"values_file={args.values_file}",
        f"values_key={args.values_key}",
        f"step={args.step}",
        f"jitter_std={args.jitter_std}",
        f"base={args.base}",
        f"epochs={args.epochs}",
        f"lr={args.lr}",
        f"weight_decay={args.weight_decay}",
        f"hide_frac={args.hide_frac}",
        f"tv_alpha={args.tv_alpha}",
        f"patience={args.patience}",
        f"device={args.device}",
        f"seed={args.seed}",
    ]))

    print(f"Saved model:  {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved meta:   {meta_path}")


if __name__ == "__main__":
    main()

