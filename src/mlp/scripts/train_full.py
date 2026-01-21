from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from src.utils.io import load_npy, load_npy_dict, ensure_dir
from src.utils.seed import seed_everything
from src.mlp.train import TrainConfig, fit_mlp

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coords_file", required=True)
    p.add_argument("--values_file", required=True)
    p.add_argument("--values_key", default=None)
    p.add_argument("--tag", default="mlp_model")
    p.add_argument("--device", default="cpu")

    # hyperparams
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)

    # output
    p.add_argument("--out_dir", default="results/models/mlp")
    args = p.parse_args()

    seed_everything(args.seed)

    coords = np.asarray(load_npy(args.coords_file), dtype=float)
    if args.values_key is None:
        values = np.asarray(load_npy(args.values_file), dtype=float).reshape(-1)
    else:
        d = load_npy_dict(args.values_file)
        values = np.asarray(d[args.values_key], dtype=float).reshape(-1)

    cfg = TrainConfig(
        hidden=args.hidden,
        depth=args.depth,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )

    model, scaler = fit_mlp(coords, values, cfg)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Save model weights
    model_path = out_dir / f"{args.tag}.pt"
    torch.save(model.state_dict(), model_path)

    # Save scaler params (mean/std)
    scaler_path = out_dir / f"{args.tag}_scaler.npz"
    np.savez(scaler_path, mean=scaler.mean, std=scaler.std)

    # Save a small metadata file
    meta_path = out_dir / f"{args.tag}_meta.txt"
    meta_path.write_text(
        "\n".join([
            f"coords_file={args.coords_file}",
            f"values_file={args.values_file}",
            f"values_key={args.values_key}",
            f"hidden={args.hidden}",
            f"depth={args.depth}",
            f"dropout={args.dropout}",
            f"lr={args.lr}",
            f"epochs={args.epochs}",
            f"batch_size={args.batch_size}",
            f"seed={args.seed}",
            f"device={args.device}",
        ])
    )

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved meta: {meta_path}")

if __name__ == "__main__":
    main()
