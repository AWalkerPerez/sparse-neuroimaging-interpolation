from __future__ import annotations
import argparse
import numpy as np

from src.utils.io import load_npy, load_npy_dict
from src.utils.seed import seed_everything

from src.evaluation.cv import cross_validate
from src.evaluation.reporting import save_cv_outputs

from src.mlp.train import TrainConfig
from src.mlp.predict import mlp_predict


def main():
    p = argparse.ArgumentParser(description="MLP KFold CV (trains per fold).")
    p.add_argument("--coords_file", required=True)
    p.add_argument("--values_file", required=True)
    p.add_argument("--values_key", default=None)

    p.add_argument("--splits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="mlp")

    p.add_argument("--device", type=str, default="cpu")

    # model/training
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # augmentation
    p.add_argument("--aug", type=str, default="none", choices=["none", "jitter", "noise", "jitter_noise"])
    p.add_argument("--jitter_std", type=float, default=0.5)
    p.add_argument("--noise_std", type=float, default=0.01)

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
        weight_decay=args.weight_decay,
        aug=args.aug,
        jitter_std=args.jitter_std,
        noise_std=args.noise_std,
        device=args.device,
        seed=args.seed,
    )

    def predict_fn(tc, tv, qc):
        return mlp_predict(tc, tv, qc, cfg)

    fold_rows, summary_rows = cross_validate(
        coords, values,
        predict_fn=predict_fn,
        n_splits=args.splits,
        seed=args.seed,
    )

    # add metadata to summary
    summary_rows[0].update({
        "method": "mlp",
        "values_key": args.values_key,
        "hidden": args.hidden,
        "depth": args.depth,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "aug": args.aug,
        "jitter_std": args.jitter_std,
        "noise_std": args.noise_std,
        "device": args.device,
        "seed": args.seed,
    })

    save_cv_outputs(fold_rows, summary_rows, tag=f"mlp_{args.tag}")

    print("Saved MLP CV outputs to results/tables.")
    print("Summary:", summary_rows[0])


if __name__ == "__main__":
    main()

