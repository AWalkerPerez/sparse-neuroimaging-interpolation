from __future__ import annotations
import argparse
import time
import numpy as np
from sklearn.model_selection import KFold

from src.utils.io import load_npy, load_npy_dict
from src.utils.seed import seed_everything
from src.evaluation.metrics import regression_metrics
from src.evaluation.reporting import save_cv_outputs

from src.unet.volume import build_volume_from_points, make_input_from_available_mask
from src.unet.train import TrainConfig, fit_unet


def main():
    p = argparse.ArgumentParser(description="U-Net 3D CV on known voxels (no leakage).")
    p.add_argument("--coords_file", required=True)
    p.add_argument("--values_file", required=True)
    p.add_argument("--values_key", default=None)
    p.add_argument("--tag", default="unet")

    p.add_argument("--splits", type=int, default=5)   # U-Net CV is heavy; default 5
    p.add_argument("--seed", type=int, default=42)

    # volume mapping
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--jitter_std", type=float, default=0.0)

    # training
    p.add_argument("--device", default="cpu")
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hide_frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--tv_alpha", type=float, default=0.0)

    args = p.parse_args()
    seed_everything(args.seed)

    coords = np.asarray(load_npy(args.coords_file), dtype=float)
    if args.values_key is None:
        values = np.asarray(load_npy(args.values_file), dtype=float).reshape(-1)
    else:
        d = load_npy_dict(args.values_file)
        values = np.asarray(d[args.values_key], dtype=float).reshape(-1)

    pack = build_volume_from_points(coords, values, step=args.step, jitter_std=args.jitter_std, seed=args.seed)
    vol_std = pack.volume_std
    known_mask = pack.known_mask.astype(bool)

    known_idx = np.argwhere(known_mask)  # (K,3)
    if known_idx.shape[0] < args.splits:
        raise ValueError("Not enough known voxels for the chosen number of splits.")

    kf = KFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    fold_rows = []
    t_start = time.time()

    for fold, (tr, va) in enumerate(kf.split(known_idx), start=1):
        train_known = np.zeros_like(known_mask, dtype=np.float32)
        val_known = np.zeros_like(known_mask, dtype=np.float32)

        tr_idx = known_idx[tr]
        va_idx = known_idx[va]
        train_known[tr_idx[:, 0], tr_idx[:, 1], tr_idx[:, 2]] = 1.0
        val_known[va_idx[:, 0], va_idx[:, 1], va_idx[:, 2]] = 1.0

        cfg = TrainConfig(
            base=args.base,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed + fold,   # vary fold seed slightly
            hide_frac=args.hide_frac,
            tv_alpha=args.tv_alpha,
            patience=args.patience,
        )

        t0 = time.time()
        model = fit_unet(vol_std, train_known, val_known, cfg)
        train_time = time.time() - t0

        # Predict with ONLY TRAIN known shown in input; score on VAL known
        x_np = make_input_from_available_mask(vol_std, train_known)
        import torch
        x = torch.tensor(x_np, dtype=torch.float32, device=cfg.device).unsqueeze(0)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy().squeeze()  # (D,H,W)

        y_true = vol_std[val_known.astype(bool)].reshape(-1)
        y_pred = pred[val_known.astype(bool)].reshape(-1)

        m = regression_metrics(y_true, y_pred)
        fold_rows.append(m | {
            "fold": fold,
            "n_train_vox": int(train_known.sum()),
            "n_val_vox": int(val_known.sum()),
            "train_time_s": float(train_time),
        })

        print(f"Fold {fold}/{args.splits} done. MSE={m['MSE']:.6g}, R2={m['R2']:.4g}")

    # Summary
    def mean_key(k):
        arr = np.array([r[k] for r in fold_rows], dtype=float)
        return float(np.nanmean(arr))

    summary_rows = [{
        "method": "unet3d",
        "values_key": args.values_key,
        "splits": args.splits,
        "MSE_mean": mean_key("MSE"),
        "MAE_mean": mean_key("MAE"),
        "RMSE_mean": mean_key("RMSE"),
        "R2_mean": mean_key("R2"),
        "PearsonR_mean": mean_key("PearsonR"),
        "train_time_mean_s": mean_key("train_time_s"),
        "train_time_total_s": float(np.nansum([r["train_time_s"] for r in fold_rows])),
        "wall_time_total_s": float(time.time() - t_start),

        # config snapshot
        "step": args.step,
        "jitter_std": args.jitter_std,
        "base": args.base,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hide_frac": args.hide_frac,
        "tv_alpha": args.tv_alpha,
        "patience": args.patience,
        "device": args.device,
        "seed": args.seed,
    }]

    save_cv_outputs(fold_rows, summary_rows, tag=f"unet_{args.tag}")
    print("Saved U-Net CV outputs to results/tables.")
    print("Summary:", summary_rows[0])


if __name__ == "__main__":
    main()

