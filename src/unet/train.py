from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from .model import UNet3D
from .losses import hybrid_loss
from .volume import make_input_from_available_mask


@dataclass
class TrainConfig:
    base: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    device: str = "cpu"
    seed: int = 42

    # self-supervised masking on TRAIN known voxels
    hide_frac: float = 0.2  # fraction of train-known voxels to hide each epoch

    # loss
    l1_alpha: float = 1.0
    tv_alpha: float = 0.0

    # scheduler/early stop
    patience: int = 25


def fit_unet(
    volume_std: np.ndarray,
    train_known_mask: np.ndarray,
    val_known_mask: np.ndarray | None,
    cfg: TrainConfig,
) -> UNet3D:
    """
    Self-supervised training:
      - We only 'know' train_known voxels.
      - Each epoch we randomly hide a subset of train_known voxels and train to predict them.
      - Validation (if provided) is computed on val_known voxels that are never shown in input.
    """
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device
    model = UNet3D(in_channels=2, out_channels=1, base=cfg.base).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    vol = torch.tensor(volume_std, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    train_known = torch.tensor(train_known_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    val_known = None
    if val_known_mask is not None:
        val_known = torch.tensor(val_known_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    best = float("inf")
    best_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    rng = np.random.default_rng(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        # Hide a random subset of TRAIN known voxels
        train_known_np = train_known_mask.astype(bool)
        idx = np.argwhere(train_known_np)
        if idx.shape[0] == 0:
            raise ValueError("train_known_mask has no known voxels.")

        hide_n = max(1, int(cfg.hide_frac * idx.shape[0]))
        hide_sel = idx[rng.choice(idx.shape[0], size=hide_n, replace=False)]

        hide_mask_np = np.zeros_like(train_known_np, dtype=np.float32)
        hide_mask_np[hide_sel[:, 0], hide_sel[:, 1], hide_sel[:, 2]] = 1.0

        available_np = train_known_mask.astype(np.float32) - hide_mask_np
        available_np = np.clip(available_np, 0.0, 1.0)

        x_np = make_input_from_available_mask(volume_std, available_np)  # (2,D,H,W)
        x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,2,D,H,W)

        hide_mask = torch.tensor(hide_mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        pred = model(x)
        loss, l1v, tvv = hybrid_loss(pred, vol, hide_mask, l1_alpha=cfg.l1_alpha, tv_alpha=cfg.tv_alpha)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Validation loss: predict with only TRAIN known available, score on VAL known
        model.eval()
        with torch.no_grad():
            if val_known is not None:
                x_full_np = make_input_from_available_mask(volume_std, train_known_mask.astype(np.float32))
                x_full = torch.tensor(x_full_np, dtype=torch.float32, device=device).unsqueeze(0)
                pred_full = model(x_full)
                vloss, _, _ = hybrid_loss(pred_full, vol, val_known, l1_alpha=cfg.l1_alpha, tv_alpha=0.0)
                val_loss = float(vloss.item())
            else:
                val_loss = float(loss.item())

        scheduler.step(val_loss)

        if val_loss < best:
            best = val_loss
            best_wts = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            break

    model.load_state_dict({k: v.to(device) for k, v in best_wts.items()})
    return model


def save_unet(model: UNet3D, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_unet(path: str | Path, cfg: TrainConfig) -> UNet3D:
    model = UNet3D(in_channels=2, out_channels=1, base=cfg.base).to(cfg.device)
    state = torch.load(path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()
    return model

