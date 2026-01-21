from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import MLPRegressor
from .data import StandardScaler, apply_augmentation


@dataclass
class TrainConfig:
    # model
    hidden: int = 128
    depth: int = 4
    dropout: float = 0.0

    # optimizer/training
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 128
    weight_decay: float = 0.0

    # augmentation (train only)
    aug: str = "none"          # none | jitter | noise | jitter_noise
    jitter_std: float = 0.5    # coordinate jitter std (MNI units)
    noise_std: float = 0.01    # target noise std

    # misc
    device: str = "cpu"
    seed: int = 42


def fit_mlp(
    train_coords: np.ndarray,
    train_values: np.ndarray,
    cfg: TrainConfig,
) -> tuple[MLPRegressor, StandardScaler]:
    """
    Train on train_coords/train_values and return (model, scaler).
    """
    # Augment ONLY training data
    aug_coords, aug_values = apply_augmentation(
        train_coords, train_values,
        aug=cfg.aug,
        jitter_std=cfg.jitter_std,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )

    # Scale coords for stable training
    scaler = StandardScaler()
    X = scaler.fit_transform(aug_coords)
    y = np.asarray(aug_values, dtype=float).reshape(-1)

    # Torch tensors
    device = cfg.device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Reproducibility
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    model = MLPRegressor(in_dim=3, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return model, scaler


def load_mlp(model_path: str, scaler_path: str, cfg: TrainConfig) -> tuple[MLPRegressor, StandardScaler]:
    """
    Load a trained model + scaler saved by train_full.py
    """
    model = MLPRegressor(in_dim=3, hidden=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(cfg.device)
    state = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    z = np.load(scaler_path)
    scaler = StandardScaler()
    scaler.mean = z["mean"]
    scaler.std = z["std"]
    return model, scaler

