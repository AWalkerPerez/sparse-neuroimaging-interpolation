from __future__ import annotations
import numpy as np
import torch

from .train import TrainConfig, fit_mlp


def mlp_predict(
    train_coords: np.ndarray,
    train_values: np.ndarray,
    query_coords: np.ndarray,
    cfg: TrainConfig,
) -> np.ndarray:
    """
    Train an MLP on (train_coords, train_values) then predict at query_coords.
    Used for CV (train-per-fold).
    """
    model, scaler = fit_mlp(train_coords, train_values, cfg)

    Xq = scaler.transform(query_coords)
    Xq_t = torch.tensor(Xq, dtype=torch.float32, device=cfg.device)

    model.eval()
    with torch.no_grad():
        preds = model(Xq_t).detach().cpu().numpy()

    return preds.reshape(-1)

