from __future__ import annotations
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Simple coordinate-based MLP regressor: (x,y,z) -> value
    """
    def __init__(self, in_dim: int = 3, hidden: int = 128, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim

        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

