from __future__ import annotations
import numpy as np


class StandardScaler:
    """
    Standardize features: (x - mean) / std
    """
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fit yet.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def apply_augmentation(
    coords: np.ndarray,
    values: np.ndarray,
    aug: str = "none",
    jitter_std: float = 0.5,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment training data only.

    aug:
      - "none"
      - "jitter"        : add Gaussian noise to coords
      - "noise"         : add Gaussian noise to values
      - "jitter_noise"  : both
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    rng = np.random.default_rng(seed)

    if aug in ("jitter", "jitter_noise"):
        coords = coords + rng.normal(0.0, jitter_std, size=coords.shape)

    if aug in ("noise", "jitter_noise"):
        values = values + rng.normal(0.0, noise_std, size=values.shape)

    return coords, values

