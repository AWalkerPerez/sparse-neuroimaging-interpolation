from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class VolumePack:
    # Standardised full volume (known voxels standardised; unknown left as 0)
    volume_std: np.ndarray          # (D, H, W)
    known_mask: np.ndarray          # (D, H, W) float32 {0,1}
    vol_shape: tuple[int, int, int]
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    scaler: StandardScaler          # fitted on known voxels


def build_volume_from_points(
    coords: np.ndarray,
    values: np.ndarray,
    step: int = 4,
    jitter_std: float = 0.0,
    seed: int = 42,
) -> VolumePack:
    """
    Map sparse (x,y,z)->value points to a regular 3D grid volume.
    Multiple points mapping to same voxel are averaged.

    Returns:
      volume_std: standardised values on known voxels
      known_mask: 1 where at least one point mapped into voxel
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    assert coords.shape[0] == values.shape[0]
    assert coords.shape[1] == 3

    rng = np.random.default_rng(seed)
    coords_aug = coords.copy()
    if jitter_std and jitter_std > 0:
        coords_aug = coords_aug + rng.normal(0, jitter_std, size=coords_aug.shape)

    raw_min = np.floor(coords_aug.min(axis=0)).astype(int)
    raw_max = np.ceil(coords_aug.max(axis=0)).astype(int)

    grid_x = np.arange(raw_min[0], raw_max[0] + 1, step)
    grid_y = np.arange(raw_min[1], raw_max[1] + 1, step)
    grid_z = np.arange(raw_min[2], raw_max[2] + 1, step)

    D, H, W = len(grid_x), len(grid_y), len(grid_z)
    vol = np.zeros((D, H, W), dtype=np.float32)
    mask = np.zeros((D, H, W), dtype=np.float32)
    counts = np.zeros((D, H, W), dtype=np.int32)

    for pt, val in zip(coords_aug, values):
        ix = int(round((pt[0] - raw_min[0]) / step))
        iy = int(round((pt[1] - raw_min[1]) / step))
        iz = int(round((pt[2] - raw_min[2]) / step))
        if 0 <= ix < D and 0 <= iy < H and 0 <= iz < W:
            vol[ix, iy, iz] += float(val)
            counts[ix, iy, iz] += 1
            mask[ix, iy, iz] = 1.0

    nz = counts > 0
    vol[nz] /= counts[nz]

    # Standardise only the known voxels
    scaler = StandardScaler()
    known_vals = vol[mask == 1].reshape(-1, 1)
    scaler.fit(known_vals)

    vol_std = vol.copy()
    vol_std[mask == 1] = scaler.transform(vol[mask == 1].reshape(-1, 1)).reshape(-1)
    # unknown voxels stay at 0

    return VolumePack(
        volume_std=vol_std,
        known_mask=mask,
        vol_shape=(D, H, W),
        grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
        scaler=scaler,
    )


def make_input_from_available_mask(
    volume_std: np.ndarray,
    available_mask: np.ndarray,
) -> np.ndarray:
    """
    Build a 2-channel input volume:
      channel 0: value * available_mask
      channel 1: available_mask
    """
    available_mask = available_mask.astype(np.float32)
    v = volume_std.astype(np.float32) * available_mask
    x = np.stack([v, available_mask], axis=0)  # (2, D, H, W)
    return x

