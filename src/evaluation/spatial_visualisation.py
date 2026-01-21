# src/evaluation/spatial_visualisation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import griddata


@dataclass
class HullGrid:
    """Container for a 3D grid masked to points inside the convex hull."""
    Xi: np.ndarray                 # (res, res, res)
    Yi: np.ndarray                 # (res, res, res)
    Zi: np.ndarray                 # (res, res, res)
    inside_mask_ravel: np.ndarray  # (res^3,) boolean mask for inside hull (on raveled grid)
    grid_points_inside: np.ndarray # (M, 3) points inside hull


def build_grid_within_convex_hull(
    coords: np.ndarray,
    res: int = 30,
    margin: float = 0.0,
) -> HullGrid:
    """
    Builds a regular 3D grid spanning coords bounds, then keeps only points inside
    the convex hull of coords (using Delaunay over hull vertices).

    coords: (N, 3)
    res: grid resolution per axis
    margin: expands bounds by +/- margin (in same units as coords)
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be shape (N,3), got {coords.shape}")

    # Convex hull + Delaunay triangulation (on hull vertices)
    hull = ConvexHull(coords)
    delaunay = Delaunay(coords[hull.vertices])

    # Grid bounds
    xmin, ymin, zmin = coords.min(axis=0) - margin
    xmax, ymax, zmax = coords.max(axis=0) + margin

    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    zi = np.linspace(zmin, zmax, res)

    Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing="ij")

    all_grid_pts = np.vstack([Xi.ravel(), Yi.ravel(), Zi.ravel()]).T
    inside_mask = delaunay.find_simplex(all_grid_pts) >= 0
    grid_pts_inside = all_grid_pts[inside_mask]

    return HullGrid(
        Xi=Xi, Yi=Yi, Zi=Zi,
        inside_mask_ravel=inside_mask,
        grid_points_inside=grid_pts_inside,
    )


def points_inside_to_volume(
    preds_inside: np.ndarray,
    hull_grid: HullGrid,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Convert predictions on grid_points_inside back to a (res,res,res) volume.
    Points outside hull are fill_value (default NaN).
    """
    preds_inside = np.asarray(preds_inside, dtype=float).reshape(-1)
    if preds_inside.shape[0] != hull_grid.grid_points_inside.shape[0]:
        raise ValueError(
            f"preds_inside length {preds_inside.shape[0]} does not match "
            f"grid_points_inside {hull_grid.grid_points_inside.shape[0]}"
        )

    full = np.full(hull_grid.inside_mask_ravel.shape[0], fill_value, dtype=float)
    full[hull_grid.inside_mask_ravel] = preds_inside
    return full.reshape(hull_grid.Xi.shape)


def linear_griddata_with_nearest_fill(
    coords: np.ndarray,
    values: np.ndarray,
    hull_grid: HullGrid,
) -> np.ndarray:
    """
    Useful baseline: linear griddata on the full grid (not only inside points),
    then fills NaNs via nearest-neighbour griddata.

    Returns volume shaped like hull_grid.Xi.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)

    V = griddata(coords, values, (hull_grid.Xi, hull_grid.Yi, hull_grid.Zi), method="linear")
    nanmask = np.isnan(V)
    if nanmask.any():
        V[nanmask] = griddata(
            coords, values,
            (hull_grid.Xi[nanmask], hull_grid.Yi[nanmask], hull_grid.Zi[nanmask]),
            method="nearest",
        )
    return V


def plot_volume_slices(
    volume: np.ndarray,
    axis: str = "z",
    n_slices: int = 6,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots evenly spaced slices through a 3D volume.
    volume: (X, Y, Z)
    axis: "x" | "y" | "z"
    """
    vol = np.asarray(volume, dtype=float)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3D, got {vol.shape}")

    axis = axis.lower()
    ax_map = {"x": 0, "y": 1, "z": 2}
    if axis not in ax_map:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    a = ax_map[axis]
    size = vol.shape[a]
    # avoid endpoints to get nicer slices
    idxs = np.linspace(1, size - 2, n_slices).astype(int)

    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    if n_slices == 1:
        axes = [axes]

    for i, idx in enumerate(idxs):
        if a == 0:
            img = vol[idx, :, :]
        elif a == 1:
            img = vol[:, idx, :]
        else:
            img = vol[:, :, idx]

        axes[i].imshow(img)
        axes[i].set_title(f"{axis}={idx}")
        axes[i].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_sparse_points_3d(
    coords: np.ndarray,
    values: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    max_points: int = 5000,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Quick 3D scatter of sparse samples (optionally colored by values).
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[idx]
        if values is not None:
            values = np.asarray(values).reshape(-1)[idx]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    if values is None:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=4)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, s=4)

    if title:
        ax.set_title(title)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()
