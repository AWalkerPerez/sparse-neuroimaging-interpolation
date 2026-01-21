import numpy as np
from .model import ordinary_kriging_3d

def kriging_predict(
  train_coords: np.ndarray,
  train_values: np.ndarray,
  query_coords: np.ndarray,
  variogram_model: str = "exponential",
) -> np.ndarray:
  """
  Ordinary Kriging (3D) prediction for query points.
  variogram_model: "exponential" | "gaussian" | "spherical"
  """
  ok = ordinary_kriging_3d(train_coords, train_values, variogram_model)
  preds, _ = ok.execute("points", query_coords[:, 0], query_coords[:, 1], query_coords[:, 2])
  return np.asarray(preds)
