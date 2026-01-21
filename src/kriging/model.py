from pykrige.ok3d import OrdinaryKriging3D

def ordinary_kriging_3d(train_coords, train_values, variogram_model: str):
  return OrdinaryKriging3D(
    train_coords[:, 0], train_coords[:, 1], train_coords[:, 2],
    train_values,
    variogram_model=variogram_model,
    verbose=False,
    enable_plotting=False,
  )
