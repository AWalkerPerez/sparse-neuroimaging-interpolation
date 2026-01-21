# U-Net (3D)

This folder contains a 3D U-Net baseline for sparse neuroimaging interpolation on a voxel grid.

## Key idea

We first map sparse samples `(x,y,z)->value` to a regular 3D grid (with a chosen voxel step).
The network input has 2 channels:
- **value channel**: known voxel values (unknown set to 0)
- **mask channel**: 1 where known, 0 where unknown

To avoid leakage and make training meaningful with a single volume, we use **self-supervised masking**:
during training, we randomly hide a fraction of known voxels and train the U-Net to predict them.

---

## Cross-validation (no leakage)

```bash
python src/unet/scripts/run_cv.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta \
  --device cpu \
  --step 4 \
  --splits 5
```
Outputs:
- `results/tables/unet_<tag>_cv_summary.csv`
- `results/tables/unet_<tag>_cv_folds.csv`
---
## Train once + save checkpoint
```bash
python src/unet/scripts/train_full.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta \
  --device cpu \
  --step 4
```
Saves:
- `results/models/unet/seeg_delta.pt`
- `results/models/unet/seeg_delta_scaler.joblib`
---
## Dense prediction + slice plots
```bash
python src/unet/scripts/run_dense.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --model_path results/models/unet/seeg_delta.pt \
  --scaler_path results/models/unet/seeg_delta_scaler.joblib \
  --tag unet_seeg_delta \
  --axis z --n_slices 6
```
Outputs:
- `results/figures/unet_seeg_delta_pred_slices_z.png`
