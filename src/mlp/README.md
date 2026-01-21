# MLP

This folder contains a coordinate-based **MLP regressor** baseline for sparse neuroimaging interpolation.

The model learns a function:

`(x, y, z) -> value`

where `(x, y, z)` are MNI coordinates and `value` is either:
- sEEG band power (Delta/Theta/Alpha/Beta/Gamma), or
- gene expression (per gene)

---

## Inputs
- `coords`: `(N, 3)` MNI coordinates
- `values`: `(N,)` scalar values at those coordinates

See `data/README.md` for file formats and loading examples.

---

## Evaluation (CV)

Evaluation uses K-Fold CV. For each fold, the MLP is **trained on the training split** and evaluated on held-out points.

Metrics: MSE, MAE, RMSE, R², Pearson R, runtime.

### sEEG example (Delta)
```bash
python src/mlp/scripts/run_cv.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta \
  --device cpu
```
### Gene example (GRIA1)
```bash
Copy code
python src/mlp/scripts/run_cv.py \
  --coords_file data/expression/expression_coords.npy \
  --values_file data/expression/expression.npy \
  --values_key GRIA1 \
  --tag gene_GRIA1 \
  --device cpu
```
---
## Train once on all data (optional)
This trains an MLP on the full dataset and saves a checkpoint:
- results/models/mlp/<tag>.pt
- results/models/mlp/<tag>_scaler.npz
```bash
Copy code
python src/mlp/scripts/train_full.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta \
  --device cpu
```
---
## Augmentation options
You can augment training data using:
- `--aug none`
- `--aug jitter` (adds Gaussian noise to coordinates)
- `--aug noise` (adds Gaussian noise to target values)
- `--aug jitter_noise` (both)

Control strength with:
- `--jitter_std`
- `--noise_std`
---

## Example commands you’ll actually run

**CV + jitter augmentation**
```bash
python src/mlp/scripts/run_cv.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta_jitter \
  --aug jitter --jitter_std 0.5
```
**CV + noise augmentation**
```bash
python src/mlp/scripts/run_cv.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta_noise \
  --aug noise --noise_std 0.01
```
