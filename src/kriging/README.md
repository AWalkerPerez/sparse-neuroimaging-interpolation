# Kriging

This folder contains the **Kriging baseline** used in the project for interpolating sparse neuroimaging measurements in 3D MNI space.

We use **Ordinary Kriging (3D)** and compare three **variogram models**:
- **Exponential**
- **Gaussian**
- **Spherical**

## Why these variograms?

In Ordinary Kriging, the **variogram model** controls how similarity between points decays with distance, which strongly affects smoothness and how local/global the interpolation becomes.

- **Exponential**: correlation decays quickly with distance (often produces less-smooth, more local behaviour)
- **Gaussian**: very smooth near the origin (strong short-range smoothness)
- **Spherical**: increases then levels off at a finite “range” (assumes points beyond some distance are effectively uncorrelated)

Comparing these three gives a simple but informative baseline for how sensitive Kriging performance is to spatial correlation assumptions.

The goal is to evaluate how classical geostatistical interpolation performs compared with learning-based methods (MLP / U-Net) on:
- **sEEG frequency-band power** (Delta, Theta, Alpha, Beta, Gamma)
- **Gene expression** (per-gene values)

---

## Inputs

Kriging expects:
- `coords`: `(N, 3)` array of MNI coordinates (x, y, z)
- `values`: `(N,)` array of scalar measurements at those coordinates

See `data/README.md` for how the two datasets are stored and loaded.

---

## How evaluation works

We run **K-Fold cross-validation** on the sparse points:
1. Split the points into train/test folds
2. Fit Ordinary Kriging on the training points
3. Predict values at the held-out test coordinates
4. Report fold-averaged metrics

Metrics reported:
- **MSE**, **MAE**, **RMSE**
- **R²**
- **Pearson correlation (R)**
- **Runtime** (fit + predict per fold)

---

## Run Kriging cross-validation

### sEEG example (Delta band)

```bash
python src/kriging/scripts/run_cv.py \
  --coords_file data/seeg/coords.npy \
  --values_file data/seeg/band_powers.npy \
  --values_key Delta \
  --tag seeg_delta
```
### Gene expression example (GRIA1)
```bash
Copy code
python src/kriging/scripts/run_cv.py \
  --coords_file data/expression/expression_coords.npy \
  --values_file data/expression/expression.npy \
  --values_key GRIA1 \
  --tag gene_GRIA1
```
---
## Outputs
The script writes results to:
- results/tables/kriging_cv_summary_<tag>.csv: 
  Mean metrics per variogram model (sorted by MSE)
- results/tables/kriging_cv_folds_<tag>.csv: 
  Per-fold metrics for each variogram model

---

## Notes
- band_powers.npy and expression.npy are saved as Python dicts inside .npy files, so loading uses:
np.load(..., allow_pickle=True).item()

- Supported variograms in this baseline are: exponential, gaussian, spherical

