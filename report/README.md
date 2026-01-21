# Report

This folder contains the final written report for this project:

- `Evaluating_Interpolation_and_In-Painting_Techniques_for_Sparse_Neuroimaging_Data_Using_Statistical_and_Machine_Learning_Approaches.pdf`

## What’s inside
The report evaluates interpolation / in-painting methods for sparse neuroimaging data in MNI space, focusing on:
- **Kriging** (Gaussian / Spherical / Exponential variograms)
- **Residual MLP**
- **3D U-Net**
across two datasets (sEEG band powers and Allen Human Brain Atlas gene expression). 

## How it maps to this repo
- Code implementations live in `src/kriging/`, `src/mlp/`, and `src/unet/`
- Shared evaluation + plotting utilities live in `src/evaluation/` and `src/utils/`
- Output tables and figures are written to `results/tables/` and `results/figures/`

## Citation
If you use this work, please cite the report (PDF above) and reference this repository.
