# **Sparse Neuroimaging Interpolation & In-Painting**

## **sEEG and Gene Expression Data**

This repository presents a comparative study of **statistical and machine learning methods** for interpolating and in-painting **sparse neuroimaging data**, focusing on two challenging modalities:

**- Stereo-EEG (sEEG)** intracranial recordings

**- Gene expression** samples from post-mortem brain tissue

The work benchmarks classical and modern approaches, evaluating not only numerical accuracy but also **spatial plausibility, interpretability, and computational efficiency**.

---

## **Project Overview**

Sparse neuroimaging datasets provide valuable functional and molecular insights but suffer from limited spatial coverage. Interpolation is therefore required to reconstruct dense representations that support analysis, visualisation, and downstream clinical or neuroscientific applications.

This project evaluates and compares:

**- Kriging** (Gaussian, Spherical, Exponential variograms)

**- Residual Multi-Layer Perceptron (MLP)**

**- 3D U-Net** (encoder–decoder with attention)

**- Classical baselines** (Spline, KNN, Linear interpolation)

A key outcome of this study is that **simpler classical methods can outperform deep learning models** in sparse neuroimaging settings, particularly when anatomical plausibility and robustness are prioritised over raw numerical metrics.

## **Key Findings**

**3D U-Net**

  - Achieved the highest R² and fastest runtime

  - Produced overly smooth outputs that often lacked anatomical realism

**Kriging and Residual MLP**

  - Lower numerical scores but more spatially interpretable reconstructions

  - Higher computational cost, especially for Kriging

**Spline interpolation**

  - Outperformed all advanced methods on gene expression data

  - Competitive performance on sEEG despite minimal model complexity

These results highlight the importance of benchmarking advanced models against strong classical baselines and evaluating spatial fidelity alongside quantitative metrics.

---

# **Methods**
## **Datasets**

**- sEEG:** Intracranial EEG atlas with electrode coordinates in MNI space and band power features (Delta–Gamma)

**- Gene Expression:** Allen Human Brain Atlas microarray samples mapped to MNI coordinates

## **Models**

**- Kriging:** 3D Ordinary Kriging using PyKrige with multiple variogram models

**- Residual MLP:** Coordinate-based regression with polynomial feature expansion, batch normalisation, dropout, and residual connections

**- 3D U-Net:** Volumetric in-painting architecture with encoder–decoder structure, residual bottlenecks, and attention modules (CBAM)

## **Evaluation**

- 10-fold cross-validation
- Data augmentation:
    - Spatial jitter
    - Gaussian noise
    - Jitter + noise
- Metrics:
    - R²
    - MSE
    - MAE
    - RMSE
- Runtime and computational cost recorded for all methods

---

## **Repository Structure**

sparse-neuroimaging-interpolation/
├── src/ Model implementations (Kriging, MLP, U-Net, evaluation)
├── experiments/ Experiment runners and configuration files
├── results/ Exported figures and tables
├── data/ Dataset instructions (no raw data included)
├── report/ Full project report (PDF)
├── requirements.txt
└── README.md

--- 

## **Report**

**Full project report:**
Evaluating Interpolation and In-Painting Techniques for Sparse Neuroimaging Data Using Statistical and Machine Learning Approaches (PDF)

The report includes:
- Full methodological details
- Quantitative results
- Qualitative 3D visualisations
- Comparison against classical interpolation baselines
- Discussion of interpretability and clinical relevance

---

## **Getting Started**
**Installation**

pip install -r requirements.txt

**Running Experiments**

python experiments/run_experiments.py --config experiments/configs/example.yaml

**Note:** Raw datasets are not included. See data/README.md for data sources and expected formats.

---

## **Tech Stack**
- Python
- NumPy, SciPy, Pandas
- PyTorch
- PyKrige
- Matplotlib
- scikit-learn
- MONAI (for 3D pipelines, where applicable)

---

## **Author**

**Andrea Walker Perez**
MSc Healthcare Technologies (AI & Medical Robotics)
King’s College London

---

## **Notes**

This repository is intended as a **research and portfolio project**.
It prioritises clarity, reproducibility, and honest evaluation over model complexity.
