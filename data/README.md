# Data

This project uses two datasets stored in the same MNI coordinate space:

## 1) sEEG frequency-band powers
Location: `data/seeg/`

Files:
- `coords.npy` (float, shape Nx3): electrode/sample coordinates in MNI (x,y,z)
- `band_powers.npy` (dict saved as .npy): keys = {Delta, Theta, Alpha, Beta, Gamma}, each value shape (N,)

Load example:
```python
import numpy as np
coords = np.load("data/seeg/coords.npy")
band_powers = np.load("data/seeg/band_powers.npy", allow_pickle=True).item()
y = band_powers["Delta"]

## 2) Gene expression data
Location: `data/expression/`

Files:
- `expression_coords.npy` (float, shape Mx3): sample coordinates in MNI (x,y,z)
- `expression.npy` (dict saved as .npy): keys = gene names (e.g., 'GRIA1'), each value shape (M,)

Load example:
```import numpy as np
coords = np.load("data/expression/expression_coords.npy")
expr = np.load("data/expression/expression.npy", allow_pickle=True).item()
y = expr["GRIA1"]


