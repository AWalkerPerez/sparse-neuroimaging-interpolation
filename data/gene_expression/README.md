# Gene expression

Files:
- `expression_coords.npy`: (M, 3) MNI coordinates
- `expression.npy`: dict with gene_name → (M,)

Notes:
- `expression.npy` is saved as a dict inside a `.npy` so you must load with:
  `np.load(..., allow_pickle=True).item()`
