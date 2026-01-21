# sEEG band powers

Files:
- `coords.npy`: (N, 3) MNI coordinates
- `band_powers.npy`: dict with {Delta, Theta, Alpha, Beta, Gamma} → (N,)

Notes:
- `band_powers.npy` is saved as a dict inside a `.npy` so you must load with:
  `np.load(..., allow_pickle=True).item()`
