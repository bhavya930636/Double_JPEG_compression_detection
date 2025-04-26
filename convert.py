import numpy as np
from scipy.io import loadmat

# === Step 1: Load the .mat file ===
mat_data = loadmat('single_error.mat')  # Replace with your path if needed
error_raw = mat_data['single_error'].squeeze()  # Shape: (N, 1)

# === Step 2: Extract individual patches ===
def extract_patch(patch_raw):
    if isinstance(patch_raw, np.ndarray):
        return patch_raw.squeeze().astype(np.float32)
    else:
        raise TypeError(f"Unexpected type for patch: {type(patch_raw)}")

patch_list = [extract_patch(p) for p in error_raw]

# === Step 3: Stack into a NumPy array ===
patch_array = np.stack(patch_list)  # Shape: (N, H, W)

# === Step 4: Save to .npz ===
np.savez_compressed('single_error_patches.npz', patches=patch_array)

print(f"Saved {patch_array.shape[0]} patches with shape {patch_array.shape[1:]} to 'error_patches.npz'")
