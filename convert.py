
import numpy as np
from scipy.io import loadmat
import sys
import os

# === Check that at least one file is passed ===
if len(sys.argv) < 2:
    print("Usage: python convert_mat_to_npz.py file1.mat file2.mat ...")
    sys.exit(1)

# === Process each .mat file ===
for mat_path in sys.argv[1:]:
    try:
        mat_data = loadmat(mat_path)
        print(mat_data)
        if 'single_error' not in mat_data:
            print(f"Key 'single_error' not found in {mat_path}, skipping.")
            continue
        error_raw = mat_data['single_error'].squeeze()

        def extract_patch(patch_raw):
            if isinstance(patch_raw, np.ndarray):
                return patch_raw.squeeze().astype(np.float32)
            else:
                raise TypeError(f"Unexpected type for patch in {mat_path}: {type(patch_raw)}")

        patch_list = [extract_patch(p) for p in error_raw]
        patch_array = np.stack(patch_list)

        npz_path = os.path.splitext(mat_path)[0] + ".npz"
        np.savez_compressed(npz_path, patches=patch_array)

        print(f"Saved {patch_array.shape[0]} patches with shape {patch_array.shape[1:]} to '{npz_path}'")

    except Exception as e:
        print(f"Failed to process {mat_path}: {e}")
