from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Load .mat file
data = loadmat('single_dct_error.mat')
dct_error_raw = data['single_dct_error'].squeeze()

def extract_patch(patch_raw):
    """
    Extract a clean (8x8) DCT error patch from the raw data.
    Handles array shapes like (1,1,8,8) or similar.
    """
    if isinstance(patch_raw, np.ndarray):
        return patch_raw.squeeze().astype(np.float32)
    else:
        raise TypeError(f"Unexpected type for DCT patch: {type(patch_raw)}")

# Number of patches to visualize
n_patches = 10
dct_patches = [extract_patch(dct_error_raw[i]) for i in range(n_patches)]

# Create the figure for DCT error visualization
fig, axs = plt.subplots(1, n_patches, figsize=(n_patches * 2.5, 2.5))

for i in range(n_patches):
    axs[i].imshow(dct_patches[i], cmap='plasma', vmin=-5, vmax=5)
    axs[i].set_title(f'Patch {i+1}')
    axs[i].axis('off')

fig.suptitle("DCT Error Patches", fontsize=14)
plt.tight_layout()
plt.show()