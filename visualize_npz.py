import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load the .npz file ===
data = np.load('single_dct_error_patches.npz')
patches = data['patches']  # Shape: (N, H, W)

# === Step 2: Number of patches to visualize ===
n_patches = 10

# === Step 3: Plot the patches ===
fig, axs = plt.subplots(1, n_patches, figsize=(n_patches * 2.5, 2.5))

for i in range(n_patches):
    axs[i].imshow(patches[i], cmap='plasma', vmin=-5, vmax=5)
    axs[i].axis('off')
    axs[i].set_title(f'Patch {i+1}')

fig.suptitle("Error Patches from .npz File", fontsize=16)
plt.tight_layout()
plt.show()
