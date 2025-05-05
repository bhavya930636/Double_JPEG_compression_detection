# import numpy as np
# import matplotlib.pyplot as plt

# # Load with allow_pickle=True
# data = np.load('single_dct_error.npz', allow_pickle=True)
# print(data)
# dct_error_raw = data['single_dct_error']

# # Patch extractor
# def extract_patch(patch_raw):
#     # Go deep into the nested structure
#     patch = patch_raw[0]  # From shape (1,) -> scalar
#     return patch.squeeze().astype(np.float32)  # Now (8,8)

# # Number of patches to visualize
# n_patches = 10
# dct_patches = [extract_patch(dct_error_raw[i]) for i in range(n_patches)]

# # Plot
# fig, axs = plt.subplots(1, n_patches, figsize=(n_patches * 2.5, 2.5))
# for i in range(n_patches):
#     axs[i].imshow(dct_patches[i], cmap='plasma', vmin=-5, vmax=5)
#     axs[i].set_title(f'Patch {i+1}')
#     axs[i].axis('off')

# fig.suptitle("DCT Error Patches (from NPZ)", fontsize=14)
# plt.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Load with allow_pickle=True
# data = np.load('single_dct_error.npz', allow_pickle=True)
# dct_error_raw = data['single_dct_error']
# round_raw = data['round']
# trunc_raw = data['trunc']

# # Patch extractor for dct_error
# def extract_patch(patch_raw):
#     patch = patch_raw[0]  # From shape (1,) -> scalar
#     return patch.squeeze().astype(np.float32)

# # Visualizing first 10 patches from dct_error_raw
# n_patches = 10
# dct_patches = [extract_patch(dct_error_raw[i]) for i in range(n_patches)]

# # Create subplots for visualizing patches
# fig, axs = plt.subplots(1, n_patches, figsize=(n_patches * 2.5, 2.5))
# for i in range(n_patches):
#     axs[i].imshow(dct_patches[i], cmap='plasma', vmin=-5, vmax=5)
#     axs[i].set_title(f'Patch {i+1}')
#     axs[i].axis('off')

# fig.suptitle("DCT Error Patches", fontsize=14)
# plt.tight_layout()
# plt.show()

# # Visualizing round_raw and trunc_raw (both 1D arrays)
# fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# # Round plot
# axs[0].plot(round_raw.flatten(), color='blue', label='Round')
# axs[0].set_title("Round Values")
# axs[0].set_xlabel("Index")
# axs[0].set_ylabel("Round Value")
# axs[0].legend()

# # Trunc plot
# axs[1].plot(trunc_raw.flatten(), color='green', label='Trunc')
# axs[1].set_title("Trunc Values")
# axs[1].set_xlabel("Index")
# axs[1].set_ylabel("Trunc Value")
# axs[1].legend()

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('single_dct_error.npz', allow_pickle=True)
dct_error_raw = data['single_dct_error']
round_raw = data['round']
trunc_raw = data['trunc']

# Extract an 8x8 patch from a nested structure
def extract_patch(patch_raw):
    patch = patch_raw[0]
    return patch.squeeze().astype(np.float32)

# --- 1. Visualize DCT Error Patches ---
n_patches = 10
dct_patches = [extract_patch(dct_error_raw[i]) for i in range(n_patches)]

fig, axs = plt.subplots(1, n_patches, figsize=(n_patches * 2.5, 2.5))
for i in range(n_patches):
    axs[i].imshow(dct_patches[i], cmap='plasma', vmin=-5, vmax=5)
    axs[i].set_title(f'Patch {i+1}')
    axs[i].axis('off')

fig.suptitle("DCT Error Patches", fontsize=14)
plt.tight_layout()
plt.show()


# --- 3. Show Unique Value Counts ---
round_vals, round_counts = np.unique(round_raw.flatten(), return_counts=True)
trunc_vals, trunc_counts = np.unique(trunc_raw.flatten(), return_counts=True)

print("Unique values and counts in 'round_raw':")
for v, c in zip(round_vals, round_counts):
    print(f"Value: {int(v)}, Count: {c}")

print("\nUnique values and counts in 'trunc_raw':")
for v, c in zip(trunc_vals, trunc_counts):
    print(f"Value: {int(v)}, Count: {c}")

# --- 4. Bar Plot for Unique Value Distribution ---
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].bar(round_vals, round_counts, color='blue')
axs[0].set_title("Distribution of Round Values")
axs[0].set_xticks(round_vals)
axs[0].set_xlabel("Value")
axs[0].set_ylabel("Count")

axs[1].bar(trunc_vals, trunc_counts, color='green')
axs[1].set_title("Distribution of Trunc Values")
axs[1].set_xticks(trunc_vals)
axs[1].set_xlabel("Value")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
