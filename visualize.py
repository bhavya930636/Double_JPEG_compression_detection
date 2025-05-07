import numpy as np
import matplotlib.pyplot as plt
import sys
import os

npz_path = sys.argv[1]
file = os.path.splitext(os.path.basename(npz_path))[0]
save_dir = os.path.join(os.path.dirname(npz_path), file)
os.makedirs(save_dir, exist_ok=True)

data = np.load(npz_path, allow_pickle=True)

if "EBSF" not in file:
    if file == "single_error":
        dct_error_raw = data['single_error']
        round_raw = data['round']
        trunc_raw = data['trunc']
    elif file == "single_dct_error":
        dct_error_raw = data['single_dct_error']
        round_raw = data['round']
        trunc_raw = data['trunc']
    elif file == "double_error":
        dct_error_raw = data['double_error']
        round_raw = data['round']
        trunc_raw = data['trunc']
    elif file == "double_dct_error":
        dct_error_raw = data['double_dct_error']
        round_raw = data['round']
        trunc_raw = data['trunc']
    else:
        print(f"Unsupported file type: {file}")
        sys.exit(1)

    def extract_patch(patch_raw):
        return patch_raw[0].squeeze().astype(np.float32)

    print(f"Extracting and saving {len(dct_error_raw)} patches to {save_dir}...")

    for i in range(len(dct_error_raw)):
        patch = extract_patch(dct_error_raw[i])
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(patch, cmap='plasma', vmin=-5, vmax=5)
        ax.axis('off')
        fig.tight_layout(pad=0)
        patch_path = os.path.join(save_dir, f"patch_{i+1:05d}.png")
        plt.savefig(patch_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"Saved all {len(dct_error_raw)} patch images.")

    # Save round/trunc value distributions
    round_vals, round_counts = np.unique(round_raw.flatten(), return_counts=True)
    trunc_vals, trunc_counts = np.unique(trunc_raw.flatten(), return_counts=True)

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
    dist_path = os.path.join(save_dir, "round_trunc_distribution.png")
    plt.savefig(dist_path)
    plt.close()

    print(f"Saved distribution plot to {dist_path}")
