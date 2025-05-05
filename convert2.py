# import numpy as np
# from scipy.io import loadmat
# import os
# import sys

# def convert_mat_to_npz(mat_file_path):
#     if not os.path.exists(mat_file_path):
#         print(f"Error: File '{mat_file_path}' not found.")
#         return

#     mat_data = loadmat(mat_file_path)
#     filtered_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
#     npz_file_path = os.path.splitext(mat_file_path)[0] + '.npz'
#     np.savez(npz_file_path, **filtered_data)
#     print(f"Successfully saved: {npz_file_path}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python mat_to_npz.py <file.mat>")
#         sys.exit(1)
    
#     mat_file = sys.argv[1]
#     convert_mat_to_npz(mat_file)

import numpy as np
from scipy.io import loadmat
import os
import sys
import shutil

def convert_mat_to_npz(mat_file_path):
    if not os.path.exists(mat_file_path):
        print(f"Error: File '{mat_file_path}' not found.")
        return

    mat_data = loadmat(mat_file_path)
    filtered_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    npz_file_path = os.path.splitext(mat_file_path)[0] + '.npz'
    np.savez(npz_file_path, **filtered_data)
    print(f"Successfully saved: {npz_file_path}")

    # If 'EBSF' is in the filename, copy to 'generated_data/'
    if 'EBSF' in os.path.basename(mat_file_path):
        target_dir = 'generated_data'
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(npz_file_path, os.path.join(target_dir, os.path.basename(npz_file_path)))
        print(f"Copied to: {target_dir}/{os.path.basename(npz_file_path)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mat_to_npz.py <file.mat>")
        sys.exit(1)
    
    mat_file = sys.argv[1]
    convert_mat_to_npz(mat_file)
