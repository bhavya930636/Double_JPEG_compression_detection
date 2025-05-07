import os
import subprocess
import shutil

base_dir = "../data/dataset/8"
script_path = "convert2.py"
output_dir = "data"  # Directory to store all processed EBSF files

os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".mat"):
            fpath = os.path.join(root, file)
            try:
                subprocess.run(["python", script_path, fpath], check=True)
                # shutil.copy(fpath, os.path.join(output_dir, file))
                # os.remove(fpath)  # Uncomment if you want to delete after copying
                print(f"Processed and copied to data/: {fpath}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process: {fpath}, Error: {e}")


#converts mat to npz 