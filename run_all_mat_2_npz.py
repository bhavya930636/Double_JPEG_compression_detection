import os
import subprocess

base_dir = "../data/dataset/8"
script_path = "convert2.py"  # Replace with your actual script name

for root, _, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".mat"):
            fpath = os.path.join(root, file)
            try:
                subprocess.run(["python", script_path, fpath], check=True)
                # os.remove(fpath)
                print(f"Processed and deleted: {fpath}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to process: {fpath}, Error: {e}")
