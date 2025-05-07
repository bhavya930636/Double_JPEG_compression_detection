import os
import subprocess

base_dir = "../data/dataset/8"

for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".npz"):
            npz_path = os.path.join(root, f)
            name = os.path.splitext(f)[0]
            new_dir = os.path.join(root, name)
            os.makedirs(new_dir, exist_ok=True)
            print(f"Created: {new_dir}")

            # Run visualise.py with npz_path as an argument
            subprocess.run(["python", "visualise_r_t.py", npz_path])


# for visualisarions