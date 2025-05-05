import os
from PIL import Image
from pathlib import Path

q_list = [20, 40, 60, 70, 75, 80, 85, 90]
q = 3    # index in q_list
Q_val = q_list[q]

dir_path = '../data/'
input_dir = Path(dir_path) / 'ucid.v2'
output_prefix = Path(dir_path) / f'Compressed_UCID_gray_full/Quality_{Q_val}'

# Create subdirectories
steps = ['single', 'double', 'triple', 'fourth', 'fifth', 'sixth']
for step in steps:
    os.makedirs(output_prefix / step, exist_ok=True)

# Sort files numerically by filename (e.g., "123.tif" -> 123)
files = sorted(input_dir.glob('*.tif'), key=lambda x: int(x.stem))

for i, file_path in enumerate(files):
    print(file_path)
    idx = int(file_path.stem)
    img = Image.open(file_path).convert('L')  # Convert to grayscale
    img.save(output_prefix / 'single' / f'{idx}.jpg', quality=Q_val)

    prev_path = output_prefix / 'single' / f'{idx}.jpg'
    for j in range(1, len(steps)):
        img = Image.open(prev_path)
        curr_path = output_prefix / steps[j] / f'{idx}.jpg'
        img.save(curr_path, quality=Q_val)
        prev_path = curr_path
