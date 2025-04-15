import os
from PIL import Image

# List of quality values
Q_list = [40, 70]

# Directory path
dir_path = r'C:\Users\CSE IIT BHILAI\Documents\mcb\jpeg_dc_detection-master\data folder'

# Directory of image files
image_dir = os.path.join(dir_path, 'ucid.v2')

# Loop through quality values
for Q_val in Q_list:  # q=2 corresponds to Q_list[1] = 40
    
    print(f"Processing images for Quality {Q_val}")

    # Create necessary directories for each quality
    prefix = os.path.join(dir_path, f'Compressed_UCID_gray_full/Quality_{Q_val}')
    os.makedirs(prefix, exist_ok=True)
    os.makedirs(os.path.join(prefix, 'single'), exist_ok=True)
    os.makedirs(os.path.join(prefix, 'double'), exist_ok=True)
    os.makedirs(os.path.join(prefix, 'triple'), exist_ok=True)
    os.makedirs(os.path.join(prefix, 'fourth'), exist_ok=True)
    os.makedirs(os.path.join(prefix, 'fifth'), exist_ok=True)
    os.makedirs(os.path.join(prefix, 'sixth'), exist_ok=True)

    # List all tif images in the directory
    files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    for i, file_name in enumerate(files):
        # Load and process the original image
        original_image_path = os.path.join(image_dir, file_name)
        original_image = Image.open(original_image_path).convert('L')  # Convert to grayscale
        
        # Save the images with different compression qualities
        single_image_path = os.path.join(prefix, 'single', f'{i+1}.jpg')
        original_image.save(single_image_path, 'JPEG', quality=Q_val)

        # Repeatedly compress and save images in each quality step
        single_image = Image.open(single_image_path)
        double_image_path = os.path.join(prefix, 'double', f'{i+1}.jpg')
        single_image.save(double_image_path, 'JPEG', quality=Q_val)

        double_image = Image.open(double_image_path)
        triple_image_path = os.path.join(prefix, 'triple', f'{i+1}.jpg')
        double_image.save(triple_image_path, 'JPEG', quality=Q_val)

        triple_image = Image.open(triple_image_path)
        fourth_image_path = os.path.join(prefix, 'fourth', f'{i+1}.jpg')
        triple_image.save(fourth_image_path, 'JPEG', quality=Q_val)

        fourth_image = Image.open(fourth_image_path)
        fifth_image_path = os.path.join(prefix, 'fifth', f'{i+1}.jpg')
        fourth_image.save(fifth_image_path, 'JPEG', quality=Q_val)

        fifth_image = Image.open(fifth_image_path)
        sixth_image_path = os.path.join(prefix, 'sixth', f'{i+1}.jpg')
        fifth_image.save(sixth_image_path, 'JPEG', quality=Q_val)

    print(f"Completed for Quality {Q_val}")
