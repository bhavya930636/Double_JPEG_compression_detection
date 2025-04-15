import os
import cv2
import numpy as np
from PIL import Image

Q_list = [20, 40, 60, 70, 75, 80, 85, 90]
dir_path = '../data/'
files = [f for f in os.listdir(os.path.join(dir_path, 'ucid.v2')) if f.endswith('.tif')]
patch_size_1 = 8
patch_size_2 = 8
stability_index = 'all'
train = False

if train:
    save_prefix = os.path.join(dir_path, 'patches_train/')
    image_range = range(0, len(files) // 2)
else:
    save_prefix = os.path.join(dir_path, 'patches_test/')
    image_range = range(len(files) // 2, len(files))

for Q_val in Q_list:
    prefix = os.path.join(dir_path, 'Compressed_UCID_gray_full', f'Quality_{Q_val}')

    for p in ['single', 'double']:
        for k in range(1, 6):
            os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', f'index_{stability_index}', p, str(k)), exist_ok=True)
        os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', 'all'), exist_ok=True)

    single_path = os.path.join(prefix, 'single/')
    double_path = os.path.join(prefix, 'double/')
    triple_path = os.path.join(prefix, 'triple/')
    fourth_path = os.path.join(prefix, 'fourth/')
    write_prefix = os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}')

    cnt_single = 0
    cnt_double = 0
    
    for f in image_range:
        print(f)
        s = cv2.imread(os.path.join(single_path, f'{f+1}.jpg'), cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(os.path.join(double_path, f'{f+1}.jpg'), cv2.IMREAD_GRAYSCALE)
        t = cv2.imread(os.path.join(triple_path, f'{f+1}.jpg'), cv2.IMREAD_GRAYSCALE)
        ft = cv2.imread(os.path.join(fourth_path, f'{f+1}.jpg'), cv2.IMREAD_GRAYSCALE)
        orig = Image.open(os.path.join(dir_path, 'ucid.v2', files[f])).convert('L')
        orig = np.array(orig)

        rows, cols = s.shape

        for i in range(0, rows - patch_size_1, patch_size_1):
            for j in range(0, cols - patch_size_2, patch_size_2):
                p1 = s[i:i+patch_size_1, j:j+patch_size_2]
                p2 = d[i:i+patch_size_1, j:j+patch_size_2]
                p3 = t[i:i+patch_size_1, j:j+patch_size_2]
                p4 = ft[i:i+patch_size_1, j:j+patch_size_2]
                print("p1 is ",p1)
                print("p2 is ",p2)
                print("p3 is ",p3)
                print("p4 is ",p4)
                diff1 = np.count_nonzero(p1 - p2)
                diff2 = np.count_nonzero(p2 - p3)
                diff3 = -1 if stability_index == 'all' else np.count_nonzero(p3 - p4)

                o_patch = orig[i:i+patch_size_1, j:j+patch_size_2]

                if diff1 > 0 and diff2 == 0 and diff3 != -1:
                    cnt_single += 1
                    path = os.path.join(write_prefix, f'index_{stability_index}', 'single', '1', f'{cnt_single}.jpg')
                    Image.fromarray(o_patch).save(path, quality=Q_val)
                    for k in range(2, 5):
                        img = Image.open(path)
                        img.save(os.path.join(write_prefix, f'index_{stability_index}', 'single', str(k), f'{cnt_single}.jpg'), quality=Q_val)

                if diff2 > 0 and diff3 == 0:
                    cnt_double += 1
                    path = os.path.join(write_prefix, f'index_{stability_index}', 'double', '1', f'{cnt_double}.jpg')
                    Image.fromarray(o_patch).save(path, quality=Q_val)
                    for k in range(2, 5):
                        img = Image.open(path)
                        img.save(os.path.join(write_prefix, f'index_{stability_index}', 'double', str(k), f'{cnt_double}.jpg'), quality=Q_val)

                if diff1 > 0 and diff3 == -1:
                    cnt_single += 1
                    path = os.path.join(write_prefix, f'index_all', 'single', '1', f'{cnt_single}.jpg')
                    Image.fromarray(o_patch).save(path, quality=Q_val)
                    for k in range(2, 4):
                        img = Image.open(path)
                        img.save(os.path.join(write_prefix, f'index_all', 'single', str(k), f'{cnt_single}.jpg'), quality=Q_val)

                if diff2 > 0 and diff3 == -1:
                    cnt_double += 1
                    path = os.path.join(write_prefix, f'index_all', 'double', '1', f'{cnt_double}.jpg')
                    Image.fromarray(o_patch).save(path, quality=Q_val)
                    for k in range(2, 5):
                        img = Image.open(path)
                        img.save(os.path.join(write_prefix, f'index_all', 'double', str(k), f'{cnt_double}.jpg'), quality=Q_val)
