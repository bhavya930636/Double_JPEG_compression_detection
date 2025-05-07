import os
import numpy as np
import cv2
import jpegio as jio
from glob import glob
import matplotlib.pyplot as plt
import matplotlib

Q_list = [70]
matplotlib.use('TkAgg')
dir_path = '/home/user/Documents/jpeg_dc_detection2/code/data'
files = (glob(os.path.join(dir_path, 'ucid.v2', '*.tif')))
# print(files)
patch_size_1 = 8
patch_size_2 = 8

stability_index = 'all'
train = False   ## once for ture once for fasle 

if train:
    save_prefix = os.path.join(dir_path, 'patches_train/')
    image_range = range(0, len(files)//2)
else:
    save_prefix = os.path.join(dir_path, 'patches_test/')
    image_range = range(len(files)//2, len(files))

for q in range(len(Q_list)):
    Q_val = Q_list[q]
    print(f"Q_val = {Q_val}")

    prefix = os.path.join(dir_path, f'Compressed_UCID_gray_full/Quality_{Q_val}')
    
    os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', f'index_{stability_index}'), exist_ok=True)
    os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', 'all'), exist_ok=True)
    os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', f'index_{stability_index}', 'single'), exist_ok=True)
    os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', f'index_{stability_index}', 'double'), exist_ok=True)
    
    for k in range(1,  6):
        os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', f'index_{stability_index}', 'double', str(k)), exist_ok=True)
        os.makedirs(os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}', f'index_{stability_index}', 'single', str(k)), exist_ok=True)
    
    single_path = os.path.join(prefix, 'single/')
    double_path = os.path.join(prefix, 'double/')
    triple_path = os.path.join(prefix, 'triple/')
    fourth_path = os.path.join(prefix, 'fourth/')
    
    write_prefix = os.path.join(save_prefix, str(patch_size_1), f'Quality_{Q_val}')
    cnt_single, cnt_double = 0, 0
    diff1, diff2, diff3 = 0, 0, 0

    print(type(image_range))
    for f in image_range:
        if f == 0:
            continue
        print(f)

        s_img = jio.read(f"{single_path}{f}.jpg")
        d_img = jio.read(f"{double_path}{f}.jpg")
        t_img = jio.read(f"{triple_path}{f}.jpg")
        q_img = jio.read(f"{fourth_path}{f}.jpg")

        s, d, t, q = s_img.coef_arrays[0], d_img.coef_arrays[0], t_img.coef_arrays[0], q_img.coef_arrays[0]
        rows, cols = s.shape
        print(rows,cols)
        ss_img = cv2.imread(f"{single_path}{f}.jpg", cv2.IMREAD_GRAYSCALE)
        # print(f)
        original_img = cv2.imread(f"/home/user/Documents/jpeg_dc_detection2/code/data/ucid.v2/{f}.tif", cv2.IMREAD_GRAYSCALE)   
        # print(ss_img)
        print(ss_img.shape)
        # print(f"{single_path}{f}.jpg")
        print(f"/home/user/Documents/jpeg_dc_detection2/code/data/ucid.v2/{f}.tif")
        # print(original_img)/
        print(original_img.shape)
        if f == 4:
            plt.imshow(original_img, cmap='gray')
            plt.axis('off')
            plt.show()

        for i in range(0, rows - patch_size_1 + 1, patch_size_1):
            for j in range(0, cols - patch_size_2 + 1, patch_size_2):
                p_s = s[i:i+patch_size_1, j:j+patch_size_2]
                p_d = d[i:i+patch_size_1, j:j+patch_size_2]
                p_t = t[i:i+patch_size_1, j:j+patch_size_2]
                p_q = q[i:i+patch_size_1, j:j+patch_size_2]

                if stability_index == '1':
                    diff1 = np.count_nonzero(p_s - p_d)
                    diff2 = np.count_nonzero(p_d - p_t)
                    diff3 = np.count_nonzero(p_t - p_q)
                elif stability_index == 'all':
                    diff1 = np.count_nonzero(p_s - p_d)
                    diff2 = np.count_nonzero(p_d - p_t)
                    diff3 = -1
                else:
                    raise ValueError('Incorrect Stability Index value, use: "1" or "all"')

                patch = original_img[i:i+patch_size_1, j:j+patch_size_2]
                # print(patch)
                
                if diff1 > 0 and diff2 == 0 and diff3 != -1:
                    cnt_single += 1
                    for k in range(4):
                        path = os.path.join(write_prefix, f'index_{stability_index}/single/{k+1}/{cnt_single}.jpg')
                        if k == 0:
                            cv2.imwrite(path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), Q_val])
                        else:
                            img = cv2.imread(os.path.join(write_prefix, f'index_{stability_index}/single/{k}/{cnt_single}.jpg'), cv2.IMREAD_GRAYSCALE)
                            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), Q_val])

                if diff1 > 0 and diff3 == -1:
                    cnt_single += 1
                    for k in range(3):
                        path = os.path.join(write_prefix, f'index_all/single/{k+1}/{cnt_single}.jpg')
                        if k == 0:
                            cv2.imwrite(path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), Q_val])
                        else:
                            img = cv2.imread(os.path.join(write_prefix, f'index_all/single/{k}/{cnt_single}.jpg'), cv2.IMREAD_GRAYSCALE)
                            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), Q_val])

                if diff2 > 0 and diff3 == -1:
                    cnt_double += 1
                    for k in range(4):
                        path = os.path.join(write_prefix, f'index_all/double/{k+1}/{cnt_double}.jpg')
                        if k == 0:
                            cv2.imwrite(path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), Q_val])
                        else:
                            img = cv2.imread(os.path.join(write_prefix, f'index_all/double/{k}/{cnt_double}.jpg'), cv2.IMREAD_GRAYSCALE)
                            cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), Q_val])
