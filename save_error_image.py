# single_train_path = Path(dir_path) / "patches_train" / str(patch_size) / f"Quality_{Q_val}" / "index_all" / "single"
# single_test_path = Path(dir_path) / "patches_test" / str(patch_size) / f"Quality_{Q_val}" / "index_all" / "single"
# double_train_path = Path(dir_path) / "patches_train" / str(patch_size) / f"Quality_{Q_val}" / "index_all" / "double"
# double_test_path = Path(dir_path) / "patches_test" / str(patch_size) / f"Quality_{Q_val}" / "index_all" / "double"

# for prefix in prefixes:
#     load_prefix = Path(dir_path) / f"patches_{prefix}"

#     read_single_path = load_prefix / f"{patch_size}/Quality_{Q_val}/index_{stability_index}/single"
#     read_double_path = load_prefix / f"{patch_size}/Quality_{Q_val}/index_{stability_index}/double"

#     single_save_path = Path(dir_path) / f"dataset/{patch_size}/{prefix}Quality_{Q_val}/index_{stability_index}/single"
#     double_save_path = Path(dir_path) / f"dataset/{patch_size}/{prefix}Quality_{Q_val}/index_{stability_index}/double"

#     single_save_path.mkdir(parents=True, exist_ok=True)
#     double_save_path.mkdir(parents=True, exist_ok=True)
import os
import numpy as np
import jpegio as jio
from PIL import Image
from tifs_2014 import TIFS_2014  # Assuming TIFS_2014.py is in the same directory

def create_img_struct(directory):
    """Helper function to list image files in a directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

def remove_zero_feature(training, testing):
    """Remove zero features from the dataset."""
    # This function will remove features that are zero in all training/testing samples
    non_zero_indices = np.any(training != 0, axis=0)
    training_new = training[:, non_zero_indices]
    testing_new = testing[:, non_zero_indices]
    return training_new, testing_new, non_zero_indices

def save_error_images_and_features(dir_path, stability_index='all', patch_size=8):
    """Equivalent to the MATLAB function saving error images and feature vectors."""
    Q_list = [20, 40, 60, 70, 75, 80, 85, 90]  # All quality factors reported

    if stability_index == '1':
        Q_list = [60, 70, 75, 80, 85, 90]  # Choosing only 60-90 for Qf == 1 analysis.

    prefixes = ['test/', 'train/']

    # Loop through quality factors
    for Q_val in Q_list:
        print(f"Processing quality: {Q_val}")
        
        # Directory paths for saving data
        for prefix in prefixes:
            load_prefix = os.path.join(dir_path, f'patches_{prefix}')
            read_single_path = os.path.join(load_prefix, str(patch_size), f'Quality_{Q_val}', f'index_{stability_index}', 'single')
            read_double_path = os.path.join(load_prefix, str(patch_size), f'Quality_{Q_val}', f'index_{stability_index}', 'double')

            single_save_path = os.path.join(dir_path, f'dataset/{patch_size}/{prefix}Quality_{Q_val}/index_{stability_index}/single')
            double_save_path = os.path.join(dir_path, f'dataset/{patch_size}/{prefix}Quality_{Q_val}/index_{stability_index}/double')

            os.makedirs(single_save_path, exist_ok=True)
            os.makedirs(double_save_path, exist_ok=True)

            for k in range(1, 5):
                os.makedirs(os.path.join(single_save_path, str(k)), exist_ok=True)
                os.makedirs(os.path.join(double_save_path, str(k)), exist_ok=True)

            # Process images and extract features
            for k in range(1, 4):
                single_path = create_img_struct(os.path.join(read_single_path, str(k)))
                double_path = create_img_struct(os.path.join(read_double_path, str(k+1)))

                if prefix == 'train/':
                    if len(single_path) > 0:
                        trunc, round, single_dct_error, single_error, single_EBSF = TIFS_2014(single_path)
                    if len(double_path) > 0:
                        trunc, round, double_dct_error, double_error, double_EBSF = TIFS_2014(double_path)

                    # Save data
                    np.save(os.path.join(single_save_path, str(k), 'single_error.npy'), {'single_error': single_error, 'trunc': trunc, 'round': round})
                    np.save(os.path.join(single_save_path, str(k), 'single_dct_error.npy'), {'single_dct_error': single_dct_error, 'trunc': trunc, 'round': round})

                    np.save(os.path.join(double_save_path, str(k+1), 'double_error.npy'), {'double_error': double_error, 'trunc': trunc, 'round': round})
                    np.save(os.path.join(double_save_path, str(k+1), 'double_dct_error.npy'), {'double_dct_error': double_dct_error, 'trunc': trunc, 'round': round})

                    if k == 1:
                        single_vec_train = single_EBSF
                        single_trunc_train = trunc
                        single_round_train = round
                        double_vec_train = double_EBSF
                        double_trunc_train = trunc
                        double_round_train = round

                else:
                    if len(single_path) > 0:
                        trunc, round, single_dct_error, single_error, single_EBSF = TIFS_2014(single_path)
                    if len(double_path) > 0:
                        trunc, round, double_dct_error, double_error, double_EBSF = TIFS_2014(double_path)

                    # Save data
                    np.save(os.path.join(single_save_path, str(k), 'single_error.npy'), {'single_error': single_error, 'trunc': trunc, 'round': round})
                    np.save(os.path.join(single_save_path, str(k), 'single_dct_error.npy'), {'single_dct_error': single_dct_error, 'trunc': trunc, 'round': round})

                    np.save(os.path.join(double_save_path, str(k+1), 'double_error.npy'), {'double_error': double_error, 'trunc': trunc, 'round': round})
                    np.save(os.path.join(double_save_path, str(k+1), 'double_dct_error.npy'), {'double_dct_error': double_dct_error, 'trunc': trunc, 'round': round})

                    if k == 1:
                        single_vec_test = single_EBSF
                        single_trunc_test = trunc
                        single_round_test = round
                        double_vec_test = double_EBSF
                        double_trunc_test = trunc
                        double_round_test = round

            # Add label column for training and testing
            single_vec_test = np.hstack([single_vec_test, np.full((single_vec_test.shape[0], 1), 0)])
            single_vec_train = np.hstack([single_vec_train, np.full((single_vec_train.shape[0], 1), 0)])
            double_vec_test = np.hstack([double_vec_test, np.full((double_vec_test.shape[0], 1), 1)])
            double_vec_train = np.hstack([double_vec_train, np.full((double_vec_train.shape[0], 1), 1)])

            # Combine training and testing sets
            training = np.vstack([single_vec_train, double_vec_train])
            testing = np.vstack([single_vec_test, double_vec_test])

            # Handle NaN values by replacing them with zeros
            training[np.isnan(training)] = 0
            testing[np.isnan(testing)] = 0

            # Remove zero features
            training_new, testing_new, indices = remove_zero_feature(training, testing)

            # Split the dataset back into single and double components
            single_vec_train = training_new[:single_vec_train.shape[0], :]
            double_vec_train = training_new[single_vec_train.shape[0]:, :]

            single_vec_test = testing_new[:single_vec_test.shape[0], :]
            double_vec_test = testing_new[single_vec_test.shape[0]:, :]

            # Save final datasets
            np.save(os.path.join(single_save_path, 'EBSF_single_train.npy'), {'single_vec_train': single_vec_train, 'indices': indices})
            np.save(os.path.join(double_save_path, 'EBSF_double_train.npy'), {'double_vec_train': double_vec_train, 'indices': indices})
            np.save(os.path.join(single_save_path, 'EBSF_single_test.npy'), {'single_vec_test': single_vec_test, 'indices': indices})
            np.save(os.path.join(double_save_path, 'EBSF_double_test.npy'), {'double_vec_test': double_vec_test, 'indices': indices})

# Example call to save error images and features
dir_path = '../../data/'
save_error_images_and_features(dir_path)
