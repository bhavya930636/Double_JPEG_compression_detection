import os
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import scipy.io as sio
from pathlib import Path
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Equivalent to MATLAB's BDCT (Block DCT) function
def bdct(block):
    """Apply block-wise DCT transformation to an image"""
    h, w = block.shape
    dct_result = np.zeros((h, w))
    
    # Process 8x8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if i+8 <= h and j+8 <= w:
                block_8x8 = block[i:i+8, j:j+8]
                dct_result[i:i+8, j:j+8] = dct(dct(block_8x8.T, norm='ortho').T, norm='ortho')
    
    return dct_result

# Equivalent to MATLAB's IDCT (Inverse Block DCT) function
def ibdct(dct_block):
    """Apply block-wise inverse DCT transformation to an image"""
    h, w = dct_block.shape
    idct_result = np.zeros((h, w))
    
    # Process 8x8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if i+8 <= h and j+8 <= w:
                block_8x8 = dct_block[i:i+8, j:j+8]
                idct_result[i:i+8, j:j+8] = idct(idct(block_8x8.T, norm='ortho').T, norm='ortho')
    
    return idct_result

# Equivalent to MATLAB's jpeg_rec_gray function - reconstructs image from JPEG data
def jpeg_rec_gray(jpeg_data):
    """Reconstruct grayscale image from JPEG data"""
    # This is a simplified version - in a real implementation you would
    # use actual JPEG decompression here
    height, width = jpeg_data['image_height'], jpeg_data['image_width']
    dct_coeffs = jpeg_data['dct_coeffs']
    quant_table = jpeg_data['quant_tables'][0]
    
    # Dequantize coefficients
    quantized_coeffs = jpeg_data['quantized_coeffs']
    dequantized = quantized_coeffs * quant_table
    
    # Apply inverse DCT
    reconstructed = ibdct(dequantized)
    
    # Clip values to valid range
    reconstructed = np.clip(reconstructed, 0, 255)
    
    return reconstructed

# Equivalent to MATLAB's jpeg_read function
def jpeg_read(image_path):
    """Read JPEG image and extract DCT coefficients and quantization tables"""
    # In a real implementation, you would extract actual DCT coefficients from JPEG
    # This is a simplified version for demonstration
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to read image: {image_path}")
        raise ValueError(f"Could not read image: {image_path}")
        
    height, width = img.shape
    
    # Get quantization table
    # In a real implementation, you would extract this from the JPEG header
    # For this example, we'll use a standard JPEG quantization table
    quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Create quantization table for the entire image
    h_blocks = height // 8
    w_blocks = width // 8
    q_repeated = np.tile(quant_table, (h_blocks, w_blocks))
    
    # Calculate DCT coefficients
    dct_coeffs = bdct(img.astype(float))
    
    # Quantize coefficients
    quantized = np.round(dct_coeffs / q_repeated)
    
    return {
        'image_height': height,
        'image_width': width,
        'dct_coeffs': dct_coeffs,
        'quantized_coeffs': quantized,
        'quant_tables': [quant_table]
    }

def remove_zero_feature(training, testing):
    """Remove features that are all zeros"""
    indices = []
    if training.size == 0 or testing.size == 0:
        logger.warning("Training or testing data is empty in remove_zero_feature")
        return training, testing, []
        
    for i in range(training.shape[1] - 1):  # Exclude the last column which is the label
        if np.sum(np.abs(training[:, i])) != 0:
            indices.append(i)
    
    indices.append(training.shape[1] - 1)  # Add the label column
    indices = np.array(indices)
    
    training_new = training[:, indices]
    testing_new = testing[:, indices]
    
    return training_new, testing_new, indices

def process_single_image(img_path):
    """Process a single image for TIFS_2014 (for parallel processing)"""
    try:
        trun = 0
        jpeg_img = jpeg_read(img_path)
        I = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if I is None:
            logger.error(f"Failed to read image for processing: {img_path}")
            return None
            
        rec = jpeg_rec_gray(jpeg_img)
        Q = jpeg_img['quant_tables'][0]
        Q_rep = np.tile(Q, (I.shape[0]//8, I.shape[1]//8))
        R = I.astype(float) - rec
        M = np.int64(bdct(R) / Q_rep)
        err = R
        dct_err = bdct(err)
        
        # Reshape for consistency with the MATLAB code
        err_reshaped = np.reshape(err, (1, 1, err.shape[0], err.shape[1]))
        dct_err_reshaped = np.reshape(dct_err, (1, 1, dct_err.shape[0], dct_err.shape[1]))
        
        zero_8 = np.zeros((8, 8))
        r_error = []
        r_error_dc = []
        r_error_ac = []
        t_error = []
        t_error_dc = []
        t_error_ac = []
        
        for i in range(0, M.shape[0], 8):
            for j in range(0, M.shape[1], 8):
                if i+8 <= M.shape[0] and j+8 <= M.shape[1]:
                    M_n = M[i:i+8, j:j+8]
                    
                    # Process unstable block only
                    if not np.array_equal(M_n, zero_8):
                        R_n = R[i:i+8, j:j+8]
                        W_n = M_n.astype(float) * Q
                        W_n = W_n.reshape(1, 64)
                        
                        # Rounding error block
                        if np.max(R_n) <= 0.5 and np.min(R_n) >= -0.5:
                            trun = 0
                            r_error.append(R_n)
                            r_error_dc.append(W_n[0, 0])
                            r_error_ac.extend(W_n[0, 1:])
                        # Truncation error block
                        else:
                            trun = 1
                            t_error.append(R_n)
                            t_error_dc.append(W_n[0, 0])
                            t_error_ac.extend(W_n[0, 1:])
        
        trunc_block = 1 if trun == 1 else 0
        round_block = 1 if trun == 0 else 0
        
        # Convert lists to numpy arrays for calculations
        r_error = np.array(r_error).flatten() if r_error else np.array([])
        t_error = np.array(t_error).flatten() if t_error else np.array([])
        r_error_dc = np.array(r_error_dc) if r_error_dc else np.array([])
        r_error_ac = np.array(r_error_ac) if r_error_ac else np.array([])
        t_error_dc = np.array(t_error_dc) if t_error_dc else np.array([])
        t_error_ac = np.array(t_error_ac) if t_error_ac else np.array([])
        
        # Calculate feature vector
        feature_vec = [
            np.mean(np.abs(r_error)) if r_error.size > 0 else 0,
            np.var(np.abs(r_error)) if r_error.size > 0 else 0,
            np.mean(np.abs(t_error)) if t_error.size > 0 else 0,
            np.var(np.abs(t_error)) if t_error.size > 0 else 0,
            np.mean(np.abs(r_error_dc)) if r_error_dc.size > 0 else 0,
            np.var(np.abs(r_error_dc)) if r_error_dc.size > 0 else 0,
            np.mean(np.abs(r_error_ac)) if r_error_ac.size > 0 else 0,
            np.var(np.abs(r_error_ac)) if r_error_ac.size > 0 else 0,
            np.mean(np.abs(t_error_dc)) if t_error_dc.size > 0 else 0,
            np.var(np.abs(t_error_dc)) if t_error_dc.size > 0 else 0,
            np.mean(np.abs(t_error_ac)) if t_error_ac.size > 0 else 0,
            np.var(np.abs(t_error_ac)) if t_error_ac.size > 0 else 0,
            len(r_error_dc) / (len(r_error_dc) + len(t_error_dc)) if (len(r_error_dc) + len(t_error_dc)) > 0 else 0
        ]
        
        return (trunc_block, round_block, dct_err_reshaped, err_reshaped, feature_vec)
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        return None

def TIFS_2014(image_paths):
    """Process images and extract error features"""
    if not image_paths:
        logger.warning("No image paths provided to TIFS_2014")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
    trunc_blocks = []
    round_blocks = []
    dct_error_images = []
    error_images = []
    final_feat = []
    
    # Process images in parallel to speed up
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_single_image, image_paths))
    
    # Filter out None results (failed processing)
    results = [r for r in results if r is not None]
    
    if not results:
        logger.warning("No valid results from image processing")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # Unpack results
    for result in results:
        trunc_block, round_block, dct_err, err, feat = result
        trunc_blocks.append(trunc_block)
        round_blocks.append(round_block)
        dct_error_images.append(dct_err)
        error_images.append(err)
        final_feat.append(feat)
    
    return (
        np.array(trunc_blocks),
        np.array(round_blocks),
        np.array(dct_error_images),
        np.array(error_images),
        np.array(final_feat)
    )

def get_error_image(image_paths):
    """Extract error images from JPEG files"""
    if not image_paths:
        return np.array([])
        
    error = []
    
    for i, path in enumerate(image_paths):
        if i % 1000 == 0:
            logger.info(f"Processing image {i}")
        
        try:
            jpeg_data = jpeg_read(path)
            I = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if I is None:
                logger.warning(f"Failed to read image: {path}")
                continue
                
            rec = jpeg_rec_gray(jpeg_data)
            err = I.astype(float) - rec
            err_reshaped = np.reshape(err, (1, 1, err.shape[0], err.shape[1]))
            error.append(err_reshaped)
        except Exception as e:
            logger.error(f"Error processing image {path}: {str(e)}")
    
    return np.array(error)

def create_img_struct(dir_path, seed=2):
    """Create image structure from directory"""
    if not os.path.exists(dir_path):
        logger.warning(f"Directory does not exist: {dir_path}")
        return []
        
    image_paths = list(Path(dir_path).glob('*.jpg'))
    array_size = len(image_paths)
    
    if array_size == 0:
        logger.warning(f"No jpg images found in directory: {dir_path}")
        return []
        
    used_size = array_size
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Select random images
    choices = random.sample(range(array_size), min(used_size, array_size))
    
    img_struct = []
    for i in range(len(choices)):
        ch = choices[i]
        read_path = os.path.join(dir_path, f"{ch}.jpg")
        if os.path.exists(read_path):
            img_struct.append(read_path)
    
    return img_struct

# Main script
def main():
    dir_path = '../data/'
    Q_list = [40,70]  # All quality factors reported
    stability_index = 'all'
    
    if stability_index == '1':
        Q_list = [60, 70, 75, 80, 85, 90]  # Choosing only 60-90 for Qf == 1 analysis
    
    ld = 1
    ls = 0
    prefixes = ['test/', 'train/']
    patch_size = 8
    
    # Choose quality factors
    for Q_val in Q_list:
        logger.info(f"Processing Quality: {Q_val}")
        
        # Define paths
        single_train_path = os.path.join(dir_path, f'dataset/{patch_size}/train/Quality_{Q_val}/index_{stability_index}/single')
        single_test_path = os.path.join(dir_path, f'dataset/{patch_size}/test/Quality_{Q_val}/index_{stability_index}/single')
        double_train_path = os.path.join(dir_path, f'dataset/{patch_size}/train/Quality_{Q_val}/index_{stability_index}/double')
        double_test_path = os.path.join(dir_path, f'dataset/{patch_size}/test/Quality_{Q_val}/index_{stability_index}/double')
        
        # Create directory structure
        os.makedirs(single_train_path, exist_ok=True)
        os.makedirs(single_test_path, exist_ok=True)
        os.makedirs(double_train_path, exist_ok=True)
        os.makedirs(double_test_path, exist_ok=True)
        
        # Initialize variables for both train and test datasets
        single_vec_train = np.array([])
        double_vec_train = np.array([])
        single_vec_test = np.array([])
        double_vec_test = np.array([])
        
        for prefix in prefixes:
            load_prefix = os.path.join(dir_path, f'patches_{prefix}')
            
            read_single_path = os.path.join(load_prefix, f'{patch_size}/Quality_{Q_val}/index_{stability_index}/single')
            read_double_path = os.path.join(load_prefix, f'{patch_size}/Quality_{Q_val}/index_{stability_index}/double')
            
            single_save_path = os.path.join(dir_path, f'dataset/{patch_size}/{prefix}Quality_{Q_val}/index_{stability_index}/single')
            double_save_path = os.path.join(dir_path, f'dataset/{patch_size}/{prefix}Quality_{Q_val}/index_{stability_index}/double')
            
            # Create directories
            os.makedirs(single_save_path, exist_ok=True)
            os.makedirs(double_save_path, exist_ok=True)
            
            for k in range(1, 5):
                os.makedirs(os.path.join(single_save_path, str(k)), exist_ok=True)
                os.makedirs(os.path.join(double_save_path, str(k)), exist_ok=True)
            
            # Process images
            for k in range(1, 4):
                single_path = create_img_struct(os.path.join(read_single_path, str(k)))
                double_path = create_img_struct(os.path.join(read_double_path, str(k+1)))
                
                if prefix == 'train/':
                    if len(single_path) > 0:
                        try:
                            trunc, round_val, single_dct_error, single_error, single_EBSF = TIFS_2014(single_path)
                            
                            # Save results only if we have valid data
                            if single_error.size > 0:
                                np.savez(os.path.join(single_save_path, str(k), 'single_error.npz'), 
                                        single_error=single_error, trunc=trunc, round=round_val)
                            
                            if single_dct_error.size > 0:
                                np.savez(os.path.join(single_save_path, str(k), 'single_dct_error.npz'), 
                                        single_dct_error=single_dct_error, trunc=trunc, round=round_val)
                            
                            if k == 1 and single_EBSF.size > 0:
                                single_vec_train = single_EBSF
                        except Exception as e:
                            logger.error(f"Error processing single train images: {str(e)}")
                    
                    if len(double_path) > 0:
                        try:
                            trunc, round_val, double_dct_error, double_error, double_EBSF = TIFS_2014(double_path)
                            
                            # Save results only if we have valid data
                            if double_error.size > 0:
                                np.savez(os.path.join(double_save_path, str(k+1), 'double_error.npz'), 
                                        double_error=double_error, trunc=trunc, round=round_val)
                            
                            if double_dct_error.size > 0:
                                np.savez(os.path.join(double_save_path, str(k+1), 'double_dct_error.npz'), 
                                        double_dct_error=double_dct_error, trunc=trunc, round=round_val)
                            
                            if k == 1 and double_EBSF.size > 0:
                                double_vec_train = double_EBSF
                        except Exception as e:
                            logger.error(f"Error processing double train images: {str(e)}")
                
                else:  # Test prefix
                    if len(single_path) > 0:
                        try:
                            trunc, round_val, single_dct_error, single_error, single_EBSF = TIFS_2014(single_path)
                            
                            # Save results only if we have valid data
                            if single_error.size > 0:
                                np.savez(os.path.join(single_save_path, str(k), 'single_error.npz'), 
                                        single_error=single_error, trunc=trunc, round=round_val)
                            
                            if single_dct_error.size > 0:
                                np.savez(os.path.join(single_save_path, str(k), 'single_dct_error.npz'), 
                                        single_dct_error=single_dct_error, trunc=trunc, round=round_val)
                            
                            if k == 1 and single_EBSF.size > 0:
                                single_vec_test = single_EBSF
                        except Exception as e:
                            logger.error(f"Error processing single test images: {str(e)}")
                    
                    if len(double_path) > 0:
                        try:
                            trunc, round_val, double_dct_error, double_error, double_EBSF = TIFS_2014(double_path)
                            
                            # Save results only if we have valid data
                            if double_error.size > 0:
                                np.savez(os.path.join(double_save_path, str(k+1), 'double_error.npz'), 
                                        double_error=double_error, trunc=trunc, round=round_val)
                            
                            if double_dct_error.size > 0:
                                np.savez(os.path.join(double_save_path, str(k+1), 'double_dct_error.npz'), 
                                        double_dct_error=double_dct_error, trunc=trunc, round=round_val)
                            
                            if k == 1 and double_EBSF.size > 0:
                                double_vec_test = double_EBSF
                        except Exception as e:
                            logger.error(f"Error processing double test images: {str(e)}")
        
        # Add labels and continue processing only if we have data
        if (single_vec_test.size > 0 and single_vec_train.size > 0 and 
            double_vec_test.size > 0 and double_vec_train.size > 0):
            
            # Add labels
            single_vec_test = np.c_[single_vec_test, np.full((single_vec_test.shape[0], 1), ls)]
            single_vec_train = np.c_[single_vec_train, np.full((single_vec_train.shape[0], 1), ls)]
            
            double_vec_test = np.c_[double_vec_test, np.full((double_vec_test.shape[0], 1), ld)]
            double_vec_train = np.c_[double_vec_train, np.full((double_vec_train.shape[0], 1), ld)]
            
            single_train_size = single_vec_train.shape[0]
            single_test_size = single_vec_test.shape[0]
            
            training = np.vstack((single_vec_train, double_vec_train))
            testing = np.vstack((single_vec_test, double_vec_test))
            
            # Replace NaNs with zeros
            training = np.nan_to_num(training)
            testing = np.nan_to_num(testing)
            
            # Remove zero features
            training_new, testing_new, indices = remove_zero_feature(training, testing)
            
            if len(indices) > 0:  # Only proceed if we have valid indices
                single_vec_train = training_new[:single_train_size, :]
                double_vec_train = training_new[single_train_size:, :]
                
                single_vec_test = testing_new[:single_test_size, :]
                double_vec_test = testing_new[single_test_size:, :]
                
                # Save features
                np.savez(os.path.join(single_train_path, 'EBSF_single_train.npz'), 
                        single_vec_train=single_vec_train, indices=indices)
                np.savez(os.path.join(double_train_path, 'EBSF_double_train.npz'), 
                        double_vec_train=double_vec_train, indices=indices)
                np.savez(os.path.join(single_test_path, 'EBSF_single_test.npz'), 
                        single_vec_test=single_vec_test, indices=indices)
                np.savez(os.path.join(double_test_path, 'EBSF_double_test.npz'), 
                        double_vec_test=double_vec_test, indices=indices)
                
                logger.info(f"Successfully saved feature data for Quality {Q_val}")
            else:
                logger.warning(f"No valid features found for Quality {Q_val}")
        else:
            logger.warning(f"Insufficient data for Quality {Q_val}, skipping feature generation")

if __name__ == "__main__":
    main()