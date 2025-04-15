import numpy as np
from PIL import Image
import jpegio as jio
from scipy.fftpack import dct, idct
import os

def bdct(block):
    """Apply 2D blockwise DCT."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def jpeg_rec_gray(jpeg_struct):
    """Reconstruct grayscale image from JPEG DCT coefficients and quant tables."""
    quant_table = jpeg_struct.quant_tables[0]
    coeffs = jpeg_struct.coef_arrays[0]
    h, w = coeffs.shape
    rec = np.zeros_like(coeffs, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = coeffs[i:i+8, j:j+8]
            dequant_block = block * quant_table
            rec[i:i+8, j:j+8] = idct(idct(dequant_block.T, norm='ortho').T, norm='ortho')

    rec = np.clip(np.round(rec), 0, 255).astype(np.uint8)
    return rec

def TIFS_2014(image_paths):
    final_feat = []
    error_images = []
    dct_error_images = []
    trunc_block = []
    round_block = []

    for img_path in image_paths:
        jpeg_img = jio.read(img_path)
        I = np.array(Image.open(img_path).convert('L'))  # grayscale
        rec = jpeg_rec_gray(jpeg_img)

        Q = jpeg_img.quant_tables[0]
        Q_rep = np.tile(Q, (I.shape[0]//8, I.shape[1]//8))

        R = I.astype(float) - rec.astype(float)
        M = np.round(bdct(R) / Q_rep)

        err = R.copy()
        dct_err = bdct(err)

        zero_8 = np.zeros((8, 8))

        r_error = []
        r_error_dc = []
        r_error_ac = []
        t_error = []
        t_error_dc = []
        t_error_ac = []

        trun = 0

        for i in range(0, M.shape[0], 8):
            for j in range(0, M.shape[1], 8):
                M_n = M[i:i+8, j:j+8]

                if np.count_nonzero(M_n == 0) != 64:
                    R_n = R[i:i+8, j:j+8]
                    W_n = (M_n * Q).flatten()

                    if np.max(R_n) <= 0.5 and np.min(R_n) >= -0.5:
                        r_error.append(R_n)
                        r_error_dc.append(W_n[0])
                        r_error_ac.extend(W_n[1:])
                    else:
                        trun = 1
                        t_error.append(R_n)
                        t_error_dc.append(W_n[0])
                        t_error_ac.extend(W_n[1:])

        if trun == 0:
            trunc_block.append(0)
            round_block.append(1)
        else:
            trunc_block.append(1)
            round_block.append(0)

        # Convert lists to arrays for feature calculation
        r_error = np.concatenate([block.flatten() for block in r_error]) if r_error else np.array([0])
        t_error = np.concatenate([block.flatten() for block in t_error]) if t_error else np.array([0])
        r_error_dc = np.array(r_error_dc) if r_error_dc else np.array([0])
        r_error_ac = np.array(r_error_ac) if r_error_ac else np.array([0])
        t_error_dc = np.array(t_error_dc) if t_error_dc else np.array([0])
        t_error_ac = np.array(t_error_ac) if t_error_ac else np.array([0])

        # Feature vector
        feature_vec = [
            np.mean(np.abs(r_error)), np.var(np.abs(r_error)),
            np.mean(np.abs(t_error)), np.var(np.abs(t_error)),
            np.mean(np.abs(r_error_dc)), np.var(np.abs(r_error_dc)),
            np.mean(np.abs(r_error_ac)), np.var(np.abs(r_error_ac)),
            np.mean(np.abs(t_error_dc)), np.var(np.abs(t_error_dc)),
            np.mean(np.abs(t_error_ac)), np.var(np.abs(t_error_ac)),
            len(r_error_dc) / (len(r_error_dc) + len(t_error_dc) + 1e-8)
        ]

        final_feat.append(feature_vec)
        error_images.append(err[np.newaxis, np.newaxis, :, :])
        dct_error_images.append(dct_err[np.newaxis, np.newaxis, :, :])

    return (
        np.array(trunc_block),
        np.array(round_block),
        np.concatenate(dct_error_images, axis=0),
        np.concatenate(error_images, axis=0),
        np.array(final_feat)
    )
