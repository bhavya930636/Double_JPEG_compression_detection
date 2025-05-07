# Double JPEG Compression Detection

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

## Dataset

Place UCID dataset images under `data/dataset/`.

---

## Step 1: Image Compression

Convert images to grayscale and apply JPEG compression at multiple levels.

```
python data_maker.py
```

---

## Step 2: Patch Extraction

Extract stable/unstable 8x8 patches using DCT comparison.

```
python patch_make.py
```

---

## Step 3: Feature Extraction (MATLAB)

Extract features from each patch and generate error matrices.

```
save_error_images
```

---

## Step 4: Convert `.mat` to `.npz`

Convert all `.mat` files to `.npz` files.

```
python run_all_mat_2_npz.py
```

---

## Step 5: Train SVM Classifier

Train and evaluate the classifier.

```
python svm.py
```

---

## Step 6: Visualize Errors


Visualize error blocks.

```
python visualize.py
```

Plot rounding and truncation error distributions.

```
python visualize_r_t.py
```

---
