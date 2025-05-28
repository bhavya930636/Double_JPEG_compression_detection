# DOUBLE JPEG COMPRESSION DETECTION FOR DISTINGUISHABLE BLOCKS IN IMAGES COMPRESSED WITH SAME QUANTIZATION MATRIX

This project performs error analysis on JPEG-compressed images using the UCID dataset. It identifies rounding and truncation errors in DCT coefficients, processes them, and trains a Support Vector Machine (SVM) model for classification of iamges as double or simple compressed.

---
 
[Demo Video Link](https://drive.google.com/file/d/1vbWHRaWgYg4m738mgKhVOpNqpbUBxr0F/view?usp=drive_link)


## 📁 Folder Structure

```
project_root/
├── *.py ``                # Python scripts
├── *.m                     # MATLAB scripts
└── ...

├── data/                   # Expects `ucid.v2` folder here (level equivalent to root)

```

---

## 📦 Setup Instructions

### 1. Dataset Setup

- Download the **UCID dataset** (e.g., from [here](https://github.com/girfa/ColorImageDatasets)).
- Place it **outside** the project root directory like so:

```
../data/ucid.v2/
```

---

### 2. Install Python Dependencies

Install using:

```bash
pip install -r requirements.txt
```

> MATLAB is required for intermediate processing steps.

---

## 🚀 Running the Pipeline

### Step 1: Generate Grayscale Images

```bash
python data_maker.py
```

This converts all UCID images to grayscale JPEGs using a specific quality factor.

---

### Step 2: Generate Patches

Run `patch_maker.py` **twice**:

```bash
python patch_maker.py
```

- First with `train=True`
- Then with `train=False`

Modify the path variable inside the script before each run to ensure correct file I/O.

---

### Step 3: MATLAB DCT Error Computation

Run this in terminal:

```bash
matlab -nodisplay -nosplash -r "save_error_images; exit"
```

This will:

- Compute error blocks  
- Extract DCT coefficients  
- Separate rounding and truncation errors  
- Save `.mat` files for training and testing  

---

### Step 4: Convert `.mat` to `.npz`

```bash
python run_all_mat_2_npz.py
```

Converts MATLAB `.mat` files to NumPy `.npz` format.

---

### Step 5: Train and Evaluate SVM

```bash
python svm.py
```

Trains an SVM classifier on the error features and evaluates the performance.

---

### Step 6: Visualization

To save error blocks as images:

```bash
python visualise.py
```

To generate plots for rounding and truncation error distributions:

```bash
python visualise_r_t.py
```

---

## 📌 Notes

- MATLAB is required for DCT error extraction.
- Ensure file paths are correctly set in each script before running.
- Tested with Python 3.8+ and MATLAB R2021b.

---

## 📄 License

This project is intended for academic and research use only.
