# EchoST: Echocardiographic Segmentation and Strain Tools

---

## Project Structure
BURAYI GÜNCELLE
```
echost/
├── datasets/
│   └── echostrain.py            # PyTorch dataset class for grayscale image + mask loading
├── scripts/
│   ├── segmentation.py          # Segmentation training/evaluation script (WIP)
├── utils/
│   ├── camus_nii2png.py         # Main script for CAMUS NIfTI → PNG conversion
│   ├── mask_utils.py            # Mask processing helpers (e.g. save RGB masks)
│   └── metrics.py               # Evaluation metrics
```

---

## Setup Instructions

### 1. Clone the repository and move into it

```bash
git clone https://github.com/eozkaynar/echost.git
```

### 2. Create and activate a virtual environment (Linux)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Install the package locally

```bash
pip install -e .
```

---


## 5. Directory Structure

The dataset should be placed under the following structure for camus: 

```
data/
├── database_nifti/             # Original CAMUS .nii.gz files (recursive structure)
├── database_split/
│   ├── subgroup_training.txt
│   ├── subgroup_validation.txt
│   └── subgroup_testing.txt
```

---

## 6. PYTHONPATH Configuration

To make `echost` modules importable via `python -m` syntax, export the project root as PYTHONPATH:

```bash
export PYTHONPATH=$(pwd)
```

You can also add this line to your `.bashrc` or `.zshrc` for persistence.

---

##  Convert CAMUS NIfTI to PNG (ED/ES + sequences)

Make sure your CAMUS dataset is placed under `data/database_nifti/`, and your split txt files are in `data/database_split/`.

To run the conversion script:

```bash
python -m echost.utils.camus_nii2png
```

The output folders will be created automatically:
- Grayscale images → `data/train/imgs/...`
- Binary masks     → `data/train/masks/...`
- RGB masks        → `data/train/masks/..._RGB`

---

##  RGB Mask Colors

The color-coded masks use the following mapping:

| Label         | Class        | RGB Color       |
|---------------|--------------|------------------|
| 1             | Left Atrium  | Red `[255, 0, 0]` |
| 2             | Myocardium   | Green `[0, 255, 0]` |
| 3             | LV Cavity    | Blue `[0, 0, 255]` |


---



---

For any issues, please contact [Your Name] or open an issue in the repository.