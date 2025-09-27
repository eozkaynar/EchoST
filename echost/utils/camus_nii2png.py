
# Preprocesses the CAMUS dataset: converts NIfTI ED/ES and half_sequence images/masks
# into 256x256 PNG format, split into train/val/test folders.
# Outputs:
#   - Grayscale images (min-max normalized)
#   - Binary masks for LV wall (label==2)
#   - Binary masks for LV cavity (label==3)
#   - RGB color masks for visualization (label==1,2,3,4 → R,G,B,Y)

import os, glob, re
import numpy as np
import SimpleITK as sitk
import cv2


# Root(s) where patientXXXX folders live (searched recursively)
ROOTS       = ["echost/data/camus/database_nifti"]

# One patient id per line, e.g., 'patient0001'
TXT_TRAIN   = "echost/data/camus/database_split/subgroup_training.txt"
TXT_VAL     = "echost/data/camus/database_split/subgroup_validation.txt"
TXT_TEST    = "echost/data/camus/database_split/subgroup_testing.txt"

# Output path
OUT_IMG_TRAIN       = "echost/data/camus/train/imgs/CAMUS_TRAIN"
OUT_MSK_TRAIN       = "echost/data/camus/train/masks/CAMUS_TRAIN"
OUT_MSK_TRAIN_LV    = "echost/data/camus/train/masks/CAMUS_TRAIN_LV"

OUT_IMG_VAL         = "echost/data/camus/validate/imgs/CAMUS_VAL"
OUT_MSK_VAL         = "echost/data/camus/validate/masks/CAMUS_VAL"
OUT_MSK_VAL_LV      = "echost/data/camus/validate/masks/CAMUS_VAL_LV"

OUT_IMG_TEST        = "echost/data/camus/test/imgs/CAMUS_TEST"
OUT_MSK_TEST        = "echost/data/camus/test/masks/CAMUS_TEST"
OUT_MSK_TEST_LV     = "echost/data/camus/test/masks/CAMUS_TEST_LV"

# RGB mask output folders
OUT_RGB_TRAIN = "echost/data/camus/train/masks/CAMUS_TRAIN_RGB"
OUT_RGB_VAL   = "echost/data/camus/validate/masks/CAMUS_VAL_RGB"
OUT_RGB_TEST  = "echost/data/camus/test/masks/CAMUS_TEST_RGB"



# All PNGs will be saved at 256x256; images use bilinear, masks use nearest
TARGET_SIZE     = (256, 256) # (width, height)
SEQ_STRIDE      = 1 # keep every frame from half_sequence (increase to subsample)

def read_list(p):
    """Read split file into a set of patient ids."""
    s=set()
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                t=line.strip()
                if t and not t.startswith("#"):
                    s.add(t.split()[0])
    return s


def ensure_dirs(have_val: bool):
    os.makedirs(OUT_IMG_TRAIN, exist_ok=True); os.makedirs(OUT_MSK_TRAIN, exist_ok=True); os.makedirs(OUT_MSK_TRAIN_LV, exist_ok=True); os.makedirs(OUT_RGB_TRAIN, exist_ok=True)
    if have_val:
        os.makedirs(OUT_IMG_VAL, exist_ok=True); os.makedirs(OUT_MSK_VAL, exist_ok=True); os.makedirs(OUT_MSK_VAL_LV, exist_ok=True); os.makedirs(OUT_RGB_VAL, exist_ok=True)
    os.makedirs(OUT_IMG_TEST, exist_ok=True); os.makedirs(OUT_MSK_TEST, exist_ok=True); os.makedirs(OUT_MSK_TEST_LV, exist_ok=True); os.makedirs(OUT_RGB_TEST, exist_ok=True)


def read_nii(p):
    """Read .nii / .nii.gz and return squeezed numpy array (H,W) or (T,H,W)."""
    img = sitk.ReadImage(p)
    arr = sitk.GetArrayFromImage(img)
    return np.squeeze(arr)

def to_u8_minmax(x):
    """Min-max normalize a grayscale array to uint8 [0,255]."""
    x = x.astype(np.float32)
    rng = float(np.max(x) - np.min(x))
    if rng < 1e-6:
        return np.zeros_like(x, np.uint8)
    return (255.0 * (x - np.min(x)) / rng).astype(np.uint8)

def save_img_256(gray2d, out_path):
    """Resize image to 256x256 (bilinear), then write as uint8 PNG."""
    gray2d = cv2.resize(gray2d, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(out_path, to_u8_minmax(gray2d))

def save_mask_lvwall_256(lbl2d, out_path):
    """
    Build a binary LV-wall mask (label==2 → 255, else 0), resize to 256x256 with NEAREST,
    and write as uint8 PNG. Nearest preserves class labels.
    """
    m = (lbl2d == 2).astype(np.uint8) * 255
    m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, m)

def save_mask_lv_256(lbl2d, out_path):
    """
    Build a binary LV-cavity mask (label==1 → 255, else 0), resize to 256x256 with NEAREST,
    and write as uint8 PNG. Nearest preserves class labels.
    """
    m = (lbl2d == 1).astype(np.uint8) * 255
    m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, m)

def save_colored_mask(lbl2d, out_path):
    """
    Create a 3-channel color mask:
    - 1 (LV cavity) → Red
    - 2 (Myocardium) → Green
    - 3 (LA) → Blue
    - 4 (RA) → Yellow
    """
    color_mask = np.zeros((lbl2d.shape[0], lbl2d.shape[1], 3), dtype=np.uint8)
    color_mask[lbl2d == 1] = [255, 0, 0] # Red
    color_mask[lbl2d == 2] = [0, 255, 0] # Green
    color_mask[lbl2d == 3] = [0, 0, 255] # Blue
    color_mask[lbl2d == 4] = [255, 255, 0] # Yellow
    color_mask = cv2.resize(color_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, color_mask)

def base_no_ext(p):
    """Strip .nii/.nii.gz from filename."""
    b = os.path.basename(p)
    if b.endswith(".nii.gz"): return b[:-7]
    if b.endswith(".nii"): return b[:-4]
    return os.path.splitext(b)[0]

def norm_key(p):
    """
    Normalize sequence naming:
    - unify 'sequnce' → 'sequence'
    - strip trailing '_gt' (so image and mask share the same key)
    """
    b = base_no_ext(p).replace("sequnce", "sequence")
    if b.endswith("_gt"):
        b = b[:-3]
    return b

def split_out_dirs(pid, train_ids, val_ids, test_ids):
    """Return (img_out, msk_out, rgb_out) folders for a given patient id, based on split."""
    if pid in train_ids: return OUT_IMG_TRAIN, OUT_MSK_TRAIN, OUT_MSK_TRAIN_LV, OUT_RGB_TRAIN
    if pid in val_ids:   return OUT_IMG_VAL,  OUT_MSK_VAL, OUT_MSK_VAL_LV, OUT_RGB_VAL
    if pid in test_ids:  return OUT_IMG_TEST, OUT_MSK_TEST, OUT_MSK_TEST_LV, OUT_RGB_TEST
    return None, None, None,None


def process_patient(pdir, pid, train_ids, val_ids, test_ids, counters):
    """Convert ED/ES and half_sequence for a single patient into 256x256 PNGs."""
    img_out, msk_out, mask_out_lv, rgb_out = split_out_dirs(pid, train_ids, val_ids, test_ids)
    if img_out is None:
        return
    # ----- ED / ES single-frame -----
    for view in ["2CH","4CH"]:
        for ph in ["ED","ES"]:
            img_c = glob.glob(os.path.join(pdir, f"{pid}_{view}_{ph}.nii*"))
            msk_c = glob.glob(os.path.join(pdir, f"{pid}_{view}_{ph}_gt.nii*"))
            if not img_c or not msk_c:
                continue
            img_np = read_nii(img_c[0])
            lbl_np = read_nii(msk_c[0])
            if img_np.ndim != 2 or lbl_np.ndim != 2:
                print(f"[WARN] Unexpected dims for {pid} {view} {ph}: {img_np.shape}/{lbl_np.shape}")
                continue
            name = f"{pid}_{view}_{ph}.png"
            save_img_256(img_np, os.path.join(img_out, name))
            save_mask_lvwall_256(lbl_np, os.path.join(msk_out, name))
            save_mask_lv_256(lbl_np, os.path.join(mask_out_lv, name))
            save_colored_mask(lbl_np, os.path.join(rgb_out, name))
            counters["pairs"] += 1

    # ----- half_sequence time-series -----
    # Collect images (exclude *_gt) — accept both 'sequence' and 'sequnce'
    seq_imgs = []
    seq_imgs += glob.glob(os.path.join(pdir, f"{pid}_*half*sequence*.nii*"))
    seq_imgs += glob.glob(os.path.join(pdir, f"{pid}_*half*sequnce*.nii*"))
    seq_imgs = [p for p in seq_imgs if not re.search(r"_gt(\.|$)", os.path.basename(p))]

    # Collect masks
    seq_msks = []
    seq_msks += glob.glob(os.path.join(pdir, f"{pid}_*half*sequence*_gt.nii*"))
    seq_msks += glob.glob(os.path.join(pdir, f"{pid}_*half*sequnce*_gt.nii*"))
    msk_map = {norm_key(m): m for m in seq_msks}

    for ip in seq_imgs:
        key = norm_key(ip)
        mp = msk_map.get(key)
        if mp is None:
         # Some patients simply don't have GT for half_sequence → skip
            continue

        img_np = read_nii(ip)
        lbl_np = read_nii(mp)
        if img_np.ndim == 2: img_np = img_np[None, ...]
        if lbl_np.ndim == 2: lbl_np = lbl_np[None, ...]
        T = img_np.shape[0]
        step = max(1, int(SEQ_STRIDE))
        for t in range(0, T, step):
            name = f"{key}_f{t:03d}.png"
            save_img_256(img_np[t], os.path.join(img_out, name))
            save_mask_lvwall_256(lbl_np[t], os.path.join(msk_out, name))
            save_mask_lv_256(lbl_np[t], os.path.join(mask_out_lv, name))
            save_colored_mask(lbl_np[t], os.path.join(rgb_out, name))
            counters["pairs"] += 1

def main():
    # Load splits, create output folders
    train_ids   = read_list(TXT_TRAIN)
    val_ids     = read_list(TXT_VAL)
    test_ids    = read_list(TXT_TEST)
    print(f"Split sizes → train:{len(train_ids)} val:{len(val_ids)} test:{len(test_ids)}")
    ensure_dirs(have_val=True)

    # Find all patient folders recursively under ROOTS
    patient_dirs = []
    for root in ROOTS:
        patient_dirs += sorted(glob.glob(os.path.join(root, "**", "patient*"), recursive=True))

    # Convert
    counters = {"pairs": 0}
    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        process_patient(pdir, pid, train_ids, val_ids, test_ids, counters)


    print(f" Saved image/mask pairs (256x256, uint8): {counters['pairs']}")

if __name__ == "__main__":
    main()
