"""Dataset path and pairing helpers (ported from CRISP).

The official AnomalyDiffusion dataset loader assumed filenames like
``000.png`` and sorted with ``int(x[:3])``.  That breaks on VisA or any
dataset whose images/masks are not exactly three numeric characters, and it
can silently pair the wrong image with the wrong mask after sorting.

These helpers:
  * filter to real image extensions;
  * sort by numeric stem when possible, otherwise lexicographically;
  * parse ``name-anomaly.txt`` robustly;
  * pair image/mask by file stem (not by list index);
  * normalize GT masks for MVTec and VisA (VisA masks may use 0/1 labels).
"""

from __future__ import annotations

import os

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def sort_key_by_numeric_prefix(filename: str):
    """Numeric-aware key: ``2.png`` sorts before ``10.png``."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def parse_name_anomaly_file(path: str):
    """Parse ``sample+anomaly`` lines; skip blanks/bad rows/trailing newline."""
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '+' not in line:
                continue
            sample, anomaly = line.split('+', 1)
            sample = sample.strip()
            anomaly = anomaly.strip()
            if sample and anomaly:
                pairs.append((sample, anomaly))
    return pairs


def list_image_files(dir_path: str):
    """Return image files only, sorted deterministically by numeric stem."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f'image directory not found: {dir_path}')
    files = [
        f for f in os.listdir(dir_path)
        if f.lower().endswith(_IMG_EXTS)
    ]
    files.sort(key=sort_key_by_numeric_prefix)
    return files


def list_image_files_recursive(dir_path: str):
    """Return image paths recursively (e.g. train/good/all/*.jpg)."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f'image directory not found: {dir_path}')
    files = []
    for root, _, names in os.walk(dir_path):
        for name in names:
            if name.lower().endswith(_IMG_EXTS):
                files.append(os.path.join(root, name))
    files.sort(key=lambda p: (sort_key_by_numeric_prefix(os.path.basename(p)), p))
    return files


def pair_image_mask_files(img_dir: str, mask_dir: str):
    """Pair images and masks by stem, never by independent list index.

    Supports datasets where image and mask extensions differ
    (e.g. image ``001.JPG`` + mask ``001.png``) and raises a clear error
    instead of silently producing mismatched pairs.
    """
    img_files = list_image_files(img_dir)
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f'mask directory not found: {mask_dir}')
    mask_files = [
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(_IMG_EXTS)
    ]
    masks_by_stem = {
        os.path.splitext(f)[0]: f for f in mask_files
    }

    img_paths, mask_paths = [], []
    missing = []
    for image_name in img_files:
        stem = os.path.splitext(image_name)[0]
        mask_name = masks_by_stem.get(stem)
        if mask_name is None:
            missing.append(image_name)
            continue
        img_paths.append(os.path.join(img_dir, image_name))
        mask_paths.append(os.path.join(mask_dir, mask_name))

    if missing:
        raise FileNotFoundError(
            f'[dataset] {len(missing)} image(s) in {img_dir} have no matching '
            f'mask stem in {mask_dir}; first missing: {missing[:5]}')
    if not img_paths:
        raise FileNotFoundError(
            f'[dataset] no image/mask pairs found in {img_dir} / {mask_dir}')
    return img_paths, mask_paths


def normalize_gt_mask_array(arr):
    """Binarize GT masks to float32 {0, 1}.

    MVTec masks are 0/255; VisA masks may be 0/1 or small multi-label ints.
    Plain ``/255`` would erase VisA defects, so use ``> 0`` for those.
    """
    import numpy as np

    mask = np.asarray(arr, dtype=np.float32)
    if mask.max() <= 1.0:
        return (mask > 0).astype(np.float32)
    scaled = mask / 255.0
    binary = (scaled >= 0.5).astype(np.float32)
    if binary.max() == 0.0 and (mask > 0).any():
        return (mask > 0).astype(np.float32)
    return binary
