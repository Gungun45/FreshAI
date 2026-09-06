"""
FreshAI - Multi-Scale Dataset Generator
Combines:
  1. Cluster/Group Produce images (from freshai_produce_focus)
  2. Single Isolated Close-up Produce images (from fruits_360)
Outputs to: datasets/freshai_produce_multiscale
"""

import os
import cv2
import glob
import shutil
import random
import yaml
import numpy as np
from pathlib import Path

random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "freshai_produce_multiscale"

# Target classes
CLASSES = ["Onion", "Tomato", "Watermelon"]
CLASS_MAP = {"Onion": 0, "Tomato": 1, "Watermelon": 2}


def get_single_fruit_bbox(img_path):
    """Calculate tight bounding box for Fruits-360 white background image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 250  # White background threshold
    coords = np.argwhere(mask)
    if len(coords) < 50:  # empty or corrupted
        return [0.5, 0.5, 0.8, 0.8]
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    # add a tiny padding
    pad = 2
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0) / float(w)
    bh = (y1 - y0) / float(h)
    return [round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4)]


def prepare_dataset():
    print(f"Generating Multi-Scale Dataset at: {OUTPUT_DIR}")

    train_img_dir = OUTPUT_DIR / "images" / "train"
    val_img_dir = OUTPUT_DIR / "images" / "val"
    train_lbl_dir = OUTPUT_DIR / "labels" / "train"
    val_lbl_dir = OUTPUT_DIR / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy existing cluster dataset ───────────────────────────────────────
    src_focus = PROJECT_ROOT / "datasets" / "freshai_produce_focus"
    copied_clusters = 0

    for split in ["train", "val"]:
        src_imgs = list((src_focus / "images" / split).glob("*.jpg")) + list((src_focus / "images" / split).glob("*.png"))
        dst_i = train_img_dir if split == "train" else val_img_dir
        dst_l = train_lbl_dir if split == "train" else val_lbl_dir

        for img_p in src_imgs:
            lbl_p = src_focus / "labels" / split / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy2(img_p, dst_i / f"cluster_{img_p.name}")
                shutil.copy2(lbl_p, dst_l / f"cluster_{lbl_p.name}")
                copied_clusters += 1

    print(f"Copied {copied_clusters} cluster/market images.")

    # ── 2. Add Single Fruits from fruits_360 ──────────────────────────────────
    f360_root = PROJECT_ROOT / "datasets" / "fruits_360" / "train"

    single_sources = {
        "Tomato": ["Tomato 1", "Tomato 2"],
        "Onion": ["Onion Red", "Onion White", "Onion Red Peeled"],
        "Watermelon": ["Melon Piel de Sapo"],
    }

    added_singles = 0
    for cls_name, folders in single_sources.items():
        cls_id = CLASS_MAP[cls_name]
        cls_images = []
        for folder in folders:
            p = f360_root / folder
            if p.exists():
                cls_images.extend(list(p.glob("*.jpg")))

        random.shuffle(cls_images)
        # Take up to 600 images per class for balanced multi-scale learning
        selected = cls_images[:600]
        n_train = int(len(selected) * 0.85)

        train_set = selected[:n_train]
        val_set = selected[n_train:]

        for img_p in train_set:
            bbox = get_single_fruit_bbox(img_p)
            if bbox:
                dst_name = f"single_{cls_name.lower()}_{img_p.stem}"
                shutil.copy2(img_p, train_img_dir / f"{dst_name}.jpg")
                (train_lbl_dir / f"{dst_name}.txt").write_text(f"{cls_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                added_singles += 1

        for img_p in val_set:
            bbox = get_single_fruit_bbox(img_p)
            if bbox:
                dst_name = f"single_{cls_name.lower()}_{img_p.stem}"
                shutil.copy2(img_p, val_img_dir / f"{dst_name}.jpg")
                (val_lbl_dir / f"{dst_name}.txt").write_text(f"{cls_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                added_singles += 1

        print(f"Added {len(selected)} single-fruit images for {cls_name} ({len(train_set)} train, {len(val_set)} val).")

    # ── 3. Write data.yaml ───────────────────────────────────────────────────
    data_yaml = {
        "path": str(OUTPUT_DIR),
        "train": str(train_img_dir),
        "val": str(val_img_dir),
        "names": {0: "Onion", 1: "Tomato", 2: "Watermelon"},
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    total_train = len(list(train_img_dir.glob("*.jpg")))
    total_val = len(list(val_img_dir.glob("*.jpg")))
    print(f"\n Multi-Scale Dataset Ready!")
    print(f"  Total Train Images: {total_train}")
    print(f"  Total Val Images:   {total_val}")
    print(f"  YAML Config:        {yaml_path}")
    return str(yaml_path)


if __name__ == "__main__":
    prepare_dataset()
