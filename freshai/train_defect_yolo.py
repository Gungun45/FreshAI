"""
FreshAI - Stage 3: YOLO Defect Detection & Segmentation Trainer
Trains YOLOv8 Detection or YOLOv8-Segmentation models on produce defects.

Defect Classes:
0: Bruise
1: Black Spot
2: Crack
3: Mold
4: Rot
5: Fungal Spot
6: Wrinkle
7: Discoloration
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from freshai.defect_detector import DEFECT_CLASSES


def generate_synthetic_defect_dataset(
    output_dir: str = "datasets/produce_defects",
    num_train: int = 60,
    num_val: int = 15,
) -> str:
    """
    Synthesizes produce images with polygon segmentation annotations
    for the 8 defect types to bootstrap YOLO segmentation training.
    """
    root = Path(output_dir)
    for split in ["train", "val"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    produce_base_colors = [
        (220, 38, 38),   # Red Apple / Tomato
        (34, 197, 94),   # Green Pepper / Cucumber
        (234, 179, 8),   # Banana / Lemon
        (249, 115, 22),  # Orange / Carrot
        (168, 85, 247),  # Eggplant / Plum
    ]

    for split, count in [("train", num_train), ("val", num_val)]:
        for idx in range(count):
            w, h = 416, 416
            img = Image.new("RGB", (w, h), color=(245, 245, 245))
            draw = ImageDraw.Draw(img)

            # Draw healthy produce body
            body_color = random.choice(produce_base_colors)
            cx, cy, rx, ry = 208, 208, random.randint(110, 160), random.randint(110, 160)
            bbox_body = [cx - rx, cy - ry, cx + rx, cy + ry]
            draw.ellipse(bbox_body, fill=body_color, outline=(30, 30, 30), width=2)

            # Add 1 to 3 defects
            num_defects = random.randint(1, 3)
            label_lines = []

            for _ in range(num_defects):
                cls_id = random.randint(0, len(DEFECT_CLASSES) - 1)
                defect_name = DEFECT_CLASSES[cls_id]

                # Defect center inside produce
                dx = cx + random.randint(-int(rx * 0.6), int(rx * 0.6))
                dy = cy + random.randint(-int(ry * 0.6), int(ry * 0.6))
                dr = random.randint(12, 35)

                # Color per defect
                if defect_name in ["Rot", "Black Spot", "Fungal Spot"]:
                    d_color = (random.randint(20, 50), random.randint(15, 35), random.randint(10, 25))
                elif defect_name == "Mold":
                    d_color = (random.randint(210, 240), random.randint(210, 240), random.randint(220, 250))
                elif defect_name == "Bruise":
                    d_color = (max(0, body_color[0] - 50), max(0, body_color[1] - 40), max(0, body_color[2] - 30))
                elif defect_name == "Crack":
                    d_color = (180, 40, 40)
                else:
                    d_color = (random.randint(180, 210), random.randint(140, 170), 50)

                # Draw defect and build polygon
                poly_pts = []
                num_pts = random.randint(6, 10)
                for p_idx in range(num_pts):
                    angle = (2 * np.pi * p_idx) / num_pts
                    rad = dr * random.uniform(0.7, 1.3)
                    px = int(np.clip(dx + rad * np.cos(angle), 5, w - 5))
                    py = int(np.clip(dy + rad * np.sin(angle), 5, h - 5))
                    poly_pts.append((px, py))

                draw.polygon(poly_pts, fill=d_color, outline=(20, 20, 20))

                # Normalize coordinates for YOLO-seg format:
                # class_id x1 y1 x2 y2 ... xn yn (normalized 0.0 - 1.0)
                norm_coords = []
                for px, py in poly_pts:
                    norm_coords.append(f"{px / w:.4f}")
                    norm_coords.append(f"{py / h:.4f}")

                label_lines.append(f"{cls_id} " + " ".join(norm_coords))

            img_filename = f"defect_sample_{split}_{idx:04d}.jpg"
            lbl_filename = f"defect_sample_{split}_{idx:04d}.txt"

            img.save(root / "images" / split / img_filename, quality=92)
            with open(root / "labels" / split / lbl_filename, "w") as lf:
                lf.write("\n".join(label_lines) + "\n")

    # Write data.yaml
    data_yaml = root / "data_defect.yaml"
    names_dict = {i: name for i, name in enumerate(DEFECT_CLASSES)}
    yaml_content = f"""path: {root.resolve().as_posix()}
train: images/train
val: images/val
names:
"""
    for i, name in enumerate(DEFECT_CLASSES):
        yaml_content += f"  {i}: {name}\n"

    with open(data_yaml, "w") as yf:
        yf.write(yaml_content)

    print(f"  Synthetic defect dataset generated at: {root.resolve()}")
    return str(data_yaml)


def train_defect_yolo(
    data_yaml: Optional[str] = None,
    epochs: int = 15,
    imgsz: int = 416,
    batch: int = 16,
    is_segmentation: bool = True,
    base_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Train YOLOv8 detection or segmentation on produce defects."""
    if not ULTRALYTICS_AVAILABLE:
        print("[Error] ultralytics package is required.")
        return {}

    if data_yaml is None or not os.path.exists(data_yaml):
        data_yaml = generate_synthetic_defect_dataset()

    model_weight = base_model or ("yolov8n-seg.pt" if is_segmentation else "yolov8n.pt")
    print(f"\n{'='*60}")
    print(f"  Training YOLO Defect {'Segmentation' if is_segmentation else 'Detection'}")
    print(f"  Base weights: {model_weight} | Epochs: {epochs} | ImgSz: {imgsz}")
    print(f"{'='*60}\n")

    model = YOLO(model_weight)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/defect",
        name="produce_defects",
        verbose=True,
    )
    return {"results": str(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--segment", action="store_true", default=True)
    parser.add_argument("--data", type=str, default=None)
    args = parser.parse_args()

    train_defect_yolo(data_yaml=args.data, epochs=args.epochs, batch=args.batch, is_segmentation=args.segment)
