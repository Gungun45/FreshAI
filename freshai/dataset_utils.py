"""
FreshAI - Dataset Preparation and Utilities for Fruit, Vegetable & Plant YOLO Training
"""

import os
import yaml
import numpy as np
from PIL import Image, ImageDraw
import random
from typing import List, Dict, Tuple, Optional


DEFAULT_FRESHAI_CLASSES = [
    "Tomato",
    "Apple",
    "Banana",
    "Orange",
    "Onion",
    "Potato",
    "Bell_Pepper",
    "Broccoli",
    "Carrot",
    "Cucumber",
    "Strawberry",
    "Lemon",
    "Garlic",
    "Growing_Fruit",
    "Growing_Vegetable",
    "Plant_Leaf",
]



def create_yolo_dataset_structure(base_dir: str = "datasets/freshai_custom") -> Dict[str, str]:
    """
    Creates standard YOLO directory structure:
    base_dir/
      images/train, images/val, images/test
      labels/train, labels/val, labels/test
    """
    dirs = {
        "train_img": os.path.join(base_dir, "images", "train"),
        "val_img": os.path.join(base_dir, "images", "val"),
        "test_img": os.path.join(base_dir, "images", "test"),
        "train_lbl": os.path.join(base_dir, "labels", "train"),
        "val_lbl": os.path.join(base_dir, "labels", "val"),
        "test_lbl": os.path.join(base_dir, "labels", "test"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def generate_data_yaml(
    dataset_dir: str,
    classes: List[str] = DEFAULT_FRESHAI_CLASSES,
    yaml_path: Optional[str] = None,
) -> str:
    """
    Generates data.yaml configuration for YOLO training.
    """
    abs_dataset_dir = os.path.abspath(dataset_dir)
    data_dict = {
        "path": abs_dataset_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
    }

    if yaml_path is None:
        yaml_path = os.path.join(dataset_dir, "data.yaml")

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)

    return yaml_path


def generate_synthetic_demo_dataset(
    output_dir: str = "datasets/freshai_demo",
    num_train: int = 40,
    num_val: int = 10,
    img_size: Tuple[int, int] = (640, 640),
) -> str:
    """
    Generates a synthetic fruit/vegetable/plant dataset with proper bounding boxes
    and YOLO label files to immediately test the fine-tuning pipeline without downloading 10GB.
    """
    dirs = create_yolo_dataset_structure(output_dir)
    classes = ["Tomato", "Apple", "Growing_Fruit", "Bell_Pepper", "Plant_Leaf"]
    
    color_map = {
        0: (220, 38, 38),   # Tomato (Red)
        1: (185, 28, 28),   # Apple (Deep Red)
        2: (132, 204, 22),  # Growing_Fruit (Greenish Yellow)
        3: (16, 185, 129),  # Bell_Pepper (Emerald)
        4: (34, 197, 94),   # Plant_Leaf (Leaf Green)
    }

    splits = [
        ("train", num_train, dirs["train_img"], dirs["train_lbl"]),
        ("val", num_val, dirs["val_img"], dirs["val_lbl"]),
    ]

    for split_name, count, img_dir, lbl_dir in splits:
        for idx in range(count):
            # Create synthetic agricultural background (vine, leaves, soil)
            bg_color = (
                random.randint(220, 245),
                random.randint(230, 250),
                random.randint(215, 235),
            )
            img = Image.new("RGB", img_size, color=bg_color)
            draw = ImageDraw.Draw(img)

            # Draw background plant vine/branches
            for _ in range(random.randint(3, 6)):
                p1 = (random.randint(0, img_size[0]), random.randint(0, img_size[1]))
                p2 = (random.randint(0, img_size[0]), random.randint(0, img_size[1]))
                draw.line([p1, p2], fill=(60, 120, 40), width=random.randint(4, 12))

            # Add objects
            num_objects = random.randint(1, 5)
            yolo_labels = []

            for _ in range(num_objects):
                cls_id = random.randint(0, len(classes) - 1)
                box_w = random.randint(80, 200)
                box_h = random.randint(80, 200)
                x1 = random.randint(10, img_size[0] - box_w - 10)
                y1 = random.randint(10, img_size[1] - box_h - 10)
                x2 = x1 + box_w
                y2 = y1 + box_h

                obj_color = color_map[cls_id]
                # Draw produce shape
                draw.ellipse([x1, y1, x2, y2], fill=obj_color, outline=(40, 40, 40), width=2)
                
                # If Tomato or Apple, draw green stem
                if cls_id in [0, 1, 2]:
                    stem_x = (x1 + x2) // 2
                    draw.line([(stem_x, y1), (stem_x - 5, y1 - 15)], fill=(34, 197, 94), width=4)

                # Normalized YOLO format: class_id center_x center_y width height
                norm_cx = (x1 + x2) / (2.0 * img_size[0])
                norm_cy = (y1 + y2) / (2.0 * img_size[1])
                norm_w = box_w / float(img_size[0])
                norm_h = box_h / float(img_size[1])

                yolo_labels.append(f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

            # Save image & label file
            img_filename = f"{split_name}_{idx:04d}.jpg"
            lbl_filename = f"{split_name}_{idx:04d}.txt"

            img.save(os.path.join(img_dir, img_filename), quality=95)
            with open(os.path.join(lbl_dir, lbl_filename), "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_labels) + "\n")

    yaml_path = generate_data_yaml(output_dir, classes=classes)
    return yaml_path