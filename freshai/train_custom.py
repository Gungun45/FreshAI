"""
FreshAI - Custom YOLO Training Script
Trains YOLOv8n on Kaggle fruit/vegetable dataset for real produce detection
"""

import os
import sys
import yaml
from pathlib import Path


DEFAULT_DATASET_YAML = "datasets/produce_yolo/vegetable-fruit-yolo/data.yaml"


def find_dataset_yaml(requested_path=None):
    """Use the real Kaggle detection dataset, never an arbitrary demo YAML."""
    if requested_path:
        return requested_path if os.path.isfile(requested_path) else None
    return DEFAULT_DATASET_YAML if os.path.isfile(DEFAULT_DATASET_YAML) else None


def fix_yaml_paths(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    dataset_dir = str(Path(yaml_path).parent.resolve())
    fixed_yaml = os.path.join(dataset_dir, "data_fixed.yaml")
    data["path"] = dataset_dir

    for split in ["train", "val", "test"]:
        if split in data:
            p = str(data[split])
            if os.path.isabs(p) and not os.path.isdir(p):
                # Try to find relative
                for suffix in [f"images/{split}", f"{split}/images", split]:
                    candidate = os.path.join(dataset_dir, suffix)
                    if os.path.isdir(candidate):
                        data[split] = candidate
                        break
            elif not os.path.isabs(p):
                candidate = os.path.join(dataset_dir, p)
                if os.path.isdir(candidate):
                    data[split] = candidate

    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("Fixed YAML saved to:", fixed_yaml)
    print("  Classes (nc=" + str(data.get("nc")) + "):", data.get("names"))
    print("  Train:", data.get("train"))
    print("  Val:  ", data.get("val"))
    return fixed_yaml


def train(epochs=30, img_size=640, batch=8, model="yolov8n.pt", data=None):
    from ultralytics import YOLO

    yaml_path = find_dataset_yaml(data)
    if not yaml_path:
        print("ERROR: Could not find data.yaml in datasets/ folder")
        sys.exit(1)

    print("\n=== FreshAI Produce Detector Training ===")
    print("Dataset YAML:", yaml_path)
    fixed_yaml = fix_yaml_paths(yaml_path)

    # Count images
    with open(fixed_yaml) as f:
        d = yaml.safe_load(f)
    for split in ["train", "val"]:
        p = d.get(split, "")
        if os.path.isdir(str(p)):
            imgs = list(Path(str(p)).glob("*.jpg")) + list(Path(str(p)).glob("*.png"))
            print("  " + split + ":", len(imgs), "images")

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    device_label = "GPU (CUDA - GTX 1650)" if device == "0" else "CPU"
    print("\nModel:", model, "| Epochs:", epochs, "| Img:", img_size, "| Batch:", batch, "| Device:", device_label)
    print("Starting YOLO training...\n")

    model_obj = YOLO(model)
    model_obj.train(
        data=fixed_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name="produce_detector",
        project="runs/freshai_detect",
        patience=10,
        save=True,
        val=True,
        verbose=True,
        device=device,
        workers=2,
    )

    best = "runs/freshai_detect/produce_detector/weights/best.pt"
    if os.path.exists(best):
        print("\n=== Training Complete! ===")
        print("Best model:", best)
    else:
        print("Check runs/freshai_detect/produce_detector/weights/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--data", type=str, default=None, help="Path to a YOLO data.yaml file")
    args = p.parse_args()
    train(args.epochs, args.imgsz, args.batch, args.model, args.data)
