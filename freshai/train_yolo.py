"""
FreshAI - Training Pipeline for YOLO Fruit, Vegetable & Plant Detection
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from freshai.dataset_utils import generate_synthetic_demo_dataset, generate_data_yaml


def train_yolo_model(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 16,
    project: str = "runs/freshai_detect",
    name: str = "custom_produce_model",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train / Fine-tune YOLO model on custom fruit/veg/plant dataset.
    """
    if YOLO is None:
        raise ImportError("Ultralytics package is not installed.")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    print("============================================================")
    print("  FreshAI YOLO Detection Training Pipeline")
    print(f"  Base Model: {base_model}")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Epochs:     {epochs} | Image Size: {imgsz} | Batch: {batch_size}")
    print("============================================================")

    model = YOLO(base_model)

    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "workers": 0,
        "project": project,
        "name": name,
        "exist_ok": True,
        "verbose": True,
    }
    if device is not None:
        train_kwargs["device"] = device

    # Start training
    results = model.train(**train_kwargs)

    # Validate trained model
    metrics = model.val()

    # Locate saved weights
    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else os.path.join(project, name)
    best_weight_path = os.path.join(save_dir, "weights", "best.pt")
    last_weight_path = os.path.join(save_dir, "weights", "last.pt")

    if not os.path.exists(best_weight_path):
        # Fallback search inside runs
        for root, _, files in os.walk("runs"):
            if "best.pt" in files:
                best_weight_path = os.path.join(root, "best.pt")
                break

    summary = {
        "best_weights": best_weight_path if os.path.exists(best_weight_path) else None,
        "last_weights": last_weight_path if os.path.exists(last_weight_path) else None,
        "save_dir": save_dir,
        "map50": getattr(metrics.box, "map50", 0.0) if hasattr(metrics, "box") else 0.0,
        "map50_95": getattr(metrics.box, "map", 0.0) if hasattr(metrics, "box") else 0.0,
        "precision": getattr(metrics.box, "mp", 0.0) if hasattr(metrics, "box") else 0.0,
        "recall": getattr(metrics.box, "mr", 0.0) if hasattr(metrics, "box") else 0.0,
    }

    print("\n------------------------------------------------------------")
    print("  Training Completed Successfully!")
    print(f"  Best Weights: {summary['best_weights']}")
    print(f"  mAP@50:       {summary['map50']:.4f}")
    print(f"  mAP@50-95:    {summary['map50_95']:.4f}")
    print("------------------------------------------------------------\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="FreshAI - Train YOLO Detection Model")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--demo", action="store_true", help="Generate and train on synthetic demo dataset")
    parser.add_argument("--project", type=str, default="runs/freshai_detect", help="Output project directory")

    args = parser.parse_args()

    data_yaml = args.data
    if args.demo or data_yaml is None:
        print("Generating FreshAI demo dataset for training verification...")
        data_yaml = generate_synthetic_demo_dataset(output_dir="datasets/freshai_demo", num_train=30, num_val=8)

    train_yolo_model(
        data_yaml=data_yaml,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        project=args.project,
    )


if __name__ == "__main__":
    main()