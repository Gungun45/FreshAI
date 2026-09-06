"""
FreshAI - Training Pipeline for YOLO Fruits-360 Produce Classification
"""

import os
import sys
import shutil
import argparse
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def prepare_fruits360_structure(raw_dir: str = "datasets/fruits_360") -> str:
    """
    Ensures standard train / val directory structure for YOLO classification.
    Fruits-360 typically has 'Training' and 'Test' or 'fruits-360/Training' and 'fruits-360/Test'.
    """
    abs_dir = os.path.abspath(raw_dir)
    
    # Check possible nested structures
    candidates = [
        abs_dir,
        os.path.join(abs_dir, "fruits-360_dataset", "fruits-360"),
        os.path.join(abs_dir, "fruits-360"),
        os.path.join(abs_dir, "fruit360"),
    ]
    
    target_root = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "Training")) or os.path.exists(os.path.join(cand, "train")):
            target_root = cand
            break

    if target_root is None:
        return abs_dir

    # Standardize 'Training' -> 'train' and 'Test' -> 'val' if needed
    train_dir = os.path.join(target_root, "train")
    val_dir = os.path.join(target_root, "val")
    training_dir = os.path.join(target_root, "Training")
    test_dir = os.path.join(target_root, "Test")

    if not os.path.exists(train_dir) and os.path.exists(training_dir):
        try:
            os.rename(training_dir, train_dir)
        except Exception:
            pass

    if not os.path.exists(val_dir) and os.path.exists(test_dir):
        try:
            os.rename(test_dir, val_dir)
        except Exception:
            pass

    return target_root


def train_classifier(
    data_dir: str = "datasets/fruits_360",
    base_model: str = "yolov8s-cls.pt",
    epochs: int = 20,
    imgsz: int = 224,
    batch_size: int = 32,
    project: str = "runs/freshai_classify",
    name: str = "fruits360_model",
) -> Dict[str, Any]:
    """
    Train YOLOv8 Classification on Fruits-360 dataset.
    """
    if YOLO is None:
        raise ImportError("Ultralytics package is not installed.")

    dataset_path = prepare_fruits360_structure(data_dir)

    print("============================================================")
    print("  FreshAI — Fruits-360 YOLO Classifier Training")
    print(f"  Base Model: {base_model}")
    print(f"  Dataset:    {dataset_path}")
    print(f"  Epochs:     {epochs} | Batch: {batch_size} | Image Size: {imgsz}")
    print("============================================================")

    model = YOLO(base_model)

    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )

    metrics = model.val()

    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else os.path.join(project, name)
    best_weights = os.path.join(save_dir, "weights", "best.pt")

    top1_acc = getattr(metrics, "top1", 0.0)
    top5_acc = getattr(metrics, "top5", 0.0)

    summary = {
        "best_weights": best_weights if os.path.exists(best_weights) else None,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "save_dir": save_dir,
    }

    print("\n------------------------------------------------------------")
    print("  Classification Training Completed!")
    print(f"  Best Weights:  {summary['best_weights']}")
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-5 Accuracy: {top5_acc:.4f}")
    print("------------------------------------------------------------\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on Fruits-360 Classification Dataset")
    parser.add_argument("--data", type=str, default="datasets/fruits_360", help="Path to Fruits-360 dataset folder")
    parser.add_argument("--model", type=str, default="yolov8s-cls.pt", help="Base model (e.g. yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image resolution")
    parser.add_argument("--project", type=str, default="runs/freshai_classify", help="Project output folder")
    parser.add_argument("--name", type=str, default="fruits360_model", help="Experiment name")

    args = parser.parse_args()

    train_classifier(
        data_dir=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
