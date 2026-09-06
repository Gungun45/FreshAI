"""
FreshAI - Resume Multi-Scale Produce Training Pipeline
Resumes and continues training from the latest checkpoint on GPU/CPU.
"""

import os
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CKPT = PROJECT_ROOT / "runs" / "detect" / "runs" / "freshai_detect" / "produce_detector-4" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "datasets" / "freshai_produce_multiscale" / "data_fixed.yaml"


def find_latest_weights() -> Path:
    candidates = list((PROJECT_ROOT / "runs" / "detect").rglob("best.pt"))
    if not candidates:
        return DEFAULT_CKPT
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def continue_training(
    ckpt_path: Path = None,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = None,
):
    ckpt = ckpt_path or find_latest_weights()
    if not ckpt.exists():
        print(f"[Error] Checkpoint not found at: {ckpt}")
        return

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("  FreshAI - Continuing Produce Object Detection Training")
    print(f"  Base Weights: {ckpt}")
    print(f"  Dataset YAML: {DATA_YAML}")
    print(f"  Epochs:       {epochs} | Batch: {batch_size} | ImgSz: {imgsz}")
    print(f"  Hardware:     {'NVIDIA CUDA GPU (0)' if device == '0' else 'CPU'}")
    print("=" * 65)

    model = YOLO(str(ckpt))
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0,  # Safe for Windows PyTorch multiprocessing
        project=str(PROJECT_ROOT / "runs" / "detect" / "runs" / "freshai_detect"),
        name="produce_detector-5",
        exist_ok=True,
        verbose=True,
    )
    print("\n  Produce Detection Training Completed!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Total target epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    args = parser.parse_args()

    continue_training(epochs=args.epochs, batch_size=args.batch, imgsz=args.imgsz)
