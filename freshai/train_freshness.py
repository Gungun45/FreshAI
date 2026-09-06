"""
FreshAI - Step 2: ConvNeXt-Tiny Freshness Model Training Script
Usage:
    python -m freshai.train_freshness --data datasets/freshness_dataset --epochs 30

Dataset structure expected:
    datasets/freshness_dataset/
        train/
            very_fresh/   (or 0_very_fresh/)
            fresh/
            ripe/
            overripe/
            deteriorating/
            spoiled/
        val/
            (same structure)

Each class folder contains produce images (jpg/png).
Freshness and Ripeness labels can be inferred from folder name OR provided via a JSON sidecar.
"""

import os
import argparse
import json
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from PIL import Image as PILImage
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from freshai.freshness_detector import (
    ConvNeXtMultiTaskHead,
    FRESHNESS_STAGES,
    RIPENESS_STAGES,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

STAGE_NAME_MAP = {
    "very_fresh": 0, "veryfresh": 0, "0_very_fresh": 0,
    "fresh": 1, "1_fresh": 1,
    "ripe": 2, "2_ripe": 2,
    "overripe": 3, "over_ripe": 3, "3_overripe": 3,
    "deteriorating": 4, "4_deteriorating": 4,
    "spoiled": 5, "5_spoiled": 5,
}

RIPENESS_FROM_FRESHNESS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}


if TORCH_AVAILABLE:
    class FreshnessDataset(Dataset):
        """
        Multi-task freshness dataset.
        Folder name → freshness class index (0-5).
        Ripeness label is derived from freshness index.
        """
        IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        def __init__(self, root_dir: str, transform=None):
            self.samples = []   # [(path, freshness_idx, ripeness_idx)]
            self.transform = transform

            for cls_folder in sorted(os.listdir(root_dir)):
                cls_path = os.path.join(root_dir, cls_folder)
                if not os.path.isdir(cls_path):
                    continue
                key = cls_folder.lower().strip()
                if key not in STAGE_NAME_MAP:
                    print(f"  [Dataset] Unknown class folder '{cls_folder}' — skipping.")
                    continue
                f_idx = STAGE_NAME_MAP[key]
                r_idx = RIPENESS_FROM_FRESHNESS[f_idx]
                for fname in os.listdir(cls_path):
                    if os.path.splitext(fname)[1].lower() in self.IMG_EXTENSIONS:
                        self.samples.append((os.path.join(cls_path, fname), f_idx, r_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, f_label, r_label = self.samples[idx]
            img = PILImage.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(f_label, dtype=torch.long), torch.tensor(r_label, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_freshness_model(
    data_dir: str = "datasets/freshness_dataset",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    imgsz: int = 224,
    project: str = "runs/freshness",
    name: str = "convnext_freshness",
    freeze_backbone_epochs: int = 2,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Train ConvNeXt-Tiny multi-task freshness model with AMP acceleration and resume capability.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for training. Install with: pip install torch torchvision"
        )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)
    best_weights_path = os.path.join(save_dir, "best_freshness.pth")
    root_best_path = os.path.join(project, "best_freshness.pth")

    # ── Transforms ──────────────────────────────────────────────────────────
    train_transform = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = FreshnessDataset(train_dir, transform=train_transform)
    val_dataset = FreshnessDataset(val_dir, transform=val_transform)

    print(f"  Training samples:   {len(train_dataset)} ({len(train_dataset)//batch_size} batches)")
    print(f"  Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise RuntimeError("No training images found. Check your dataset folder structure.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = ConvNeXtMultiTaskHead(
        num_freshness_classes=len(FRESHNESS_STAGES),
        num_ripeness_classes=len(RIPENESS_STAGES),
        pretrained=True,
    ).to(device)

    start_epoch = 0
    best_val_acc = 0.0

    # Resume if checkpoint exists
    if resume and os.path.exists(best_weights_path):
        try:
            ckpt = torch.load(best_weights_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            start_epoch = ckpt.get("epoch", 1)
            best_val_acc = ckpt.get("val_freshness_acc", 0.0) or 0.0
            print(f"  🔄 Resumed from checkpoint (Epoch {start_epoch}, previous val_acc={best_val_acc*100:.2f}%)")
        except Exception as e:
            print(f"  Could not resume from checkpoint ({e}), starting fresh.")

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    freshness_criterion = nn.CrossEntropyLoss()
    ripeness_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - start_epoch))
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    history = []

    print(f"\n{'='*60}")
    print(f"  FreshAI — ConvNeXt-Tiny Freshness Training (AMP GPU Accelerated)")
    print(f"  Epochs: {start_epoch+1} to {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_f_correct = train_r_correct = train_total = 0

        for batch_idx, (imgs, f_labels, r_labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            f_labels = f_labels.to(device, non_blocking=True)
            r_labels = r_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    f_logits, r_logits, _ = model(imgs)
                    loss = freshness_criterion(f_logits, f_labels) + 0.5 * ripeness_criterion(r_logits, r_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                f_logits, r_logits, _ = model(imgs)
                loss = freshness_criterion(f_logits, f_labels) + 0.5 * ripeness_criterion(r_logits, r_labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_f_correct += (f_logits.argmax(1) == f_labels).sum().item()
            train_r_correct += (r_logits.argmax(1) == r_labels).sum().item()
            train_total += imgs.size(0)

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"    Batch [{batch_idx+1:4d}/{len(train_loader):4d}] Loss: {loss.item():.4f} "
                      f"Acc: {(train_f_correct/train_total)*100:5.1f}%", flush=True)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_f_correct = val_r_correct = val_total = 0

        with torch.no_grad():
            for imgs, f_labels, r_labels in val_loader:
                imgs, f_labels, r_labels = imgs.to(device, non_blocking=True), f_labels.to(device, non_blocking=True), r_labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        f_logits, r_logits, _ = model(imgs)
                else:
                    f_logits, r_logits, _ = model(imgs)
                    
                val_f_correct += (f_logits.argmax(1) == f_labels).sum().item()
                val_r_correct += (r_logits.argmax(1) == r_labels).sum().item()
                val_total += imgs.size(0)

        train_f_acc = train_f_correct / train_total
        val_f_acc = val_f_correct / val_total if val_total > 0 else 0.0
        val_r_acc = val_r_correct / val_total if val_total > 0 else 0.0

        history.append({
            "epoch": epoch + 1,
            "train_freshness_acc": round(train_f_acc, 4),
            "val_freshness_acc": round(val_f_acc, 4),
            "val_ripeness_acc": round(val_r_acc, 4),
        })

        print(f"\n  ✦ Epoch [{epoch+1:3d}/{epochs}]  "
              f"Train F-Acc: {train_f_acc*100:5.2f}%  "
              f"Val F-Acc: {val_f_acc*100:5.2f}%  "
              f"Val R-Acc: {val_r_acc*100:5.2f}%\n")

        # Save best model
        if val_f_acc >= best_val_acc or not os.path.exists(best_weights_path):
            best_val_acc = val_f_acc
            save_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_freshness_acc": val_f_acc,
                "val_ripeness_acc": val_r_acc,
                "freshness_classes": FRESHNESS_STAGES,
                "ripeness_classes": RIPENESS_STAGES,
            }
            torch.save(save_payload, best_weights_path)
            torch.save(save_payload, root_best_path)
            print(f"  ⭐ Best model saved → {best_weights_path} & {root_best_path} ({val_f_acc*100:.2f}%)")

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_weights": best_weights_path if os.path.exists(best_weights_path) else None,
        "best_val_freshness_acc": round(best_val_acc, 4),
        "epochs": epochs,
        "save_dir": save_dir,
        "history_path": history_path,
    }

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Val Freshness Acc: {best_val_acc*100:.2f}%")
    print(f"  Best Weights: {best_weights_path}")
    print(f"{'='*60}\n")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Tiny Freshness Detection Model"
    )
    parser.add_argument("--data", type=str, default="datasets/freshness_dataset",
                        help="Path to freshness dataset (with train/ and val/ subdirs)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--project", type=str, default="runs/freshness")
    parser.add_argument("--name", type=str, default="convnext_freshness")
    parser.add_argument("--freeze-epochs", type=int, default=2,
                        help="Epochs to train only the heads before unfreezing backbone")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")
    args = parser.parse_args()

    train_freshness_model(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        freeze_backbone_epochs=args.freeze_epochs,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
