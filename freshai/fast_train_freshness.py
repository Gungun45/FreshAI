"""
FreshAI — High-Speed GPU ConvNeXt-Tiny Freshness Trainer
Optimized for NVIDIA GPUs (GTX 1650 / RTX) with:
1. Balanced class sampling (prevents class imbalance, speeds up epochs by 10x)
2. PyTorch AMP (Automatic Mixed Precision) for fast GPU computation
3. ImageNet pretrained ConvNeXt-Tiny backbone with warm-up fine-tuning
4. Auto-save to 'runs/freshness/best_freshness.pth'
"""

import os
import sys
import json
import time
import random
import shutil
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image as PILImage

FRESHNESS_STAGES = ["Very Fresh", "Fresh", "Ripe", "Overripe", "Deteriorating", "Spoiled"]
STAGE_FOLDERS = ["very_fresh", "fresh", "ripe", "overripe", "deteriorating", "spoiled"]
RIPENESS_STAGES = ["Unripe", "Nearly Ripe", "Ripe", "Overripe"]
RIPENESS_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}
CONVNEXT_DIM = 768


class BalancedFreshnessDataset(Dataset):
    """
    Dataset that loads a balanced sample from each freshness class.
    """
    IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, root_dir: str, max_per_class: int = 1200, transform=None, seed: int = 42):
        self.samples = []
        self.transform = transform
        random.seed(seed)

        for f_idx, folder_name in enumerate(STAGE_FOLDERS):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            all_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in self.IMG_EXT
            ]
            
            # Subsample if more than max_per_class
            if len(all_files) > max_per_class:
                selected_files = random.sample(all_files, max_per_class)
            else:
                selected_files = all_files
            
            r_idx = RIPENESS_MAP[f_idx]
            for p in selected_files:
                self.samples.append((p, f_idx, r_idx))
        
        # Shuffle across classes
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, f_label, r_label = self.samples[idx]
        try:
            img = PILImage.open(path).convert("RGB")
        except Exception:
            img = PILImage.new("RGB", (224, 224), (128, 128, 128))
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(f_label, dtype=torch.long), torch.tensor(r_label, dtype=torch.long)


class ConvNeXtMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        backbone = models.convnext_tiny(weights=weights)
        self.feature_extractor = nn.Sequential(
            backbone.features, backbone.avgpool, nn.Flatten(1)
        )
        self.freshness_head = nn.Sequential(
            nn.LayerNorm(CONVNEXT_DIM),
            nn.Linear(CONVNEXT_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(FRESHNESS_STAGES))
        )
        self.ripeness_head = nn.Sequential(
            nn.LayerNorm(CONVNEXT_DIM),
            nn.Linear(CONVNEXT_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(RIPENESS_STAGES))
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.freshness_head(feats), self.ripeness_head(feats), feats


def run_fast_training(
    data_dir: str = "datasets/freshness_dataset",
    epochs: int = 10,
    batch_size: int = 32,
    max_train_per_class: int = 1000,
    max_val_per_class: int = 300,
    lr: float = 3e-4,
    save_path: str = "runs/freshness/best_freshness.pth"
):
    print("=" * 65)
    print("  🌿 FreshAI — High-Speed ConvNeXt-Tiny Freshness Trainer")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [Hardware] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training dir not found: {train_dir}")

    # Transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print("  [Data] Loading balanced dataset splits...")
    train_ds = BalancedFreshnessDataset(train_dir, max_per_class=max_train_per_class, transform=train_transform)
    val_ds = BalancedFreshnessDataset(val_dir, max_per_class=max_val_per_class, transform=val_transform)

    print(f"  [Data] Balanced Training Samples:   {len(train_ds)}")
    print(f"  [Data] Balanced Validation Samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    print("  [Model] Initializing ConvNeXt-Tiny multi-task model...")
    model = ConvNeXtMultiTaskModel().to(device)

    # Freeze backbone for initial 2 epochs
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    f_criterion = nn.CrossEntropyLoss()
    r_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n  🚀 Starting Fast Training ({epochs} Epochs, Batch Size: {batch_size})")
    print("-" * 65)

    start_time = time.time()

    for epoch in range(epochs):
        ep_start = time.time()

        # Unfreeze backbone after epoch 2 for fine-tuning
        if epoch == 2:
            print("  ✦ Unfreezing ConvNeXt-Tiny backbone for end-to-end fine-tuning...")
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 2)

        # Training Phase
        model.train()
        train_loss = 0.0
        train_f_correct = 0
        train_total = 0

        for imgs, f_labels, r_labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            f_labels = f_labels.to(device, non_blocking=True)
            r_labels = r_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    f_logits, r_logits, _ = model(imgs)
                    loss = f_criterion(f_logits, f_labels) + 0.5 * r_criterion(r_logits, r_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                f_logits, r_logits, _ = model(imgs)
                loss = f_criterion(f_logits, f_labels) + 0.5 * r_criterion(r_logits, r_labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_f_correct += (f_logits.argmax(1) == f_labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()
        train_acc = train_f_correct / train_total if train_total > 0 else 0

        # Validation Phase
        model.eval()
        val_f_correct = 0
        val_r_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, f_labels, r_labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                f_labels = f_labels.to(device, non_blocking=True)
                r_labels = r_labels.to(device, non_blocking=True)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        f_logits, r_logits, _ = model(imgs)
                else:
                    f_logits, r_logits, _ = model(imgs)

                val_f_correct += (f_logits.argmax(1) == f_labels).sum().item()
                val_r_correct += (r_logits.argmax(1) == r_labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_f_correct / val_total if val_total > 0 else 0
        val_r_acc = val_r_correct / val_total if val_total > 0 else 0
        ep_duration = time.time() - ep_start

        print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] ({ep_duration:4.1f}s) | "
              f"Train Acc: {train_acc*100:5.1f}% | "
              f"Val Freshness: {val_acc*100:5.1f}% | "
              f"Val Ripeness: {val_r_acc*100:5.1f}%", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_freshness_acc": round(val_acc, 4),
                "val_ripeness_acc": round(val_r_acc, 4),
                "freshness_classes": FRESHNESS_STAGES,
                "ripeness_classes": RIPENESS_STAGES,
            }, save_path)
            print(f"     ⭐ Best Model Saved → {save_path} ({val_acc*100:.1f}%)")

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"  🎉 Training Completed in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print(f"  🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  📁 Model Saved to: {os.path.abspath(save_path)}")
    print("=" * 65)
    return save_path, best_val_acc


if __name__ == "__main__":
    run_fast_training()
