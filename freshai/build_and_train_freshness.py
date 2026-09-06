"""
FreshAI — Auto Freshness Dataset Builder + ConvNeXt-Tiny Trainer
Builds a freshness training dataset from Fruits-360 and trains the model.
Runs fully automatically — no manual commands needed.
"""

import os
import shutil
import json
import math
import random
import sys
from pathlib import Path

# ─── Freshness stage mapping from Fruits-360 class names ───────────────────
# We map every Fruits-360 class to a freshness stage based on:
#   - Ripe variants   → Ripe
#   - Standard fresh  → Fresh / Very Fresh
#   - Overripe / aged → Overripe
#   - Processed (dried/wax) → Deteriorating
FRESHNESS_MAP = {
    # ── Very Fresh (bright, perfect, peak-season produce) ───────────────────
    "Apple Braeburn": "very_fresh",
    "Apple Crimson Snow": "very_fresh",
    "Apple Golden 1": "very_fresh",
    "Apple Pink Lady": "very_fresh",
    "Apple Red 1": "very_fresh",
    "Apple Red Delicious": "very_fresh",
    "Cherry 1": "very_fresh",
    "Cherry Rainier": "very_fresh",
    "Cherry Wax Red": "very_fresh",
    "Cherry Wax Yellow": "very_fresh",
    "Strawberry": "very_fresh",
    "Blueberry": "very_fresh",
    "Raspberry": "very_fresh",
    "Grape Blue": "very_fresh",
    "Grape Pink": "very_fresh",
    "Grape White": "very_fresh",
    "Kiwi": "very_fresh",
    "Lemon": "very_fresh",
    "Orange": "very_fresh",
    "Pineapple": "very_fresh",
    "Tomato 1": "very_fresh",
    "Pepper Green": "very_fresh",
    "Pepper Red": "very_fresh",
    "Pepper Yellow": "very_fresh",
    "Cauliflower": "very_fresh",
    "Eggplant": "very_fresh",
    "Beetroot": "very_fresh",

    # ── Fresh (good quality, standard shelf produce) ─────────────────────────
    "Apple Golden 2": "fresh",
    "Apple Golden 3": "fresh",
    "Apple Granny Smith": "fresh",
    "Apple Red 2": "fresh",
    "Apple Red 3": "fresh",
    "Apple Red Yellow 1": "fresh",
    "Apple Red Yellow 2": "fresh",
    "Apricot": "fresh",
    "Banana": "fresh",
    "Banana Lady Finger": "fresh",
    "Cantaloupe 1": "fresh",
    "Cantaloupe 2": "fresh",
    "Cherry 2": "fresh",
    "Cherry Wax Black": "fresh",
    "Clementine": "fresh",
    "Grapefruit Pink": "fresh",
    "Grapefruit White": "fresh",
    "Grape White 2": "fresh",
    "Grape White 3": "fresh",
    "Grape White 4": "fresh",
    "Guava": "fresh",
    "Lemon Meyer": "fresh",
    "Limes": "fresh",
    "Mandarine": "fresh",
    "Mango": "fresh",
    "Melon Piel de Sapo": "fresh",
    "Nectarine": "fresh",
    "Onion Red": "fresh",
    "Onion White": "fresh",
    "Papaya": "fresh",
    "Peach": "fresh",
    "Pear": "fresh",
    "Pear Abate": "fresh",
    "Pear Forelle": "fresh",
    "Pear Williams": "fresh",
    "Plum": "fresh",
    "Pomegranate": "fresh",
    "Potato Red": "fresh",
    "Potato White": "fresh",
    "Strawberry Wedge": "fresh",
    "Tomato 2": "fresh",
    "Pitahaya Red": "fresh",
    "Pear Kaiser": "fresh",
    "Peach 2": "fresh",

    # ── Ripe (at peak ripeness, perfect to eat now) ──────────────────────────
    "Avocado ripe": "ripe",
    "Mango Red": "ripe",
    "Banana Red": "ripe",
    "Plum 2": "ripe",
    "Plum 3": "ripe",
    "Peach Flat": "ripe",
    "Nectarine Flat": "ripe",
    "Kaki": "ripe",        # Persimmon — fully ripe when orange
    "Physalis": "ripe",
    "Passion Fruit": "ripe",
    "Granadilla": "ripe",
    "Maracuja": "ripe",
    "Lychee": "ripe",
    "Rambutan": "ripe",
    "Mulberry": "ripe",
    "Pomelo Sweetie": "ripe",
    "Tangelo": "ripe",
    "Mangostan": "ripe",
    "Kumquats": "ripe",
    "Pear Monster": "ripe",
    "Pear Red": "ripe",
    "Pepino": "ripe",
    "Carambula": "ripe",
    "Tamarillo": "ripe",
    "Redcurrant": "ripe",
    "Huckleberry": "ripe",
    "Salak": "ripe",
    "Cactus fruit": "ripe",

    # ── Overripe (past peak, should be used soon) ────────────────────────────
    "Avocado": "overripe",   # Unripe avocado looks dull/hard
    "Pineapple Mini": "overripe",
    "Onion Red Peeled": "overripe",
    "Potato Red Washed": "overripe",
    "Potato Sweet": "overripe",
    "Quince": "overripe",
    "Kohlrabi": "overripe",

    # ── Deteriorating (aged, dried, heavily processed) ───────────────────────
    "Dates": "deteriorating",
    "Chestnut": "deteriorating",
    "Hazelnut": "deteriorating",
    "Nut Forest": "deteriorating",
    "Nut Pecan": "deteriorating",
    "Physalis with Husk": "deteriorating",
    "Ginger Root": "deteriorating",
    "Cocos": "deteriorating",

    # ── Spoiled — not really represented cleanly in Fruits-360 ───────────────
    # We'll generate spoiled examples synthetically using color-jitter
}

FRESHNESS_STAGES = ["very_fresh", "fresh", "ripe", "overripe", "deteriorating", "spoiled"]


def build_freshness_dataset(
    source_dir: str = "datasets/fruits_360",
    output_dir: str = "datasets/freshness_dataset",
    val_ratio: float = 0.2,
    spoiled_augment_count: int = 600,
    seed: int = 42,
) -> str:
    """
    Build a freshness classification dataset from Fruits-360 by remapping
    class names to freshness stages via FRESHNESS_MAP.
    Also synthesizes 'spoiled' examples by applying darkening + desaturation augmentation.
    """
    random.seed(seed)

    print("\n" + "="*60)
    print("  FreshAI — Building Freshness Dataset from Fruits-360")
    print("="*60)

    # Create output directories
    for split in ["train", "val"]:
        for stage in FRESHNESS_STAGES:
            os.makedirs(os.path.join(output_dir, split, stage), exist_ok=True)

    stage_counts = {s: {"train": 0, "val": 0} for s in FRESHNESS_STAGES}

    # ── Copy images per mapping ──────────────────────────────────────────────
    for fruits_class, stage in FRESHNESS_MAP.items():
        src_train = os.path.join(source_dir, "train", fruits_class)
        src_val = os.path.join(source_dir, "val", fruits_class)

        for src_dir, split in [(src_train, "train"), (src_val, "val")]:
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(output_dir, split, stage)
            imgs = [f for f in os.listdir(src_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_fname in imgs:
                src_path = os.path.join(src_dir, img_fname)
                # Prefix filename with class to avoid conflicts
                safe_class = fruits_class.replace(" ", "_")
                dst_fname = f"{safe_class}_{img_fname}"
                dst_path = os.path.join(dst_dir, dst_fname)
                shutil.copy2(src_path, dst_path)
                stage_counts[stage][split] += 1

    # ── Synthesize 'spoiled' examples from overripe + deteriorating ─────────
    print("\n  Generating synthetic 'spoiled' examples via PIL augmentation...")
    try:
        from PIL import Image, ImageEnhance, ImageFilter
        import numpy as np

        spoiled_src_dirs = []
        for stage in ["deteriorating", "overripe"]:
            spoiled_src_dirs.append(os.path.join(output_dir, "train", stage))
            spoiled_src_dirs.append(os.path.join(output_dir, "val", stage))

        src_images = []
        for d in spoiled_src_dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_images.append(os.path.join(d, f))

        random.shuffle(src_images)

        for i in range(min(spoiled_augment_count, len(src_images))):
            split = "train" if i < int(spoiled_augment_count * (1 - val_ratio)) else "val"
            src_path = src_images[i % len(src_images)]
            dst_dir = os.path.join(output_dir, split, "spoiled")
            dst_fname = f"synth_spoiled_{i:04d}.jpg"
            dst_path = os.path.join(dst_dir, dst_fname)

            img = Image.open(src_path).convert("RGB")
            # Apply spoilage effects
            img = ImageEnhance.Color(img).enhance(random.uniform(0.05, 0.25))    # Desaturate
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.3, 0.55)) # Darken
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.4, 0.7))    # Low contrast
            if random.random() > 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            img.save(dst_path, quality=85)
            stage_counts["spoiled"][split] += 1

    except Exception as e:
        print(f"  Warning: Could not generate synthetic spoiled images: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n  Dataset Summary:")
    total_train = total_val = 0
    for stage in FRESHNESS_STAGES:
        tr = stage_counts[stage]["train"]
        vl = stage_counts[stage]["val"]
        total_train += tr
        total_val += vl
        print(f"    {stage:<20}: {tr:>5} train  |  {vl:>4} val")
    print(f"    {'TOTAL':<20}: {total_train:>5} train  |  {total_val:>4} val")
    print(f"\n  Output directory: {os.path.abspath(output_dir)}")

    return output_dir


def train_model(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-4,
    imgsz: int = 224,
    project: str = "runs/freshness",
    name: str = "convnext_freshness",
    freeze_backbone_epochs: int = 3,
):
    """Train ConvNeXt-Tiny freshness model on the prepared dataset."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    import torchvision.models as models
    from PIL import Image as PILImage

    STAGE_NAME_MAP = {s: i for i, s in enumerate(FRESHNESS_STAGES)}
    RIPENESS_FROM_FRESHNESS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}
    IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    CONVNEXT_DIM = 768

    class FreshnessDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.samples = []
            self.transform = transform
            for cls_folder in sorted(os.listdir(root_dir)):
                cls_path = os.path.join(root_dir, cls_folder)
                if not os.path.isdir(cls_path):
                    continue
                key = cls_folder.lower().strip()
                if key not in STAGE_NAME_MAP:
                    continue
                f_idx = STAGE_NAME_MAP[key]
                r_idx = RIPENESS_FROM_FRESHNESS[f_idx]
                for fname in os.listdir(cls_path):
                    if os.path.splitext(fname)[1].lower() in IMG_EXT:
                        self.samples.append((os.path.join(cls_path, fname), f_idx, r_idx))

        def __len__(self): return len(self.samples)

        def __getitem__(self, idx):
            path, f_label, r_label = self.samples[idx]
            try:
                img = PILImage.open(path).convert("RGB")
            except Exception:
                img = PILImage.new("RGB", (224, 224), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(f_label, dtype=torch.long), torch.tensor(r_label, dtype=torch.long)

    class ConvNeXtMultiTask(nn.Module):
        def __init__(self):
            super().__init__()
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            backbone = models.convnext_tiny(weights=weights)
            self.feature_extractor = nn.Sequential(
                backbone.features, backbone.avgpool, nn.Flatten(1)
            )
            self.freshness_head = nn.Sequential(
                nn.LayerNorm(CONVNEXT_DIM), nn.Linear(CONVNEXT_DIM, 256),
                nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(FRESHNESS_STAGES))
            )
            self.ripeness_head = nn.Sequential(
                nn.LayerNorm(CONVNEXT_DIM), nn.Linear(CONVNEXT_DIM, 128),
                nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 4)
            )

        def forward(self, x):
            feats = self.feature_extractor(x)
            return self.freshness_head(feats), self.ripeness_head(feats), feats

    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)

    train_transform = T.Compose([
        T.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FreshnessDataset(os.path.join(data_dir, "train"), train_transform)
    val_ds   = FreshnessDataset(os.path.join(data_dir, "val"),   val_transform)
    print(f"\n  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError("No training images found. Dataset build may have failed.")

    num_workers = 0  # Windows-safe default
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = ConvNeXtMultiTask().to(device)

    # Freeze backbone initially
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    f_criterion = nn.CrossEntropyLoss()
    r_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = os.path.join(save_dir, "best_freshness.pth")
    history = []

    print(f"\n{'='*60}")
    print(f"  Training ConvNeXt-Tiny Freshness Model")
    print(f"  Epochs: {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Save: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        if epoch == freeze_backbone_epochs:
            print(f"  [Epoch {epoch+1}] Unfreezing backbone — full fine-tune begins.")
            for param in model.feature_extractor.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - freeze_backbone_epochs)
            )

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        for batch_idx, (imgs, f_labels, r_labels) in enumerate(train_loader):
            imgs, f_labels, r_labels = imgs.to(device), f_labels.to(device), r_labels.to(device)
            optimizer.zero_grad()
            f_logits, r_logits, _ = model(imgs)
            loss = f_criterion(f_logits, f_labels) + 0.5 * r_criterion(r_logits, r_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_correct += (f_logits.argmax(1) == f_labels).sum().item()
            train_total += imgs.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)}  loss={loss.item():.4f}", flush=True)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, f_labels, r_labels in val_loader:
                imgs, f_labels, r_labels = imgs.to(device), f_labels.to(device), r_labels.to(device)
                f_logits, _, _ = model(imgs)
                val_correct += (f_logits.argmax(1) == f_labels).sum().item()
                val_total += imgs.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc   = val_correct   / val_total   if val_total   > 0 else 0
        history.append({"epoch": epoch+1, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})

        print(f"\n  ✦ Epoch [{epoch+1:2d}/{epochs}]  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_freshness_acc": val_acc,
                "freshness_classes": FRESHNESS_STAGES,
                "ripeness_classes": ["Unripe", "Nearly Ripe", "Ripe", "Overripe"],
            }, best_weights_path)
            print(f"  ✅ Saved best model → {best_weights_path}  (val_acc={val_acc:.4f})")

    # Also copy best weights to the expected location for auto-loading
    fallback_path = "runs/freshness/best_freshness.pth"
    os.makedirs("runs/freshness", exist_ok=True)
    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, fallback_path)
        print(f"\n  ✅ Copied best model → {fallback_path}  (auto-loaded by FreshnessDetector)")

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f} ({round(best_val_acc*100,1)}%)")
    print(f"  Weights: {fallback_path}")
    print(f"{'='*60}\n")
    return best_weights_path, best_val_acc


# ─── MAIN: Build data + Train ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  default="datasets/fruits_360",     help="Source Fruits-360 directory")
    parser.add_argument("--output",  default="datasets/freshness_dataset", help="Output freshness dataset directory")
    parser.add_argument("--epochs",  type=int, default=20,              help="Training epochs")
    parser.add_argument("--batch",   type=int, default=32,              help="Batch size")
    parser.add_argument("--lr",      type=float, default=1e-4,          help="Learning rate")
    parser.add_argument("--rebuild", action="store_true",               help="Force rebuild dataset even if exists")
    args = parser.parse_args()

    # Step 1: Build dataset
    dataset_exists = (
        os.path.isdir(os.path.join(args.output, "train")) and
        any(os.listdir(os.path.join(args.output, "train"))) and
        not args.rebuild
    )
    if dataset_exists:
        print(f"\n  Dataset already exists at {args.output} — skipping rebuild.")
        print(f"  (Use --rebuild to force regeneration)")
    else:
        data_dir = build_freshness_dataset(
            source_dir=args.source,
            output_dir=args.output,
        )

    # Step 2: Train model
    weights_path, best_acc = train_model(
        data_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )

    print(f"\n🎉 FreshAI Freshness Model is ready!")
    print(f"   Weights: {weights_path}")
    print(f"   Best Val Accuracy: {round(best_acc*100,1)}%")
    print(f"\n   The model will auto-load on next app/server start.")
    print(f"   No additional commands needed — freshness is shown for every detected object.")
