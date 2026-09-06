"""
FreshAI - Stage 4: Multimodal Freshness Model Training Script

Pipeline:
1. Ingests produce training images (e.g. from datasets/freshness_dataset)
2. Extracts:
   • ConvNeXt-Tiny deep features & stage probabilities
   • OpenCV color statistics (RGB, HSV, LAB, region percentages)
   • OpenCV GLCM texture metrics (Contrast, Homogeneity, Energy, Dissimilarity, Correlation)
   • YOLO Defect Detection & Segmentation metrics (Defect Area %, defect breakdown)
3. Fuses all signals into standardized ProduceFeatureVectors
4. Trains a PyTorch Fusion MLP classifier/regressor to accurately predict freshness stages
5. Saves model weights to runs/fusion/best_fusion_model.pt

Usage:
    python -m freshai.train_fusion_freshness --epochs 25 --data datasets/freshness_dataset
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from freshai.freshness_detector import ConvNeXtFreshnessDetector, FRESHNESS_STAGES
from freshai.defect_detector import FreshAIDefectDetector
from freshai.feature_extractor import ProduceFeatureExtractor
from freshai.fusion import assemble_feature_vector, ProduceFeatureVector
from freshai.fusion_predictor import FusionMLP


STAGE_MAP = {
    "very_fresh": 0, "veryfresh": 0, "0_very_fresh": 0,
    "fresh": 1, "1_fresh": 1,
    "ripe": 2, "2_ripe": 2,
    "overripe": 3, "over_ripe": 3, "3_overripe": 3,
    "deteriorating": 4, "4_deteriorating": 4,
    "spoiled": 5, "5_spoiled": 5,
}


def extract_multimodal_dataset(
    data_dir: str,
    max_samples_per_class: int = 40,
    convnext_detector: Optional[ConvNeXtFreshnessDetector] = None,
    defect_detector: Optional[FreshAIDefectDetector] = None,
    feature_extractor: Optional[ProduceFeatureExtractor] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract multimodal feature vectors (X) and freshness labels (y) from produce images.
    """
    convnext_detector = convnext_detector or ConvNeXtFreshnessDetector()
    defect_detector = defect_detector or FreshAIDefectDetector()
    feature_extractor = feature_extractor or ProduceFeatureExtractor()

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    filepaths: List[str] = []

    train_root = os.path.join(data_dir, "train") if os.path.exists(os.path.join(data_dir, "train")) else data_dir

    if not os.path.exists(train_root):
        print(f"[Dataset] Directory {train_root} does not exist. Generating synthetic training samples...")
        return generate_synthetic_fusion_dataset(num_samples=180)

    print(f"\n{'='*60}")
    print(f"  Extracting Multimodal Feature Vectors from: {train_root}")
    print(f"{'='*60}")

    for class_folder in sorted(os.listdir(train_root)):
        cls_path = os.path.join(train_root, class_folder)
        if not os.path.isdir(cls_path):
            continue
        key = class_folder.lower().strip()
        if key not in STAGE_MAP:
            continue
        target_stage_idx = STAGE_MAP[key]

        img_files = [
            f for f in os.listdir(cls_path)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        selected_files = img_files[:max_samples_per_class]
        print(f"  Processing '{class_folder}' ({len(selected_files)} images)...")

        for fname in selected_files:
            fpath = os.path.join(cls_path, fname)
            try:
                img_pil = Image.open(fpath).convert("RGB")
                img_np = np.array(img_pil)

                # 1. ConvNeXt Baseline
                fr = convnext_detector.predict(img_pil, produce_class=key)

                # 2. Defect Detection & Segmentation
                dr = defect_detector.predict(img_np, produce_class=key)

                # 3. OpenCV Color & Texture Extraction
                fe = feature_extractor.extract(img_np)

                # 4. Assemble Vector
                fv = assemble_feature_vector(fr, dr, fe, produce_class=key)
                dense_vec = fv.to_dense_vector(include_deep=False)

                features_list.append(dense_vec)
                labels_list.append(target_stage_idx)
                filepaths.append(fpath)
            except Exception as e:
                print(f"    Failed {fname}: {e}")

    if len(features_list) == 0:
        print("  No valid images found in dataset. Generating synthetic fusion dataset...")
        return generate_synthetic_fusion_dataset(num_samples=180)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)
    print(f"  Extracted {len(X)} feature vectors of dimension {X.shape[1]}.\n")
    return X, y, filepaths


def generate_synthetic_fusion_dataset(num_samples: int = 180) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate realistic synthetic multimodal feature vectors covering all 6 freshness stages
    with realistic defect area %, color degradation, and GLCM texture features.
    """
    print("  Synthesizing multimodal feature distributions across 6 stages...")
    X_list, y_list = [], []

    samples_per_stage = max(10, num_samples // 6)
    np.random.seed(42)

    for stage_idx in range(6):
        for _ in range(samples_per_stage):
            # ConvNeXt stage probability simulation
            base_prob = np.random.uniform(0.65, 0.95)
            f_probs = [0.03] * 6
            f_probs[stage_idx] = base_prob
            rem = (1.0 - base_prob) / 5.0
            for i in range(6):
                if i != stage_idx:
                    f_probs[i] = rem

            # Ripeness index correlation
            r_idx = min(3, max(0, stage_idx // 2))
            r_prob = np.random.uniform(0.70, 0.95)
            r_probs = [0.05] * 4
            r_probs[r_idx] = r_prob

            # Defect Area % correlates with deterioration
            if stage_idx == 0:    # Very Fresh
                defect_pct = np.random.uniform(0.0, 1.0)
                rot_pct, mold_pct, spot_pct = 0.0, 0.0, 0.0
                homogeneity = np.random.uniform(0.75, 0.90)
                contrast = np.random.uniform(0.15, 0.40)
            elif stage_idx == 1:  # Fresh
                defect_pct = np.random.uniform(0.5, 3.5)
                rot_pct, mold_pct = 0.0, 0.0
                spot_pct = np.random.uniform(0.0, 1.0)
                homogeneity = np.random.uniform(0.65, 0.82)
                contrast = np.random.uniform(0.25, 0.55)
            elif stage_idx == 2:  # Ripe
                defect_pct = np.random.uniform(2.0, 7.0)
                rot_pct, mold_pct = 0.0, 0.0
                spot_pct = np.random.uniform(0.5, 2.5)
                homogeneity = np.random.uniform(0.55, 0.75)
                contrast = np.random.uniform(0.35, 0.70)
            elif stage_idx == 3:  # Overripe
                defect_pct = np.random.uniform(6.0, 14.0)
                rot_pct = np.random.uniform(0.0, 2.0)
                mold_pct = 0.0
                spot_pct = np.random.uniform(2.0, 6.0)
                homogeneity = np.random.uniform(0.40, 0.65)
                contrast = np.random.uniform(0.60, 1.20)
            elif stage_idx == 4:  # Deteriorating
                defect_pct = np.random.uniform(12.0, 28.0)
                rot_pct = np.random.uniform(2.0, 8.0)
                mold_pct = np.random.uniform(0.5, 4.0)
                spot_pct = np.random.uniform(4.0, 10.0)
                homogeneity = np.random.uniform(0.25, 0.50)
                contrast = np.random.uniform(1.00, 2.00)
            else:                 # Spoiled
                defect_pct = np.random.uniform(25.0, 60.0)
                rot_pct = np.random.uniform(6.0, 25.0)
                mold_pct = np.random.uniform(4.0, 20.0)
                spot_pct = np.random.uniform(8.0, 20.0)
                homogeneity = np.random.uniform(0.15, 0.35)
                contrast = np.random.uniform(1.50, 3.50)

            # Color shifts (browning, loss of green/saturation)
            avg_hue = float(np.clip(12.4 - stage_idx * 1.5 + np.random.normal(0, 1), 0, 180))
            saturation = float(np.clip(82.6 - stage_idx * 7.0 + np.random.normal(0, 3), 10, 100))
            brightness = float(np.clip(75.0 - stage_idx * 6.0 + np.random.normal(0, 4), 10, 100))
            brown_pct = float(np.clip(stage_idx * 4.5 + np.random.normal(0, 1), 0, 80))
            dark_pct = float(np.clip(stage_idx * 3.5 + np.random.normal(0, 1), 0, 80))
            mold_pct_reg = mold_pct

            fv = ProduceFeatureVector(
                freshness_prob=base_prob,
                freshness_stage_idx=stage_idx,
                freshness_all_probs=f_probs,
                ripeness_prob=r_prob,
                ripeness_stage_idx=r_idx,
                ripeness_all_probs=r_probs,
                defect_area_pct=defect_pct,
                defect_count=int(defect_pct / 2) + 1,
                defect_severity_score=min(1.0, defect_pct / 40.0),
                bruise_area_pct=defect_pct * 0.3,
                rot_area_pct=rot_pct,
                mold_area_pct=mold_pct,
                spot_area_pct=spot_pct,
                avg_hue=avg_hue,
                avg_saturation=saturation,
                avg_brightness=brightness,
                mean_r=180.0 - stage_idx * 10,
                mean_g=140.0 - stage_idx * 15,
                mean_b=80.0 - stage_idx * 5,
                mean_l=70.0 - stage_idx * 8,
                mean_a=15.0,
                mean_b_lab=30.0,
                chroma=33.0,
                green_region_pct=max(0.0, 50.0 - stage_idx * 12),
                yellow_region_pct=max(0.0, 40.0 - stage_idx * 5),
                red_region_pct=15.0,
                brown_region_pct=brown_pct,
                dark_decay_pct=dark_pct,
                pale_mold_pct=mold_pct_reg,
                texture_contrast=contrast,
                texture_homogeneity=homogeneity,
                texture_dissimilarity=contrast * 0.7,
                texture_energy=homogeneity * 0.5,
                texture_correlation=0.85 - stage_idx * 0.08,
                texture_entropy=2.5 + stage_idx * 0.4,
                surface_roughness=100.0 + stage_idx * 120.0,
                edge_density=0.05 + stage_idx * 0.03,
            )

            X_list.append(fv.to_dense_vector(include_deep=False))
            y_list.append(stage_idx)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), []


def train_fusion_model(
    data_dir: str = "datasets/freshness_dataset",
    output_dir: str = "runs/fusion",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001,
    max_samples: int = 40,
) -> Dict[str, Any]:
    """
    Train the Multimodal Freshness Fusion Model.
    """
    if not TORCH_AVAILABLE:
        print("[Error] PyTorch is required for training the fusion model.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract or synthesize multimodal training features
    X, y, _ = extract_multimodal_dataset(data_dir=data_dir, max_samples_per_class=max_samples)

    # Shuffle and train/val split (80/20)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Standardize features
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    train_dataset = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    in_features = X.shape[1]
    model = FusionMLP(in_features=in_features, num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights_path = os.path.join(output_dir, "best_fusion_model.pt")

    print(f"\n{'='*60}")
    print(f"  Training Multimodal Fusion MLP (in_features={in_features}, epochs={epochs})")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        val_acc = val_correct / max(1, val_total)

        if epoch % 5 == 0 or epoch == epochs or val_acc > best_val_acc:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Loss: {total_loss/max(1,total):.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "in_features": in_features,
                "feature_names": ProduceFeatureVector.feature_names(include_deep=False),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "val_acc": val_acc,
                "epochs": epoch,
            }, best_weights_path)

    print(f"\n  Training Complete! Best Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"  Model saved to: {best_weights_path}")

    summary = {
        "best_val_accuracy": round(best_val_acc, 4),
        "in_features": in_features,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "weights_path": best_weights_path,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FreshAI Multimodal Freshness Fusion Model")
    parser.add_argument("--data", type=str, default="datasets/freshness_dataset", help="Path to produce dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=40, help="Max images per class to load")
    args = parser.parse_args()

    train_fusion_model(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
    )
