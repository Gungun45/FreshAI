"""
FreshAI - Step 2: Freshness Detection
Model: ConvNeXt-Tiny (Multi-Task Learning)
Outputs: Freshness Stage + Ripeness Stage + Visual Features
Fallback: Rule-based color heuristics (works Day 1, no trained model needed)
"""

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

# ─── Optional deep-learning imports ───────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

FRESHNESS_STAGES = [
    "Very Fresh",
    "Fresh",
    "Ripe",
    "Overripe",
    "Deteriorating",
    "Spoiled",
]

RIPENESS_STAGES = [
    "Unripe",
    "Nearly Ripe",
    "Ripe",
    "Overripe",
]

# Hex badge colors for each freshness stage (index-matched to FRESHNESS_STAGES)
FRESHNESS_COLORS = [
    "#16A34A",   # Very Fresh  – deep green
    "#22C55E",   # Fresh       – medium green
    "#EAB308",   # Ripe        – amber/yellow
    "#F97316",   # Overripe    – orange
    "#DC2626",   # Deteriorating – red
    "#6B7280",   # Spoiled     – grey
]

# Friendly shelf-life estimates per freshness stage
FRESHNESS_SHELF_LIFE = [
    "7–10 days",
    "4–7 days",
    "1–3 days",
    "Eat today",
    "Process immediately",
    "Discard",
]

FRESHNESS_EMOJI = ["🌿", "✅", "🟡", "🟠", "⚠️", "❌"]

RIPENESS_EMOJI = ["🔵", "🟢", "🟡", "🟠"]

# Default ConvNeXt-Tiny feature dimension
CONVNEXT_FEATURES_DIM = 768


def format_produce_stage_name(stage_idx: int, produce_class: str) -> str:
    """
    Produce-appropriate stage terminology.
    Prevents non-climacteric vegetables and allium bulbs (Onion, Potato, Garlic, Carrot)
    from being inappropriately labeled 'Ripe' when they are actually aged or softening.
    """
    p_lower = produce_class.lower()
    is_bulb_or_root = any(k in p_lower for k in ["onion", "potato", "garlic", "carrot", "turnip", "radish"])
    if is_bulb_or_root:
        vegetable_stages = [
            "Peak Sound (Firm)",
            "Sound / Fresh",
            "Aged / Softening",
            "Over-aged / Bruised",
            "Deteriorating (Mold)",
            "Spoiled (Rotten)",
        ]
        return vegetable_stages[min(max(0, stage_idx), len(vegetable_stages) - 1)]
    return FRESHNESS_STAGES[min(max(0, stage_idx), len(FRESHNESS_STAGES) - 1)]


def format_ripeness_stage_name(ripeness_idx: int, produce_class: str) -> str:
    p_lower = produce_class.lower()
    is_non_climacteric = any(k in p_lower for k in ["onion", "potato", "garlic", "carrot", "broccoli"])
    if is_non_climacteric:
        return "Mature / Sound" if ripeness_idx <= 2 else "Over-mature / Soft"
    return RIPENESS_STAGES[min(max(0, ripeness_idx), len(RIPENESS_STAGES) - 1)]


# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FreshnessResult:
    freshness_stage: str = "Unknown"
    freshness_stage_idx: int = 0
    freshness_prob: float = 0.0
    freshness_probs: List[float] = field(default_factory=list)
    ripeness_stage: str = "Unknown"
    ripeness_stage_idx: int = 0
    ripeness_prob: float = 0.0
    ripeness_probs: List[float] = field(default_factory=list)
    shelf_life_estimate: str = "N/A"
    badge_color: str = "#16A34A"
    freshness_emoji: str = "🌿"
    ripeness_emoji: str = "🟢"
    visual_features: Optional[List[float]] = None   # 768-d backbone embedding
    model_mode: str = "heuristic"                   # "convnext" | "heuristic"
    produce_class: str = ""

    def freshness_score_0_100(self) -> int:
        """Convert freshness stage to a 0-100 score (100 = Very Fresh)."""
        n = len(FRESHNESS_STAGES)
        return max(0, int(round((n - 1 - self.freshness_stage_idx) / (n - 1) * 100)))

    def arrhenius_shelf_life(self, temp_c: float = 22.0):
        """Compute scientific shelf life using Arrhenius Kinetics equation."""
        from freshai.arrhenius import estimate_arrhenius_shelf_life
        return estimate_arrhenius_shelf_life(
            produce_name=self.produce_class,
            freshness_score_0_100=self.freshness_score_0_100(),
            current_temp_c=temp_c,
        )

    def to_dict(self) -> Dict[str, Any]:
        arr_res = self.arrhenius_shelf_life(22.0)
        return {
            "freshness_stage": self.freshness_stage,
            "freshness_probability": round(self.freshness_prob, 4),
            "freshness_probability_percent": f"{round(self.freshness_prob * 100, 1)}%",
            "freshness_score": self.freshness_score_0_100(),
            "freshness_all_probs": {
                FRESHNESS_STAGES[i]: round(float(p), 4)
                for i, p in enumerate(self.freshness_probs)
            } if self.freshness_probs else {},
            "ripeness_stage": self.ripeness_stage,
            "ripeness_probability": round(self.ripeness_prob, 4),
            "ripeness_probability_percent": f"{round(self.ripeness_prob * 100, 1)}%",
            "ripeness_all_probs": {
                RIPENESS_STAGES[i]: round(float(p), 4)
                for i, p in enumerate(self.ripeness_probs)
            } if self.ripeness_probs else {},
            "shelf_life_estimate": self.shelf_life_estimate,
            "arrhenius_kinetics": arr_res.to_dict(),
            "badge_color": self.badge_color,
            "freshness_emoji": self.freshness_emoji,
            "model_mode": self.model_mode,
            "produce_class": self.produce_class,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ConvNeXt-Tiny Multi-Task Architecture
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class ConvNeXtMultiTaskHead(nn.Module):
        """
        Multi-task wrapper around ConvNeXt-Tiny.
        Shared backbone → two prediction heads (freshness + ripeness).
        """

        def __init__(
            self,
            num_freshness_classes: int = 6,
            num_ripeness_classes: int = 4,
            pretrained: bool = True,
        ):
            super().__init__()

            # Shared backbone (ConvNeXt-Tiny)
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)

            # Strip the classifier head — keep features up to the avgpool
            self.feature_extractor = nn.Sequential(
                backbone.features,
                backbone.avgpool,
                nn.Flatten(1),
            )
            in_features = CONVNEXT_FEATURES_DIM  # ConvNeXt-Tiny last-stage channels

            # Freshness head
            self.freshness_head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_freshness_classes),
            )

            # Ripeness head
            self.ripeness_head = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_ripeness_classes),
            )

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            features = self.feature_extractor(x)            # (B, 768)
            freshness_logits = self.freshness_head(features)  # (B, 6)
            ripeness_logits = self.ripeness_head(features)    # (B, 4)
            return freshness_logits, ripeness_logits, features


# ──────────────────────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    _CONVNEXT_TRANSFORM = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _to_pil(image: Union[str, np.ndarray, "Image.Image"]) -> "Image.Image":
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, str):
        return Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise ValueError(f"Unsupported image type: {type(image)}")


# ──────────────────────────────────────────────────────────────────────────────
# Color Heuristic Fallback (works without PyTorch / trained weights)
# ──────────────────────────────────────────────────────────────────────────────

def _softmax(logits: List[float]) -> List[float]:
    exp_vals = [math.exp(v - max(logits)) for v in logits]
    s = sum(exp_vals)
    return [v / s for v in exp_vals]


def _rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Simple RGB → HSV (r,g,b in 0-255). Returns h in 0-360, s,v in 0-1."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    delta = mx - mn
    v = mx
    s = delta / mx if mx != 0 else 0.0
    if delta == 0:
        h = 0.0
    elif mx == r:
        h = 60 * (((g - b) / delta) % 6)
    elif mx == g:
        h = 60 * ((b - r) / delta + 2)
    else:
        h = 60 * ((r - g) / delta + 4)
    return h, s, v


def _analyze_crop_hsv(img_pil: "Image.Image") -> Tuple[float, float, float, float, float, float, float]:
    """
    Analyse HSV and color distribution statistics of a produce crop.
    Accurately distinguishes true black mold / fungal rot from natural dry peel / shading.
    Returns (mean_h, mean_s, mean_v, brown_ratio, green_ratio, yellow_ratio, spot_ratio).
    """
    img_small = img_pil.resize((64, 64)).convert("RGB")
    pixels = np.array(img_small).reshape(-1, 3).tolist()
    total = len(pixels)

    h_sum = s_sum = v_sum = 0.0
    brown_count = 0
    green_count = 0
    yellow_count = 0
    spot_count = 0

    for (r, g, b) in pixels:
        h, s, v = _rgb_to_hsv(r, g, b)
        h_sum += h
        s_sum += s
        v_sum += v

        # True dark rot / black fungal mold (near-black or deep dark sunken spots)
        if v < 0.18 or (v < 0.22 and s < 0.28):
            spot_count += 1
            brown_count += 1
        # Discoloration / bruising (moderate browning on non-onion crops)
        elif 12 <= h <= 35 and s > 0.35 and 0.22 <= v < 0.45:
            brown_count += 1
        # Fresh green indicator
        elif 75 <= h <= 165 and s > 0.20 and v > 0.25:
            green_count += 1
        # Yellow / Ripe indicator
        elif 45 < h < 75 and s > 0.25 and v > 0.40:
            yellow_count += 1

    mean_h = h_sum / total if total > 0 else 0.0
    mean_s = s_sum / total if total > 0 else 0.0
    mean_v = v_sum / total if total > 0 else 0.0
    brown_ratio = brown_count / total if total > 0 else 0.0
    green_ratio = green_count / total if total > 0 else 0.0
    yellow_ratio = yellow_count / total if total > 0 else 0.0
    spot_ratio = spot_count / total if total > 0 else 0.0

    return mean_h, mean_s, mean_v, brown_ratio, green_ratio, yellow_ratio, spot_ratio


def estimate_freshness_heuristic(
    crop_image: Union[str, np.ndarray, "Image.Image"],
    produce_class: str = "",
) -> FreshnessResult:
    """
    Research-calibrated Freshness & Ripeness Categorization grounded in:
    'Fruits and Vegetables Freshness Categorization Using Deep Learning' (CMC 2022).
    
    Categorizes produce into:
      • Pure-Fresh / Very Fresh (100–80 score, 7–14 days)
      • Fresh / Medium-Fresh / Ripe (60–80 score, 3–7 days)
      • Overripe / Deteriorating (20–40 score, 1–2 days)
      • Spoiled (0 score, Discard)
    """
    img_pil = _to_pil(crop_image)
    mean_h, mean_s, mean_v, brown_ratio, green_ratio, yellow_ratio, spot_ratio = _analyze_crop_hsv(img_pil)

    p_lower = produce_class.lower().strip()
    score = 0.5  # Base score (0 = Very Fresh, 5 = Spoiled)
    ripeness_idx = 2  # Default Ripe

    # ── Produce-Specific Visual Calibration (CMC 2022 Research Guidelines) ──
    if "banana" in p_lower:
        # Green Banana = Unripe, Pure Fresh
        if green_ratio > 0.35:
            score = 0.2
            ripeness_idx = 0  # Unripe
        # Golden Yellow Banana with minimal spots = Pure Fresh / Fresh
        elif yellow_ratio > 0.45 and spot_ratio < 0.08:
            score = 1.0
            ripeness_idx = 1  # Nearly Ripe / Fresh
        # Banana with sugar freckles = Medium Fresh / Peak Ripe (100% edible)
        elif 0.08 <= spot_ratio <= 0.25:
            score = 2.0
            ripeness_idx = 2  # Ripe / Medium-Fresh
        # Overripe banana with significant browning
        elif 0.25 < spot_ratio <= 0.50:
            score = 3.4
            ripeness_idx = 3  # Overripe
        # Mostly black/brown banana = Rotten / Spoiled
        else:
            score = 4.8
            ripeness_idx = 3

    elif "tomato" in p_lower:
        # Green/Orange Tomato = Unripe/Turning
        if green_ratio > 0.25:
            score = 0.5
            ripeness_idx = 0
        # Vibrant Glossy Red Tomato = Pure Fresh
        elif (mean_h <= 20 or mean_h >= 340) and mean_s > 0.45 and spot_ratio < 0.05:
            score = 0.4
            ripeness_idx = 2  # Ripe & Pure Fresh
        # Deep red, slight softening = Medium Fresh
        elif spot_ratio < 0.12 and mean_v > 0.30:
            score = 1.8
            ripeness_idx = 2  # Ripe / Medium-Fresh
        # Overripe / soft
        elif spot_ratio <= 0.25:
            score = 3.2
            ripeness_idx = 3
        # Dark rotting lesions / mold
        else:
            score = 4.7
            ripeness_idx = 3

    elif "apple" in p_lower:
        # Crisp green/red apple with healthy sheen
        if (green_ratio > 0.30 or mean_s > 0.40) and spot_ratio < 0.05:
            score = 0.3
            ripeness_idx = 1 if green_ratio > 0.30 else 2
        # Slight dullness or minor single spot = Medium Fresh
        elif spot_ratio < 0.12:
            score = 1.8
            ripeness_idx = 2
        # Moderate bruising / softness = Overripe
        elif spot_ratio <= 0.28:
            score = 3.3
            ripeness_idx = 3
        # Large brown decay / rot
        else:
            score = 4.8
            ripeness_idx = 3

    elif "onion" in p_lower:
        # Onion Specific Calibration (Distinguishing healthy dry outer peel from black mold)
        if spot_ratio >= 0.18 or (brown_ratio >= 0.30 and mean_v < 0.30):
            score = 4.8  # Extensive black mold / deep rot -> Spoiled
            ripeness_idx = 3
        elif spot_ratio >= 0.08:
            score = 3.8  # Notable black mold patches -> Deteriorating (Peel or use today)
            ripeness_idx = 3
        elif spot_ratio >= 0.035:
            score = 2.0  # Normal harvest dust / dry papery flakes -> Medium-Fresh (3–5 days)
            ripeness_idx = 2
        elif spot_ratio >= 0.01:
            score = 1.0  # Normal healthy dry onion -> Fresh (1–2 weeks)
            ripeness_idx = 2
        else:
            score = 0.4  # Pristine clean onion skin -> Very Fresh (2–3 weeks)
            ripeness_idx = 1

    elif "potato" in p_lower:
        # Potato Specific Calibration (Greening / Black Spots / Sprouting)
        if green_ratio > 0.15:
            score = 4.5  # Greening (Solanine) -> Deteriorating / Toxic
            ripeness_idx = 0
        elif spot_ratio >= 0.10:
            score = 4.7  # Black rot / soft spots -> Spoiled
            ripeness_idx = 3
        elif spot_ratio >= 0.04:
            score = 3.6  # Minor eyes / surface spots -> Deteriorating
            ripeness_idx = 3
        elif spot_ratio < 0.02 and mean_v > 0.35:
            score = 0.8  # Clean, firm skin -> Fresh
            ripeness_idx = 2
        else:
            score = 2.0  # Medium Fresh
            ripeness_idx = 2

    elif "orange" in p_lower or "citrus" in p_lower or "lemon" in p_lower:
        # Bright vibrant citrus rind = Pure Fresh
        if (yellow_ratio > 0.35 or 20 <= mean_h <= 60) and mean_s > 0.35 and spot_ratio < 0.06:
            score = 0.4
            ripeness_idx = 2
        # Thinned or slightly dull rind = Medium Fresh
        elif spot_ratio < 0.15:
            score = 1.7
            ripeness_idx = 2
        # Browning or green/white mold patch = Rotten
        elif spot_ratio > 0.25 or mean_v < 0.25:
            score = 4.6
            ripeness_idx = 3
        else:
            score = 3.0
            ripeness_idx = 3

    elif any(v in p_lower for v in ["cucumber", "brinjal", "eggplant", "pepper", "capsicum", "chili"]):
        # Deep solid green / vibrant purple = Pure Fresh
        if (green_ratio > 0.30 or (mean_h >= 260 and mean_h <= 310)) and spot_ratio < 0.08:
            score = 0.3
            ripeness_idx = 1
        # Slight yellowing or softness = Medium Fresh
        elif spot_ratio < 0.20:
            score = 2.0
            ripeness_idx = 2
        # Wrinkled / dark rotting spots
        else:
            score = 4.5
            ripeness_idx = 3

    else:
        # General produce evaluation based on color vibrancy and surface degradation
        score += spot_ratio * 4.5
        if mean_s < 0.20:
            score += 1.2  # Dull / dried
        elif mean_s > 0.45 and spot_ratio < 0.05:
            score -= 0.5  # High natural saturation & clear surface = Fresh

        if mean_v < 0.25:
            score += 1.8  # Very dark / decaying
        elif mean_v > 0.50 and spot_ratio < 0.05:
            score -= 0.3  # Bright & glossy

        # General ripeness
        if green_ratio > 0.35:
            ripeness_idx = 0
        elif yellow_ratio > 0.30:
            ripeness_idx = 2
        elif mean_h <= 25 and mean_s > 0.35:
            ripeness_idx = 2
        elif spot_ratio > 0.30:
            ripeness_idx = 3
        else:
            ripeness_idx = 1

    # Clamp score to [0.0, 5.0]
    score = max(0.0, min(5.0, score))
    freshness_idx = int(round(score))
    freshness_idx = min(freshness_idx, len(FRESHNESS_STAGES) - 1)

    raw_logits = [
        max(0.01, 2.5 - abs(i - score))
        for i in range(len(FRESHNESS_STAGES))
    ]
    freshness_probs = _softmax(raw_logits)
    freshness_prob = freshness_probs[freshness_idx]

    ripe_logits = [
        max(0.01, 2.0 - abs(i - ripeness_idx))
        for i in range(len(RIPENESS_STAGES))
    ]
    ripeness_probs = _softmax(ripe_logits)
    ripeness_prob = ripeness_probs[ripeness_idx]

    return FreshnessResult(
        freshness_stage=format_produce_stage_name(freshness_idx, produce_class),
        freshness_stage_idx=freshness_idx,
        freshness_prob=freshness_prob,
        freshness_probs=freshness_probs,
        ripeness_stage=format_ripeness_stage_name(ripeness_idx, produce_class),
        ripeness_stage_idx=ripeness_idx,
        ripeness_prob=ripeness_prob,
        ripeness_probs=ripeness_probs,
        shelf_life_estimate=FRESHNESS_SHELF_LIFE[freshness_idx],
        badge_color=FRESHNESS_COLORS[freshness_idx],
        freshness_emoji=FRESHNESS_EMOJI[freshness_idx],
        ripeness_emoji=RIPENESS_EMOJI[ripeness_idx],
        visual_features=None,
        model_mode="heuristic",
        produce_class=produce_class,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main Detector Class
# ──────────────────────────────────────────────────────────────────────────────

class ConvNeXtFreshnessDetector:
    """
    FreshAI Step 2 — Freshness Detector using ConvNeXt-Tiny and Research-Calibrated Fusion.
    """

    def __init__(
        self,
        weights_path: str = "runs/freshness/best_freshness.pth",
        device: Optional[str] = None,
    ):
        self.weights_path = weights_path
        self.device = device or ("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
        self.model = None
        self.mode = "heuristic"
        self._try_load_model()

    def _try_load_model(self):
        """Try to load ConvNeXt-Tiny weights; silently fall back to heuristic."""
        if not TORCH_AVAILABLE:
            print("[FreshnessDetector] PyTorch not available — using color heuristic mode.")
            return

        if not os.path.exists(self.weights_path):
            print(f"[FreshnessDetector] No trained weights found at '{self.weights_path}'. "
                  f"Using color heuristic mode (works without training).")
            return

        try:
            import torchvision.models as models
            in_features = CONVNEXT_FEATURES_DIM

            class _MultiTaskModel(nn.Module):
                def __init__(self_inner):
                    super().__init__()
                    backbone = models.convnext_tiny(weights=None)
                    self_inner.feature_extractor = nn.Sequential(
                        backbone.features, backbone.avgpool, nn.Flatten(1)
                    )
                    self_inner.freshness_head = nn.Sequential(
                        nn.LayerNorm(in_features), nn.Linear(in_features, 256),
                        nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(FRESHNESS_STAGES))
                    )
                    self_inner.ripeness_head = nn.Sequential(
                        nn.LayerNorm(in_features), nn.Linear(in_features, 128),
                        nn.GELU(), nn.Dropout(0.3), nn.Linear(128, len(RIPENESS_STAGES))
                    )

                def forward(self_inner, x):
                    feats = self_inner.feature_extractor(x)
                    return self_inner.freshness_head(feats), self_inner.ripeness_head(feats), feats

            model = _MultiTaskModel()
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.mode = "convnext"
            val_acc = checkpoint.get("val_freshness_acc", "?")
            print(f"[FreshnessDetector] ✅ ConvNeXt-Tiny model loaded from '{self.weights_path}' "
                  f"(val_acc={val_acc})")
        except Exception as e:
            print(f"[FreshnessDetector] Could not load model weights ({e}) — using heuristic fallback.")
            self.model = None
            self.mode = "heuristic"

    def predict(
        self,
        image: Union[str, np.ndarray, "Image.Image"],
        produce_class: str = "",
    ) -> FreshnessResult:
        """
        Run freshness prediction on an image (full image or pre-cropped bbox).
        Incorporates defect safety guard for black mold, spots, and surface rot.
        """
        img_pil = _to_pil(image)
        heur_result = estimate_freshness_heuristic(img_pil, produce_class)

        if self.mode == "convnext" and self.model is not None:
            conv_result = self._predict_convnext(img_pil, produce_class)
            
            # Defect & Biological Food Safety Guard:
            # If EITHER the neural network or visual color/spot heuristics detects decay,
            # overripeness, or black mold, the degraded state MUST be respected.
            # Never falsely report "Very Fresh" if one of the models detects decay.
            fused_idx = max(conv_result.freshness_stage_idx, heur_result.freshness_stage_idx)
            if conv_result.freshness_stage_idx == fused_idx:
                return conv_result
            return heur_result
        else:
            return heur_result

    def _predict_convnext(self, img_pil: "Image.Image", produce_class: str) -> FreshnessResult:
        """Run ConvNeXt-Tiny multi-task inference."""
        tensor = _CONVNEXT_TRANSFORM(img_pil).unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            freshness_logits, ripeness_logits, features = self.model(tensor)

        # Softmax probabilities
        f_probs = torch.softmax(freshness_logits, dim=1)[0].tolist()
        r_probs = torch.softmax(ripeness_logits, dim=1)[0].tolist()
        feats = features[0].tolist()

        f_idx = int(torch.argmax(freshness_logits, dim=1).item())
        r_idx = int(torch.argmax(ripeness_logits, dim=1).item())

        return FreshnessResult(
            freshness_stage=format_produce_stage_name(f_idx, produce_class),
            freshness_stage_idx=f_idx,
            freshness_prob=f_probs[f_idx],
            freshness_probs=f_probs,
            ripeness_stage=format_ripeness_stage_name(r_idx, produce_class),
            ripeness_stage_idx=r_idx,
            ripeness_prob=r_probs[r_idx],
            ripeness_probs=r_probs,
            shelf_life_estimate=FRESHNESS_SHELF_LIFE[f_idx],
            badge_color=FRESHNESS_COLORS[f_idx],
            freshness_emoji=FRESHNESS_EMOJI[f_idx],
            ripeness_emoji=RIPENESS_EMOJI[r_idx],
            visual_features=feats,
            model_mode="convnext",
            produce_class=produce_class,
        )

    def predict_from_detection(
        self,
        full_image: Union[str, np.ndarray, "Image.Image"],
        bbox_xyxy: List[float],
        produce_class: str = "",
    ) -> FreshnessResult:
        """
        Crop a detected bounding box from the full image and run freshness prediction.

        Args:
            full_image: Full scene image
            bbox_xyxy: [x1, y1, x2, y2] bounding box (pixel coordinates)
            produce_class: Produce class name from YOLO detector
        """
        img_pil = _to_pil(full_image)
        w, h = img_pil.size

        x1 = max(0, int(bbox_xyxy[0]))
        y1 = max(0, int(bbox_xyxy[1]))
        x2 = min(w, int(bbox_xyxy[2]))
        y2 = min(h, int(bbox_xyxy[3]))

        # Guard: bbox must have positive area
        if x2 <= x1 or y2 <= y1:
            return self.predict(img_pil, produce_class)

        crop = img_pil.crop((x1, y1, x2, y2))
        return self.predict(crop, produce_class)

    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, "Image.Image"]],
        produce_classes: Optional[List[str]] = None,
    ) -> List[FreshnessResult]:
        """Run freshness prediction on a list of crops."""
        results = []
        for i, img in enumerate(images):
            cls = produce_classes[i] if produce_classes and i < len(produce_classes) else ""
            results.append(self.predict(img, cls))
        return results

    @property
    def is_model_loaded(self) -> bool:
        return self.mode == "convnext" and self.model is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "model_loaded": self.is_model_loaded,
            "weights_path": self.weights_path,
            "torch_available": TORCH_AVAILABLE,
            "freshness_classes": FRESHNESS_STAGES,
            "ripeness_classes": RIPENESS_STAGES,
        }
