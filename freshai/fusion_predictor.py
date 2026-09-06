"""
FreshAI - Stage 4: Freshness Fusion Predictor
Synthesizes ConvNeXt-Tiny + OpenCV Color/Texture + YOLO Defect metrics
to predict final calibrated freshness, produce grade, shelf life, and spoilage risk.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from freshai.freshness_detector import (
    FRESHNESS_STAGES,
    FRESHNESS_COLORS,
    FRESHNESS_EMOJI,
    FRESHNESS_SHELF_LIFE,
    format_produce_stage_name,
)
from freshai.fusion import ProduceFeatureVector
from freshai.arrhenius import estimate_arrhenius_shelf_life

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FusionMLP(nn.Module if TORCH_AVAILABLE else object):
    """Multi-layer perceptron for fusion feature classification."""
    def __init__(self, in_features: int = 47, num_classes: int = 6):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class MultimodalFreshnessAssessment:
    freshness_stage: str = "Fresh"
    freshness_stage_idx: int = 1
    freshness_score: int = 80             # 0 to 100
    freshness_prob: float = 0.85
    shelf_life_estimate: str = "4-7 days"
    quality_grade: str = "Grade A"        # Grade A, Grade B, Grade C, Grade D
    spoilage_risk: str = "Low"            # Low, Moderate, High, Critical
    badge_color: str = "#22C55E"
    freshness_emoji: str = "✅"

    # Penalty breakdown
    base_convnext_score: int = 85
    defect_penalty_pts: float = 0.0
    texture_penalty_pts: float = 0.0
    color_penalty_pts: float = 0.0

    # Key highlights for UI
    defect_area_pct: float = 0.0
    ripeness_stage: str = "Ripe"
    ripeness_prob: float = 0.90
    model_type: str = "multimodal_fusion"

    def arrhenius_shelf_life(self, produce_class: str = "", temp_c: float = 22.0, defect_area_pct: Optional[float] = None):
        defect = self.defect_area_pct if defect_area_pct is None else defect_area_pct
        return estimate_arrhenius_shelf_life(
            produce_name=produce_class,
            freshness_score_0_100=self.freshness_score,
            current_temp_c=temp_c,
            defect_area_pct=defect,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "freshness_stage": self.freshness_stage,
            "freshness_stage_idx": self.freshness_stage_idx,
            "freshness_score": self.freshness_score,
            "freshness_probability": round(self.freshness_prob, 3),
            "shelf_life_estimate": self.shelf_life_estimate,
            "quality_grade": self.quality_grade,
            "spoilage_risk": self.spoilage_risk,
            "badge_color": self.badge_color,
            "freshness_emoji": self.freshness_emoji,
            "penalties": {
                "base_convnext_score": self.base_convnext_score,
                "defect_penalty_points": round(self.defect_penalty_pts, 1),
                "texture_penalty_points": round(self.texture_penalty_pts, 1),
                "color_penalty_points": round(self.color_penalty_pts, 1),
            },
            "defect_area_percent": round(self.defect_area_pct, 2),
            "ripeness_stage": self.ripeness_stage,
            "model_type": self.model_type,
        }


class FreshAIFusionPredictor:
    """
    Multimodal Freshness Fusion Predictor.
    Evaluates produce freshness by fusing deep visual embeddings,
    OpenCV color/texture degradation metrics, and YOLO defect segmentation area.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path or "runs/fusion/best_fusion_model.pt"
        self.model = None
        self.loaded_from_disk = False

        if TORCH_AVAILABLE and os.path.exists(self.weights_path):
            try:
                ckpt = torch.load(self.weights_path, map_location="cpu")
                in_feats = ckpt.get("in_features", 47)
                self.model = FusionMLP(in_features=in_feats, num_classes=6)
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.eval()
                self.loaded_from_disk = True
                print(f"[FusionPredictor] Loaded trained fusion model from {self.weights_path}")
            except Exception as e:
                print(f"[FusionPredictor] Could not load fusion model weights: {e}")
                self.model = None

    def predict(self, feature_vec: ProduceFeatureVector) -> MultimodalFreshnessAssessment:
        """
        Produce a calibrated multimodal freshness evaluation from the unified feature vector.
        """
        # If trained neural fusion model is loaded, run neural inference
        if self.loaded_from_disk and self.model is not None and TORCH_AVAILABLE:
            return self._predict_neural(feature_vec)

        # High-precision physics & empirical quality fusion engine
        return self._predict_calibrated_fusion(feature_vec)

    def _predict_neural(self, fv: ProduceFeatureVector) -> MultimodalFreshnessAssessment:
        dense = fv.to_dense_vector(include_deep=False)
        inp = torch.tensor(dense, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(inp)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        stage_idx = int(np.argmax(probs))
        prob = float(probs[stage_idx])

        # Convert stage to continuous score
        base_score = int(round((5 - stage_idx) / 5.0 * 100))
        # Defect penalty adjustment
        defect_pen = fv.defect_area_pct * 1.5 + fv.defect_severity_score * 25.0
        final_score = int(np.clip(base_score - defect_pen, 0, 100))

        stage_idx = self._score_to_stage_idx(final_score)
        grade, risk = self._compute_grade_and_risk(final_score, fv.defect_area_pct, fv.rot_area_pct, fv.mold_area_pct)

        return MultimodalFreshnessAssessment(
            freshness_stage=format_produce_stage_name(stage_idx, fv.produce_class),
            freshness_stage_idx=stage_idx,
            freshness_score=final_score,
            freshness_prob=prob,
            shelf_life_estimate=FRESHNESS_SHELF_LIFE[stage_idx],
            quality_grade=grade,
            spoilage_risk=risk,
            badge_color=FRESHNESS_COLORS[stage_idx],
            freshness_emoji=FRESHNESS_EMOJI[stage_idx],
            base_convnext_score=base_score,
            defect_penalty_pts=defect_pen,
            texture_penalty_pts=0.0,
            color_penalty_pts=0.0,
            defect_area_pct=fv.defect_area_pct,
            ripeness_stage=fv.ripeness_stage,
            ripeness_prob=fv.ripeness_prob,
            model_type="neural_fusion_mlp",
        )

    def _predict_calibrated_fusion(self, fv: ProduceFeatureVector) -> MultimodalFreshnessAssessment:
        """
        Physics & biological post-harvest degradation calculation.
        Combines ConvNeXt visual baseline with physical defect area % and texture breakdown.
        """
        # 1. Base score from ConvNeXt (0 to 100)
        # Invert stage index: 0 (Very Fresh) -> 100, 5 (Spoiled) -> 0
        base_stage_idx = fv.freshness_stage_idx
        base_score = int(round((5.0 - base_stage_idx) / 5.0 * 100))

        # 2. Defect Penalty Calculation
        # Critical defects (Rot & Mold) have severe penalties
        critical_area = fv.rot_area_pct * 2.5 + fv.mold_area_pct * 3.0
        moderate_area = fv.spot_area_pct * 1.5 + fv.crack_area_pct * 1.2
        minor_area = fv.bruise_area_pct * 0.8 + fv.wrinkle_area_pct * 0.6
        total_defect_penalty = critical_area + moderate_area + minor_area + (fv.defect_severity_score * 15.0)

        # 3. Texture Degradation Penalty
        # Fresh produce has low contrast and high homogeneity; shriveled produce has high roughness and edge cracking
        texture_penalty = 0.0
        if fv.texture_homogeneity < 0.65:
            texture_penalty += (0.65 - fv.texture_homogeneity) * 28.0
        if fv.texture_contrast > 1.0:
            texture_penalty += min(18.0, (fv.texture_contrast - 1.0) * 10.0)
        if fv.surface_roughness > 350:
            texture_penalty += min(12.0, (fv.surface_roughness - 350) / 60.0)
        if fv.edge_density > 0.15:
            texture_penalty += min(8.0, (fv.edge_density - 0.15) * 40.0)

        # 4. Color Degradation Penalty (enzymatic browning and necrosis)
        color_penalty = 0.0
        if fv.brown_region_pct > 5.0:
            color_penalty += (fv.brown_region_pct - 5.0) * 1.2
        if fv.dark_decay_pct > 3.0:
            color_penalty += (fv.dark_decay_pct - 3.0) * 2.0
        if fv.pale_mold_pct > 2.0:
            color_penalty += (fv.pale_mold_pct - 2.0) * 3.0

        # Compute Final Freshness Score
        final_score = base_score - (total_defect_penalty + texture_penalty + color_penalty)
        final_score = int(np.clip(final_score, 0, 100))

        # Determine Stage Index from Final Score
        final_stage_idx = self._score_to_stage_idx(final_score)

        # Calibrate Probability: blend ConvNeXt prob with consistency penalty
        calibrated_prob = max(0.55, min(0.98, fv.freshness_prob * (1.0 - min(0.35, total_defect_penalty / 100.0))))

        # Grade & Spoilage Risk
        grade, risk = self._compute_grade_and_risk(final_score, fv.defect_area_pct, fv.rot_area_pct, fv.mold_area_pct)

        return MultimodalFreshnessAssessment(
            freshness_stage=format_produce_stage_name(final_stage_idx, fv.produce_class),
            freshness_stage_idx=final_stage_idx,
            freshness_score=final_score,
            freshness_prob=calibrated_prob,
            shelf_life_estimate=FRESHNESS_SHELF_LIFE[final_stage_idx],
            quality_grade=grade,
            spoilage_risk=risk,
            badge_color=FRESHNESS_COLORS[final_stage_idx],
            freshness_emoji=FRESHNESS_EMOJI[final_stage_idx],
            base_convnext_score=base_score,
            defect_penalty_pts=total_defect_penalty,
            texture_penalty_pts=texture_penalty,
            color_penalty_pts=color_penalty,
            defect_area_pct=fv.defect_area_pct,
            ripeness_stage=fv.ripeness_stage,
            ripeness_prob=fv.ripeness_prob,
            model_type="calibrated_multimodal_engine",
        )

    def _score_to_stage_idx(self, score: int) -> int:
        if score >= 90:
            return 0  # Very Fresh
        elif score >= 75:
            return 1  # Fresh
        elif score >= 55:
            return 2  # Ripe
        elif score >= 35:
            return 3  # Overripe
        elif score >= 15:
            return 4  # Deteriorating
        else:
            return 5  # Spoiled

    def _compute_grade_and_risk(self, score: int, defect_pct: float, rot_pct: float, mold_pct: float) -> Tuple[str, str]:
        # Critical spoilage overrides
        if rot_pct > 2.0 or mold_pct > 1.5 or score < 20:
            return "Grade D (Reject / Compost)", "Critical Spoilage"

        if score >= 85 and defect_pct < 3.0:
            return "Grade A (Premium Fresh)", "Very Low"
        elif score >= 70 and defect_pct < 8.0:
            return "Grade B (Standard Market)", "Low"
        elif score >= 45 and defect_pct < 18.0:
            return "Grade C (Process / Discount)", "Moderate"
        else:
            return "Grade D (Overripe / Expiring)", "High"
