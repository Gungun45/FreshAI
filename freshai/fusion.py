"""
FreshAI - Stage 4: Multimodal Feature Combination & Fusion Vector

Combines outputs from:
1. ConvNeXt-Tiny:
   • Freshness Probabilities (6 classes)
   • Ripeness Probabilities (4 classes)
   • Deep Visual Features (768-d backbone embedding)
2. OpenCV:
   • Color Features (RGB, HSV, LAB, Region %)
   • Texture Features (GLCM Contrast, Homogeneity, Energy, Dissimilarity, Correlation, Roughness)
3. YOLO Defect Detection & Segmentation:
   • Defect Area %
   • Defect Count & Severity
   • Per-defect breakdown (Bruise %, Rot %, Mold %, Black Spot %, Crack %, etc.)

Diagram:
ConvNeXt-Tiny
 │
 ├── Freshness
 ├── Ripeness
 └── Deep Visual Features
 |
OpenCV 
 │ 
 |── Color ───┤
 └── Texture ─ ┤
 │
YOLO 
 │
 └── Defect % ┘
 │
 ▼
 Feature Vector
   Freshness Probability → 0.82
   Ripeness Probability  → 0.91
   Defect Area           → 7.4%
   Average Hue           → 12.4
   Saturation            → 82.6
   Texture Contrast      → 0.42
   Texture Homogeneity   → 0.81
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from freshai.freshness_detector import FreshnessResult, FRESHNESS_STAGES, RIPENESS_STAGES
from freshai.defect_detector import DefectResult
from freshai.feature_extractor import ProduceExtractedFeatures


@dataclass
class ProduceFeatureVector:
    """
    Standardized Multimodal Feature Vector combining ConvNeXt deep visual features,
    OpenCV color & GLCM texture metrics, and YOLO defect detection & segmentation.
    """
    # Produce metadata
    produce_class: str = ""

    # ConvNeXt-Tiny Signals
    freshness_prob: float = 0.0
    freshness_stage: str = "Unknown"
    freshness_stage_idx: int = 0
    freshness_all_probs: List[float] = field(default_factory=list) # 6 values

    ripeness_prob: float = 0.0
    ripeness_stage: str = "Unknown"
    ripeness_stage_idx: int = 0
    ripeness_all_probs: List[float] = field(default_factory=list)  # 4 values

    deep_visual_features: Optional[List[float]] = None             # 768-d embedding

    # YOLO Defect Detection & Segmentation Signals
    defect_area_pct: float = 0.0       # e.g. 7.4%
    defect_count: int = 0
    defect_severity_score: float = 0.0
    bruise_area_pct: float = 0.0
    rot_area_pct: float = 0.0
    mold_area_pct: float = 0.0
    spot_area_pct: float = 0.0
    crack_area_pct: float = 0.0
    wrinkle_area_pct: float = 0.0

    # OpenCV Color Signals
    avg_hue: float = 0.0               # e.g. 12.4
    avg_saturation: float = 0.0        # e.g. 82.6
    avg_brightness: float = 0.0
    mean_r: float = 0.0
    mean_g: float = 0.0
    mean_b: float = 0.0
    mean_l: float = 0.0                # LAB Lightness
    mean_a: float = 0.0                # LAB a*
    mean_b_lab: float = 0.0            # LAB b*
    chroma: float = 0.0
    green_region_pct: float = 0.0
    yellow_region_pct: float = 0.0
    red_region_pct: float = 0.0
    brown_region_pct: float = 0.0
    dark_decay_pct: float = 0.0
    pale_mold_pct: float = 0.0

    # OpenCV Texture Signals (GLCM & Surface Dynamics)
    texture_contrast: float = 0.0      # e.g. 0.42
    texture_homogeneity: float = 0.0   # e.g. 0.81
    texture_dissimilarity: float = 0.0
    texture_energy: float = 0.0
    texture_correlation: float = 0.0
    texture_entropy: float = 0.0
    surface_roughness: float = 0.0
    edge_density: float = 0.0

    def to_highlight_dict(self) -> Dict[str, Any]:
        """Returns the key visual metrics matching the prompt's reference sample."""
        return {
            "Freshness Probability": round(self.freshness_prob, 2),
            "Ripeness Probability": round(self.ripeness_prob, 2),
            "Defect Area": f"{round(self.defect_area_pct, 1)}%",
            "Average Hue": round(self.avg_hue, 1),
            "Saturation": round(self.avg_saturation, 1),
            "Texture Contrast": round(self.texture_contrast, 2),
            "Texture Homogeneity": round(self.texture_homogeneity, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "produce_class": self.produce_class,
            "convnext": {
                "freshness_stage": self.freshness_stage,
                "freshness_prob": round(self.freshness_prob, 4),
                "freshness_stage_idx": self.freshness_stage_idx,
                "freshness_all_probs": [round(p, 4) for p in self.freshness_all_probs],
                "ripeness_stage": self.ripeness_stage,
                "ripeness_prob": round(self.ripeness_prob, 4),
                "ripeness_stage_idx": self.ripeness_stage_idx,
                "ripeness_all_probs": [round(p, 4) for p in self.ripeness_all_probs],
                "has_deep_features": self.deep_visual_features is not None,
                "deep_features_dim": len(self.deep_visual_features) if self.deep_visual_features else 0,
            },
            "yolo_defect": {
                "defect_area_percent": round(self.defect_area_pct, 2),
                "defect_count": self.defect_count,
                "defect_severity_score": round(self.defect_severity_score, 3),
                "breakdown": {
                    "rot_pct": round(self.rot_area_pct, 2),
                    "mold_pct": round(self.mold_area_pct, 2),
                    "bruise_pct": round(self.bruise_area_pct, 2),
                    "spot_pct": round(self.spot_area_pct, 2),
                    "crack_pct": round(self.crack_area_pct, 2),
                    "wrinkle_pct": round(self.wrinkle_area_pct, 2),
                }
            },
            "opencv_color": {
                "avg_hue": round(self.avg_hue, 1),
                "avg_saturation": round(self.avg_saturation, 1),
                "avg_brightness": round(self.avg_brightness, 1),
                "rgb_means": [round(self.mean_r, 1), round(self.mean_g, 1), round(self.mean_b, 1)],
                "cielab": [round(self.mean_l, 1), round(self.mean_a, 1), round(self.mean_b_lab, 1)],
                "chroma": round(self.chroma, 1),
                "regions": {
                    "green_pct": round(self.green_region_pct, 1),
                    "yellow_pct": round(self.yellow_region_pct, 1),
                    "red_pct": round(self.red_region_pct, 1),
                    "brown_pct": round(self.brown_region_pct, 1),
                    "dark_decay_pct": round(self.dark_decay_pct, 1),
                    "pale_mold_pct": round(self.pale_mold_pct, 1),
                }
            },
            "opencv_texture": {
                "contrast": round(self.texture_contrast, 4),
                "homogeneity": round(self.texture_homogeneity, 4),
                "dissimilarity": round(self.texture_dissimilarity, 4),
                "energy": round(self.texture_energy, 4),
                "correlation": round(self.texture_correlation, 4),
                "entropy": round(self.texture_entropy, 4),
                "roughness": round(self.surface_roughness, 2),
                "edge_density": round(self.edge_density, 4),
            },
            "key_summary": self.to_highlight_dict(),
        }

    @staticmethod
    def feature_names(include_deep: bool = False, deep_dim: int = 768) -> List[str]:
        names = [
            # ConvNeXt predictions
            "freshness_prob", "freshness_stage_idx",
            "f_prob_very_fresh", "f_prob_fresh", "f_prob_ripe", "f_prob_overripe", "f_prob_deteriorating", "f_prob_spoiled",
            "ripeness_prob", "ripeness_stage_idx",
            "r_prob_unripe", "r_prob_nearly_ripe", "r_prob_ripe", "r_prob_overripe",
            # YOLO Defect metrics
            "defect_area_pct", "defect_count", "defect_severity_score",
            "bruise_area_pct", "rot_area_pct", "mold_area_pct", "spot_area_pct", "crack_area_pct", "wrinkle_area_pct",
            # OpenCV Color
            "avg_hue", "avg_saturation", "avg_brightness",
            "mean_r", "mean_g", "mean_b",
            "mean_l", "mean_a", "mean_b_lab", "chroma",
            "green_region_pct", "yellow_region_pct", "red_region_pct",
            "brown_region_pct", "dark_decay_pct", "pale_mold_pct",
            # OpenCV Texture
            "texture_contrast", "texture_homogeneity", "texture_dissimilarity",
            "texture_energy", "texture_correlation", "texture_entropy",
            "surface_roughness", "edge_density",
        ]
        if include_deep:
            names += [f"deep_feat_{i}" for i in range(deep_dim)]
        return names

    def to_dense_vector(self, include_deep: bool = False) -> np.ndarray:
        """Convert into numerical 1D float32 numpy array for ML models."""
        f_probs = self.freshness_all_probs if len(self.freshness_all_probs) == 6 else [0.0] * 6
        r_probs = self.ripeness_all_probs if len(self.ripeness_all_probs) == 4 else [0.0] * 4

        vec = [
            self.freshness_prob, float(self.freshness_stage_idx),
            f_probs[0], f_probs[1], f_probs[2], f_probs[3], f_probs[4], f_probs[5],
            self.ripeness_prob, float(self.ripeness_stage_idx),
            r_probs[0], r_probs[1], r_probs[2], r_probs[3],
            self.defect_area_pct, float(self.defect_count), self.defect_severity_score,
            self.bruise_area_pct, self.rot_area_pct, self.mold_area_pct, self.spot_area_pct, self.crack_area_pct, self.wrinkle_area_pct,
            self.avg_hue, self.avg_saturation, self.avg_brightness,
            self.mean_r, self.mean_g, self.mean_b,
            self.mean_l, self.mean_a, self.mean_b_lab, self.chroma,
            self.green_region_pct, self.yellow_region_pct, self.red_region_pct,
            self.brown_region_pct, self.dark_decay_pct, self.pale_mold_pct,
            self.texture_contrast, self.texture_homogeneity, self.texture_dissimilarity,
            self.texture_energy, self.texture_correlation, self.texture_entropy,
            self.surface_roughness, self.edge_density,
        ]

        if include_deep:
            if self.deep_visual_features and len(self.deep_visual_features) > 0:
                vec.extend(self.deep_visual_features)
            else:
                vec.extend([0.0] * 768)

        return np.array(vec, dtype=np.float32)


def assemble_feature_vector(
    freshness_res: FreshnessResult,
    defect_res: DefectResult,
    feature_res: ProduceExtractedFeatures,
    produce_class: str = "",
) -> ProduceFeatureVector:
    """
    Assemble the complete multimodal feature vector from the three subsystems.
    """
    col = feature_res.color
    tex = feature_res.texture
    bk = defect_res.defect_type_breakdown

    return ProduceFeatureVector(
        produce_class=produce_class or freshness_res.produce_class,
        # ConvNeXt
        freshness_prob=freshness_res.freshness_prob,
        freshness_stage=freshness_res.freshness_stage,
        freshness_stage_idx=freshness_res.freshness_stage_idx,
        freshness_all_probs=freshness_res.freshness_probs,
        ripeness_prob=freshness_res.ripeness_prob,
        ripeness_stage=freshness_res.ripeness_stage,
        ripeness_stage_idx=freshness_res.ripeness_stage_idx,
        ripeness_all_probs=freshness_res.ripeness_probs,
        deep_visual_features=freshness_res.visual_features,
        # YOLO Defects
        defect_area_pct=defect_res.total_defect_area_pct,
        defect_count=defect_res.defect_count,
        defect_severity_score=defect_res.defect_severity_score,
        bruise_area_pct=bk.get("Bruise", 0.0),
        rot_area_pct=bk.get("Rot", 0.0),
        mold_area_pct=bk.get("Mold", 0.0),
        spot_area_pct=bk.get("Black Spot", 0.0) + bk.get("Fungal Spot", 0.0),
        crack_area_pct=bk.get("Crack", 0.0),
        wrinkle_area_pct=bk.get("Wrinkle", 0.0),
        # OpenCV Color
        avg_hue=col.avg_hue,
        avg_saturation=col.avg_saturation,
        avg_brightness=col.avg_brightness,
        mean_r=col.mean_r,
        mean_g=col.mean_g,
        mean_b=col.mean_b,
        mean_l=col.mean_l,
        mean_a=col.mean_a,
        mean_b_lab=col.mean_b_lab,
        chroma=col.chroma,
        green_region_pct=col.green_region_pct,
        yellow_region_pct=col.yellow_region_pct,
        red_region_pct=col.red_region_pct,
        brown_region_pct=col.brown_region_pct,
        dark_decay_pct=col.dark_decay_pct,
        pale_mold_pct=col.pale_mold_pct,
        # OpenCV Texture
        texture_contrast=tex.contrast,
        texture_homogeneity=tex.homogeneity,
        texture_dissimilarity=tex.dissimilarity,
        texture_energy=tex.energy,
        texture_correlation=tex.correlation,
        texture_entropy=tex.entropy,
        surface_roughness=tex.roughness,
        edge_density=tex.edge_density,
    )
