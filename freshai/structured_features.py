"""
FreshAI - Stage 5: Structured Feature Representation
Fuses Visual Features (ConvNeXt + YOLO + OpenCV) with Environmental Data (Temperature, Humidity)
and Timeline/Storage Inputs into a unified structured tabular feature representation for XGBoost & LightGBM.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import numpy as np

from freshai.fusion import ProduceFeatureVector

SUPPORTED_PRODUCE_CLASSES = [
    "Apple",
    "Banana",
    "Tomato",
    "Onion",
    "Watermelon",
    "Orange",
    "Carrot",
    "Broccoli",
    "Potato",
    "Other",
]

STORAGE_CONDITIONS = [
    "Refrigerated",
    "Pantry / Room Temperature",
    "Cool Cellar",
    "Warm / Direct Sunlight",
]


@dataclass
class EnvironmentalData:
    """Environmental parameters impacting produce metabolism and degradation rate."""
    temperature_c: float = 22.0     # Ambient temperature in degrees Celsius
    humidity_pct: float = 60.0      # Relative humidity (0 - 100%)

    @property
    def vapor_pressure_deficit(self) -> float:
        """
        Calculates Vapor Pressure Deficit (VPD in kPa), a primary driver of produce moisture loss.
        """
        # Saturated vapor pressure (Tetens equation)
        es = 0.61078 * math.exp((17.27 * self.temperature_c) / (self.temperature_c + 237.3))
        ea = es * (self.humidity_pct / 100.0)
        return max(0.0, es - ea)

    @property
    def thermal_acceleration_factor(self) -> float:
        """
        Biochemical rate acceleration based on Q10 rule relative to 4°C cold storage.
        Q10 ~ 2.2 for typical horticultural produce.
        """
        q10 = 2.2
        return math.pow(q10, max(-0.5, (self.temperature_c - 4.0) / 10.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_c": round(self.temperature_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "vpd_kpa": round(self.vapor_pressure_deficit, 3),
            "thermal_acceleration_factor": round(self.thermal_acceleration_factor, 2),
        }


@dataclass
class TimelineStorageData:
    """Timeline tracking and post-harvest storage conditions."""
    days_since_purchase: float = 2.0
    storage_condition: str = "Pantry / Room Temperature"
    is_growing_plant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "days_since_purchase": round(self.days_since_purchase, 1),
            "storage_condition": self.storage_condition,
            "is_growing_plant": self.is_growing_plant,
        }


# Standardized Feature Names for Tree Models (XGBoost & LightGBM)
STRUCTURED_FEATURE_NAMES: List[str] = [
    # Visual Signals (ConvNeXt + YOLO + OpenCV)
    "freshness_prob",
    "ripeness_prob",
    "defect_area_pct",
    "defect_count",
    "defect_severity_index",
    "avg_hue",
    "avg_saturation",
    "avg_brightness",
    "lab_lightness",
    "lab_a",
    "lab_b",
    "lab_chroma",
    "green_ratio",
    "yellow_ratio",
    "red_ratio",
    "brown_ratio",
    "dark_spot_ratio",
    "mold_ratio",
    "texture_contrast",
    "texture_dissimilarity",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
    "texture_entropy",
    "surface_roughness",
    "edge_density",
    # Environmental Signals
    "temperature_c",
    "humidity_pct",
    "vapor_pressure_deficit",
    "thermal_acceleration_factor",
    # Timeline & Degradation Dynamics
    "days_since_purchase",
    "effective_metabolic_days",
    "is_growing_plant",
    # Storage Condition One-Hot (4 categories)
    "storage_refrigerated",
    "storage_pantry",
    "storage_cellar",
    "storage_warm",
    # Produce Type One-Hot (10 categories)
    "produce_apple",
    "produce_banana",
    "produce_tomato",
    "produce_onion",
    "produce_watermelon",
    "produce_orange",
    "produce_carrot",
    "produce_broccoli",
    "produce_potato",
    "produce_other",
]


@dataclass
class StructuredProduceInput:
    """Combines produce visual feature vector with environmental and timeline factors."""
    feature_vector: ProduceFeatureVector
    environmental: EnvironmentalData = field(default_factory=EnvironmentalData)
    timeline: TimelineStorageData = field(default_factory=TimelineStorageData)
    produce_class: str = "Produce"

    def to_array(self) -> np.ndarray:
        """Assembles a 1D NumPy array of length matching STRUCTURED_FEATURE_NAMES."""
        fv = self.feature_vector
        env = self.environmental
        tl = self.timeline

        # Effective metabolic days combines elapsed days with thermal acceleration
        eff_days = tl.days_since_purchase * env.thermal_acceleration_factor

        # Storage one-hot
        sc = tl.storage_condition.lower()
        storage_refrig = 1.0 if "refrig" in sc or "fridge" in sc else 0.0
        storage_pantry = 1.0 if "pantry" in sc or "room" in sc else 0.0
        storage_cellar = 1.0 if "cellar" in sc or "cool" in sc else 0.0
        storage_warm = 1.0 if "warm" in sc or "sun" in sc else 0.0

        # Produce type one-hot
        pc = self.produce_class.lower()
        produce_flags = [
            1.0 if "apple" in pc else 0.0,
            1.0 if "banana" in pc else 0.0,
            1.0 if "tomato" in pc else 0.0,
            1.0 if "onion" in pc else 0.0,
            1.0 if "watermelon" in pc or "melon" in pc else 0.0,
            1.0 if "orange" in pc or "citrus" in pc else 0.0,
            1.0 if "carrot" in pc else 0.0,
            1.0 if "broccoli" in pc else 0.0,
            1.0 if "potato" in pc else 0.0,
        ]
        # 'Other' is 1 if none of the above are matched
        produce_other = 1.0 if sum(produce_flags) == 0.0 else 0.0

        features = [
            fv.freshness_prob,
            fv.ripeness_prob,
            fv.defect_area_pct,
            float(fv.defect_count),
            fv.defect_severity_score,
            fv.avg_hue,
            fv.avg_saturation,
            fv.avg_brightness,
            fv.mean_l,
            fv.mean_a,
            fv.mean_b_lab,
            fv.chroma,
            fv.green_region_pct,
            fv.yellow_region_pct,
            fv.red_region_pct,
            fv.brown_region_pct,
            fv.dark_decay_pct,
            fv.pale_mold_pct,
            fv.texture_contrast,
            fv.texture_dissimilarity,
            fv.texture_homogeneity,
            fv.texture_energy,
            fv.texture_correlation,
            fv.texture_entropy,
            fv.surface_roughness,
            fv.edge_density,
            env.temperature_c,
            env.humidity_pct,
            env.vapor_pressure_deficit,
            env.thermal_acceleration_factor,
            tl.days_since_purchase,
            eff_days,
            1.0 if tl.is_growing_plant else 0.0,
            storage_refrig,
            storage_pantry,
            storage_cellar,
            storage_warm,
            *produce_flags,
            produce_other,
        ]
        return np.array(features, dtype=np.float32)

    def to_feature_dict(self) -> Dict[str, float]:
        arr = self.to_array()
        return {name: float(val) for name, val in zip(STRUCTURED_FEATURE_NAMES, arr)}
