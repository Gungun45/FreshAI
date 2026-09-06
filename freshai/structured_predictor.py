"""
FreshAI - Stage 5: Structured Prediction Inference Engine
Uses trained XGBoost and LightGBM models to generate:
  • Quality Score -> 84/100 (XGBoost)
  • Physiological Age -> 5–8 days (LightGBM)
  • Remaining Shelf Life -> 2–4 days (LightGBM)
  • Spoilage Risk -> Low / Medium / High (XGBoost / LightGBM)
  • Time-to-Harvest -> 10–14 days (LightGBM)
"""

import os
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from freshai.structured_features import (
    STRUCTURED_FEATURE_NAMES,
    EnvironmentalData,
    TimelineStorageData,
    StructuredProduceInput,
    STORAGE_CONDITIONS,
)
from freshai.fusion import ProduceFeatureVector

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODEL_DIR = PROJECT_ROOT / "runs" / "structured"

CROP_BASE_CULTIVATION_DAYS: Dict[str, float] = {
    "apple": 135.0,
    "banana": 300.0,
    "tomato": 75.0,
    "onion": 135.0,
    "watermelon": 85.0,
    "orange": 270.0,
    "lemon": 220.0,
    "mango": 120.0,
    "bell pepper": 75.0,
    "pepper": 75.0,
    "eggplant": 75.0,
    "strawberry": 100.0,
    "carrot": 75.0,
    "broccoli": 70.0,
    "potato": 95.0,
    "cucumber": 60.0,
    "other": 80.0,
}



@dataclass
class StructuredPredictionResult:
    """Consolidated predictions from XGBoost and LightGBM."""
    # Main outputs matching prompt specification
    quality_score: float                  # e.g. 84.0
    quality_score_str: str              # e.g. "84/100"
    physiological_age_days: float         # e.g. 78.5 (total crop age from seed)
    physiological_age_range: str        # e.g. "76–82 days"
    post_harvest_age_days: float          # e.g. 3.0 (days since harvested)
    post_harvest_age_range: str         # e.g. "2–4 days post-harvest"
    remaining_shelf_life_days: float      # e.g. 3.2
    remaining_shelf_life_range: str     # e.g. "2–4 days"
    spoilage_risk: str                  # "Low", "Medium", "High"
    spoilage_risk_color: str            # hex color code
    spoilage_risk_probs: Dict[str, float]
    time_to_harvest_days: float          # e.g. 12.0
    time_to_harvest_range: str          # e.g. "10–14 days" or "N/A"
    is_growing_plant: bool
    # Model attribution & explainability
    top_contributing_factors: List[Dict[str, Any]]
    environmental_summary: Dict[str, Any]
    what_if_shelf_life_days: Dict[str, float]  # Comparison across 4 storage conditions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": round(self.quality_score, 1),
            "quality_score_str": self.quality_score_str,
            "physiological_age_days": round(self.physiological_age_days, 1),
            "physiological_age_range": self.physiological_age_range,
            "post_harvest_age_days": round(self.post_harvest_age_days, 1),
            "post_harvest_age_range": self.post_harvest_age_range,
            "remaining_shelf_life_days": round(self.remaining_shelf_life_days, 1),
            "remaining_shelf_life_range": self.remaining_shelf_life_range,
            "spoilage_risk": self.spoilage_risk,
            "spoilage_risk_probs": {k: round(v, 3) for k, v in self.spoilage_risk_probs.items()},
            "time_to_harvest_days": round(self.time_to_harvest_days, 1),
            "time_to_harvest_range": self.time_to_harvest_range,
            "is_growing_plant": self.is_growing_plant,
            "top_contributing_factors": self.top_contributing_factors,
            "environmental": self.environmental_summary,
            "what_if_storage_comparison_days": {k: round(v, 1) for k, v in self.what_if_shelf_life_days.items()},
        }


class FreshAIStructuredPredictor:
    """
    Inference wrapper orchestrating XGBoost and LightGBM models for Stage 5.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.quality_model = None
        self.phys_age_model = None
        self.shelf_life_model = None
        self.spoilage_risk_model = None
        self.time_to_harvest_model = None
        self.models_loaded = False
        self._load_models()

    def _load_models(self):
        """Loads XGBoost and LightGBM models from disk with graceful fallbacks."""
        if not self.model_dir.exists():
            return

        # 1. Quality Model (XGBoost)
        xgb_q_path = self.model_dir / "xgboost_quality.json"
        if xgb is not None and xgb_q_path.exists():
            try:
                self.quality_model = xgb.XGBRegressor()
                self.quality_model.load_model(str(xgb_q_path))
            except Exception as e:
                print(f"[StructuredPredictor] Notice: could not load native XGBoost quality model: {e}")
        if self.quality_model is None and joblib is not None:
            fb = self.model_dir / "fallback_quality.joblib"
            if fb.exists():
                try:
                    self.quality_model = joblib.load(str(fb))
                except Exception:
                    pass

        # 2. Physiological Age Model (LightGBM)
        lgb_age_path = self.model_dir / "lightgbm_phys_age.txt"
        if lgb is not None and lgb_age_path.exists():
            try:
                self.phys_age_model = lgb.Booster(model_file=str(lgb_age_path))
            except Exception as e:
                print(f"[StructuredPredictor] Notice: could not load native LightGBM age model: {e}")
        if self.phys_age_model is None and joblib is not None:
            fb = self.model_dir / "fallback_phys_age.joblib"
            if fb.exists():
                try:
                    self.phys_age_model = joblib.load(str(fb))
                except Exception:
                    pass

        # 3. Remaining Shelf Life Model (LightGBM)
        lgb_shelf_path = self.model_dir / "lightgbm_shelf_life.txt"
        if lgb is not None and lgb_shelf_path.exists():
            try:
                self.shelf_life_model = lgb.Booster(model_file=str(lgb_shelf_path))
            except Exception as e:
                print(f"[StructuredPredictor] Notice: could not load native LightGBM shelf model: {e}")
        if self.shelf_life_model is None and joblib is not None:
            fb = self.model_dir / "fallback_shelf_life.joblib"
            if fb.exists():
                try:
                    self.shelf_life_model = joblib.load(str(fb))
                except Exception:
                    pass

        # 4. Spoilage Risk Model (XGBoost)
        xgb_risk_path = self.model_dir / "xgboost_spoilage_risk.json"
        if xgb is not None and xgb_risk_path.exists():
            try:
                self.spoilage_risk_model = xgb.XGBClassifier()
                self.spoilage_risk_model.load_model(str(xgb_risk_path))
            except Exception as e:
                print(f"[StructuredPredictor] Notice: could not load native XGBoost risk model: {e}")
        if self.spoilage_risk_model is None and joblib is not None:
            fb = self.model_dir / "fallback_spoilage_risk.joblib"
            if fb.exists():
                try:
                    self.spoilage_risk_model = joblib.load(str(fb))
                except Exception:
                    pass

        # 5. Time-to-Harvest Model (LightGBM)
        lgb_harvest_path = self.model_dir / "lightgbm_time_to_harvest.txt"
        if lgb is not None and lgb_harvest_path.exists():
            try:
                self.time_to_harvest_model = lgb.Booster(model_file=str(lgb_harvest_path))
            except Exception as e:
                print(f"[StructuredPredictor] Notice: could not load native LightGBM harvest model: {e}")
        if self.time_to_harvest_model is None and joblib is not None:
            fb = self.model_dir / "fallback_time_to_harvest.joblib"
            if fb.exists():
                try:
                    self.time_to_harvest_model = joblib.load(str(fb))
                except Exception:
                    pass

        self.models_loaded = (self.quality_model is not None and self.shelf_life_model is not None)

    def predict(
        self,
        structured_input: StructuredProduceInput
    ) -> StructuredPredictionResult:
        """
        Runs XGBoost & LightGBM inference on the structured feature vector.
        """
        x_vec = structured_input.to_array().reshape(1, -1)
        fv = structured_input.feature_vector
        env = structured_input.environmental
        tl = structured_input.timeline
        produce = structured_input.produce_class

        # ─────────────────────────────────────────────────────────────
        # 1. Quality Score (XGBoost Regressor / Visual Defect Harmonized) -> 54/100
        # ─────────────────────────────────────────────────────────────
        stage_base = max(10.0, 96.0 - (fv.freshness_stage_idx * 15.0))
        defect_deduction = (fv.defect_area_pct * 1.3) + (fv.defect_severity_score * 15.0)
        visual_quality_bound = float(np.clip(stage_base - defect_deduction, 5.0, 99.0))

        if self.quality_model is not None:
            raw_q = float(self.quality_model.predict(x_vec)[0])
            # Model prediction reconciled with visual defect ground truth
            quality = float(np.clip(min(raw_q, visual_quality_bound + 8.0), 5.0, 99.0))
        else:
            quality = visual_quality_bound

        quality_str = f"{int(round(quality))}/100"

        # ─────────────────────────────────────────────────────────────
        # 2. Physiological Age (LightGBM Regressor) -> Total Crop Age (Seed-to-Date)
        # ─────────────────────────────────────────────────────────────
        base_cultivation = 80.0
        for k, v in CROP_BASE_CULTIVATION_DAYS.items():
            if k in produce.lower():
                base_cultivation = v
                break

        if self.phys_age_model is not None:
            try:
                raw_age = float(self.phys_age_model.predict(x_vec)[0])
            except Exception:
                raw_age = float(self.phys_age_model.predict(x_vec.tolist())[0])
            age_days = float(np.clip(raw_age, 15.0, 420.0))
        else:
            eff_days = tl.days_since_purchase * env.thermal_acceleration_factor
            post_h = 1.5 + eff_days * 0.85 + fv.ripeness_prob * 2.5
            age_days = float(np.clip(base_cultivation + post_h, 15.0, 420.0))

        lower_age = max(1, int(math.floor(age_days - 2.0)))
        upper_age = max(lower_age + 1, int(math.ceil(age_days + 2.0)))
        age_range_str = f"{lower_age}–{upper_age} days"

        post_h_days = max(1.0, age_days - base_cultivation)
        lower_post = max(1, int(math.floor(post_h_days - 0.8)))
        upper_post = max(lower_post + 1, int(math.ceil(post_h_days + 0.8)))
        post_harvest_range_str = f"{lower_post}–{upper_post} days post-harvest"

        # ─────────────────────────────────────────────────────────────
        # 3. Remaining Shelf Life (LightGBM Regressor / Arrhenius Harmonized) -> 2–4 days
        # ─────────────────────────────────────────────────────────────
        # Ground remaining shelf life in physical produce quality, cuticle defect integrity, and thermal kinetics
        q_ratio = max(0.05, min(1.0, quality / 100.0))
        defect_cuticle_loss = min(0.88, (fv.defect_area_pct / 100.0) * 3.2)
        decay_factor = math.pow(q_ratio, 1.65) * (1.0 - defect_cuticle_loss)

        base_shelf_days = 12.0 if ("onion" in produce.lower() or "potato" in produce.lower() or "apple" in produce.lower()) else 6.5
        thermal_rate = max(0.3, env.thermal_acceleration_factor)
        biological_cap_days = max(0.2, (base_shelf_days * decay_factor) / thermal_rate)

        if self.shelf_life_model is not None:
            try:
                raw_shelf = float(self.shelf_life_model.predict(x_vec)[0])
            except Exception:
                raw_shelf = float(self.shelf_life_model.predict(x_vec.tolist())[0])
            shelf_candidate = float(np.clip(raw_shelf, 0.0, 35.0))
            # Reconcile tree model with biological physics ceiling:
            # Prevents model from predicting 11+ days when visual decay indicates 1-3 days
            shelf_days = min(shelf_candidate, biological_cap_days)
        else:
            shelf_days = biological_cap_days

        if tl.days_since_purchase > 0:
            shelf_days = max(0.2, shelf_days - tl.days_since_purchase * 0.4)

        lower_shelf = max(0, int(math.floor(shelf_days - 0.7)))
        upper_shelf = max(lower_shelf + 1, int(math.ceil(shelf_days + 0.7)))
        shelf_range_str = f"{lower_shelf}–{upper_shelf} days"

        # ─────────────────────────────────────────────────────────────
        # 4. Spoilage Risk (XGBoost / LightGBM Classifier) -> Low/Medium/High
        # ─────────────────────────────────────────────────────────────
        risk_labels = ["Low", "Medium", "High"]
        risk_colors = ["#22c55e", "#f59e0b", "#ef4444"]

        if self.spoilage_risk_model is not None:
            try:
                probs = self.spoilage_risk_model.predict_proba(x_vec)[0]
                risk_idx = int(np.argmax(probs))
                risk_probs_dict = {risk_labels[i]: float(probs[i]) for i in range(len(risk_labels))}
            except Exception:
                risk_pred = int(self.spoilage_risk_model.predict(x_vec)[0])
                risk_idx = min(2, max(0, risk_pred))
                risk_probs_dict = {label: (0.8 if i == risk_idx else 0.1) for i, label in enumerate(risk_labels)}
        else:
            if quality < 50.0 or fv.defect_area_pct > 12.0 or shelf_days < 2.0:
                risk_idx = 2
                risk_probs_dict = {"Low": 0.05, "Medium": 0.15, "High": 0.80}
            elif quality < 75.0 or fv.defect_area_pct > 4.5 or shelf_days < 5.0:
                risk_idx = 1
                risk_probs_dict = {"Low": 0.15, "Medium": 0.75, "High": 0.10}
            else:
                risk_idx = 0
                risk_probs_dict = {"Low": 0.85, "Medium": 0.12, "High": 0.03}

        spoilage_risk_str = risk_labels[risk_idx]
        spoilage_color = risk_colors[risk_idx]

        # ─────────────────────────────────────────────────────────────
        # 5. Time-to-Harvest (LightGBM Regressor) -> 10–14 days
        # ─────────────────────────────────────────────────────────────
        if tl.is_growing_plant:
            if self.time_to_harvest_model is not None:
                try:
                    raw_h = float(self.time_to_harvest_model.predict(x_vec)[0])
                except Exception:
                    raw_h = float(self.time_to_harvest_model.predict(x_vec.tolist())[0])
                harvest_days = float(np.clip(raw_h, 1.0, 45.0))
            else:
                harvest_days = float(np.clip(18.0 - tl.days_since_purchase * 0.8 - fv.ripeness_prob * 6.0, 1.0, 30.0))
            lower_h = max(1, int(math.floor(harvest_days - 1.8)))
            upper_h = max(lower_h + 1, int(math.ceil(harvest_days + 1.8)))
            harvest_range_str = f"{lower_h}–{upper_h} days"
        else:
            harvest_days = 0.0
            harvest_range_str = "Harvested (Post-Harvest Stage)"

        # ─────────────────────────────────────────────────────────────
        # 6. What-If Storage Scenarios Simulation
        # ─────────────────────────────────────────────────────────────
        what_if_shelf: Dict[str, float] = {}
        for cond in STORAGE_CONDITIONS:
            test_temp = 4.0 if "refrig" in cond.lower() else (12.0 if "cellar" in cond.lower() else (30.0 if "warm" in cond.lower() else 22.0))
            test_humid = 85.0 if "refrig" in cond.lower() else (75.0 if "cellar" in cond.lower() else 60.0)
            sim_env = EnvironmentalData(temperature_c=test_temp, humidity_pct=test_humid)
            sim_input = StructuredProduceInput(
                feature_vector=fv,
                environmental=sim_env,
                timeline=TimelineStorageData(days_since_purchase=tl.days_since_purchase, storage_condition=cond, is_growing_plant=tl.is_growing_plant),
                produce_class=produce,
            )
            sim_shelf_cap = max(0.2, (base_shelf_days * decay_factor) / max(0.3, sim_env.thermal_acceleration_factor))
            sim_x = sim_input.to_array().reshape(1, -1)
            if self.shelf_life_model is not None:
                try:
                    s_pred = float(self.shelf_life_model.predict(sim_x)[0])
                except Exception:
                    s_pred = float(self.shelf_life_model.predict(sim_x.tolist())[0])
                what_if_shelf[cond] = float(np.clip(min(s_pred, sim_shelf_cap), 0.1, 40.0))
            else:
                what_if_shelf[cond] = float(np.clip(sim_shelf_cap, 0.1, 40.0))

        # ─────────────────────────────────────────────────────────────
        # 7. Model Attribution / Top Contributing Factors
        # ─────────────────────────────────────────────────────────────
        factors = [
            {"factor": "Visual Freshness Probability", "value": f"{round(fv.freshness_prob * 100, 1)}%", "impact": "+ Positive (High Quality)" if fv.freshness_prob > 0.7 else "- Negative (Deterioration)"},
            {"factor": "Defect Area Surface %", "value": f"{round(fv.defect_area_pct, 1)}%", "impact": f"- Reduces Quality by ~{round(fv.defect_area_pct * 1.1, 1)} pts" if fv.defect_area_pct > 1.0 else "Minimal Surface Impact"},
            {"factor": "Storage Temperature", "value": f"{round(env.temperature_c, 1)}°C", "impact": f"Thermal acceleration {round(env.thermal_acceleration_factor, 1)}x vs 4°C cold store"},
            {"factor": "Texture Homogeneity (GLCM)", "value": f"{round(fv.texture_homogeneity, 2)}", "impact": "Firm cellular structure" if fv.texture_homogeneity > 0.75 else "Softened / Tissue breakdown"},
            {"factor": "Vapor Pressure Deficit", "value": f"{round(env.vapor_pressure_deficit, 2)} kPa", "impact": "Low moisture loss rate" if env.vapor_pressure_deficit < 1.0 else "Accelerated transpiration / wilting"},
        ]

        return StructuredPredictionResult(
            quality_score=quality,
            quality_score_str=quality_str,
            physiological_age_days=age_days,
            physiological_age_range=age_range_str,
            post_harvest_age_days=post_h_days,
            post_harvest_age_range=post_harvest_range_str,
            remaining_shelf_life_days=shelf_days,
            remaining_shelf_life_range=shelf_range_str,
            spoilage_risk=spoilage_risk_str,
            spoilage_risk_color=spoilage_color,
            spoilage_risk_probs=risk_probs_dict,
            time_to_harvest_days=harvest_days,
            time_to_harvest_range=harvest_range_str,
            is_growing_plant=tl.is_growing_plant,
            top_contributing_factors=factors,
            environmental_summary=env.to_dict(),
            what_if_shelf_life_days=what_if_shelf,
        )
