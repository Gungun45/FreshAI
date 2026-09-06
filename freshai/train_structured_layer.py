"""
FreshAI - Stage 5: Structured Prediction Layer Training
Trains XGBoost and LightGBM models on multimodal feature vectors + environmental + timeline inputs.
Predicts:
  1. Quality Score (XGBoost Regressor) -> e.g. 84/100
  2. Physiological Age (LightGBM Regressor) -> e.g. 5–8 days
  3. Remaining Shelf Life (LightGBM Regressor) -> e.g. 2–4 days
  4. Spoilage Risk (XGBoost Classifier) -> Low / Medium / High
  5. Time-to-Harvest (LightGBM Regressor) -> e.g. 10–14 days
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

# Optional native packages with fallback
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
except ImportError:
    HistGradientBoostingRegressor = None
    HistGradientBoostingClassifier = None

from freshai.structured_features import (
    STRUCTURED_FEATURE_NAMES,
    SUPPORTED_PRODUCE_CLASSES,
    STORAGE_CONDITIONS,
    EnvironmentalData,
    TimelineStorageData,
    StructuredProduceInput,
)
from freshai.fusion import ProduceFeatureVector


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "runs" / "structured"

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



def generate_structured_training_data(
    num_samples: int = 2400,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generates a physiologically grounded dataset spanning all produce categories,
    freshness levels, defect severities, temperatures (0°C to 38°C), and timelines.
    """
    random.seed(seed)
    np.random.seed(seed)

    X_list = []
    y_quality = []
    y_phys_age = []
    y_shelf_life = []
    y_spoilage_risk = []
    y_time_to_harvest = []

    for i in range(num_samples):
        # 1. Produce category
        produce = random.choice(SUPPORTED_PRODUCE_CLASSES[:-1])  # avoid "Other" dominate
        is_plant = (produce in ["Tomato", "Watermelon", "Broccoli", "Carrot"]) and (random.random() < 0.15)

        # 2. Storage & Environmental conditions
        storage = random.choice(STORAGE_CONDITIONS)
        if storage == "Refrigerated":
            temp_c = float(np.clip(np.random.normal(4.0, 1.5), 1.0, 8.0))
            humid_pct = float(np.clip(np.random.normal(85.0, 5.0), 65.0, 95.0))
        elif storage == "Cool Cellar":
            temp_c = float(np.clip(np.random.normal(12.0, 2.0), 8.0, 16.0))
            humid_pct = float(np.clip(np.random.normal(75.0, 8.0), 55.0, 90.0))
        elif storage == "Warm / Direct Sunlight":
            temp_c = float(np.clip(np.random.normal(30.0, 3.5), 24.0, 40.0))
            humid_pct = float(np.clip(np.random.normal(45.0, 12.0), 20.0, 80.0))
        else:  # Pantry / Room Temperature
            temp_c = float(np.clip(np.random.normal(22.0, 2.5), 16.0, 26.0))
            humid_pct = float(np.clip(np.random.normal(60.0, 10.0), 35.0, 85.0))

        env = EnvironmentalData(temperature_c=temp_c, humidity_pct=humid_pct)

        # 3. Timeline
        if is_plant:
            days_elapsed = float(np.random.uniform(0.0, 14.0))
        else:
            days_elapsed = float(np.random.exponential(scale=3.5))
        tl = TimelineStorageData(
            days_since_purchase=days_elapsed,
            storage_condition=storage,
            is_growing_plant=is_plant,
        )

        # 4. Visual degradation state (correlated with elapsed time & temperature)
        effective_days = days_elapsed * env.thermal_acceleration_factor

        # Latent degradation index: 0.0 (fresh harvest) to 1.0+ (completely spoiled)
        degrade_idx = float(np.clip(
            effective_days / 14.0 + np.random.normal(0.0, 0.12),
            0.0, 1.5
        ))

        # Freshness probability decreases with degradation
        freshness_prob = float(np.clip(1.0 - 0.75 * degrade_idx + np.random.normal(0.0, 0.05), 0.05, 0.99))
        ripeness_prob = float(np.clip(0.4 + 0.6 * degrade_idx - 0.3 * max(0.0, degrade_idx - 1.0), 0.1, 0.99))

        # Defect area % increases with degradation
        if degrade_idx < 0.3:
            defect_area_pct = float(np.clip(np.random.exponential(scale=1.5), 0.0, 6.0))
        elif degrade_idx < 0.7:
            defect_area_pct = float(np.clip(np.random.normal(6.5, 3.0), 1.0, 18.0))
        else:
            defect_area_pct = float(np.clip(np.random.normal(22.0, 8.0), 8.0, 65.0))

        defect_count = int(max(0, np.round(defect_area_pct / 3.0 + np.random.normal(0, 1))))
        defect_severity = float(np.clip(defect_area_pct / 20.0 + (0.3 if defect_count > 3 else 0.0), 0.0, 1.0))

        # Color & Texture correlates
        avg_hue = float(np.clip(np.random.normal(25.0 - 15.0 * degrade_idx, 8.0), 0.0, 180.0))
        avg_sat = float(np.clip(np.random.normal(85.0 - 20.0 * degrade_idx, 10.0), 10.0, 100.0))
        avg_val = float(np.clip(np.random.normal(80.0 - 25.0 * degrade_idx, 12.0), 10.0, 100.0))

        green_ratio = float(np.clip(0.6 * max(0.0, 0.6 - degrade_idx) + np.random.normal(0, 0.05), 0.0, 1.0))
        yellow_ratio = float(np.clip(0.7 * (1.0 - abs(degrade_idx - 0.5)) + np.random.normal(0, 0.05), 0.0, 1.0))
        red_ratio = float(np.clip(0.6 * (1.0 - abs(degrade_idx - 0.6)) + np.random.normal(0, 0.05), 0.0, 1.0))
        brown_ratio = float(np.clip(0.8 * max(0.0, degrade_idx - 0.5) + np.random.normal(0, 0.05), 0.0, 1.0))
        dark_spot_ratio = float(np.clip(defect_area_pct / 100.0 * 0.7, 0.0, 1.0))
        mold_ratio = float(np.clip(max(0.0, degrade_idx - 0.8) * 0.4, 0.0, 1.0))

        # Texture metrics
        texture_contrast = float(np.clip(0.25 + 0.5 * degrade_idx + defect_area_pct * 0.015, 0.05, 1.5))
        texture_homogeneity = float(np.clip(0.92 - 0.35 * degrade_idx - defect_area_pct * 0.01, 0.2, 0.98))
        texture_dissimilarity = float(np.clip(texture_contrast * 0.75, 0.05, 1.2))
        texture_energy = float(np.clip(0.45 - 0.25 * degrade_idx, 0.05, 0.8))
        texture_corr = float(np.clip(0.85 - 0.3 * degrade_idx, 0.1, 0.98))
        texture_entropy = float(np.clip(1.2 + 1.8 * degrade_idx, 0.5, 4.5))
        surface_roughness = float(np.clip(100.0 + 350.0 * degrade_idx, 20.0, 900.0))
        edge_density = float(np.clip(0.02 + 0.15 * degrade_idx, 0.01, 0.5))
        color_entropy = float(np.clip(1.5 + 1.2 * degrade_idx, 0.5, 3.8))

        fv = ProduceFeatureVector(
            freshness_prob=freshness_prob,
            ripeness_prob=ripeness_prob,
            defect_area_pct=defect_area_pct,
            defect_count=defect_count,
            defect_severity_score=defect_severity,
            avg_hue=avg_hue,
            avg_saturation=avg_sat,
            avg_brightness=avg_val,
            mean_l=avg_val * 0.9,
            mean_a=15.0 if "tomato" in produce.lower() or "apple" in produce.lower() else -10.0,
            mean_b_lab=25.0,
            chroma=30.0,
            green_region_pct=green_ratio * 100.0,
            yellow_region_pct=yellow_ratio * 100.0,
            red_region_pct=red_ratio * 100.0,
            brown_region_pct=brown_ratio * 100.0,
            dark_decay_pct=dark_spot_ratio * 100.0,
            pale_mold_pct=mold_ratio * 100.0,
            texture_contrast=texture_contrast,
            texture_dissimilarity=texture_dissimilarity,
            texture_homogeneity=texture_homogeneity,
            texture_energy=texture_energy,
            texture_correlation=texture_corr,
            texture_entropy=texture_entropy,
            surface_roughness=surface_roughness,
            edge_density=edge_density,
        )

        inp = StructuredProduceInput(
            feature_vector=fv,
            environmental=env,
            timeline=tl,
            produce_class=produce,
        )
        x_vec = inp.to_array()
        X_list.append(x_vec)

        # ── Ground Truth Targets ──────────────────────────────────────────
        # Target 1: Quality Score (0 to 100) -> e.g. 84/100 for fresh produce with minor defect
        texture_degrade_penalty = (
            max(0.0, (0.75 - texture_homogeneity) * 18.0)
            + max(0.0, (texture_contrast - 0.7) * 8.0)
            + max(0.0, (surface_roughness - 300.0) / 70.0)
            + max(0.0, (edge_density - 0.12) * 25.0)
        )
        base_quality = 96.0 - 38.0 * degrade_idx - 1.2 * defect_area_pct - 14.0 * brown_ratio - texture_degrade_penalty
        quality_score = float(np.clip(base_quality + np.random.normal(0, 2.0), 5.0, 99.0))
        y_quality.append(quality_score)

        # Target 2: Physiological Crop Age in days (Seed/Transplant to current stage)
        base_growth = CROP_BASE_CULTIVATION_DAYS.get(produce.lower(), 80.0)
        if is_plant:
            phys_age = float(np.clip(
                base_growth * (0.35 + 0.60 * ripeness_prob) + np.random.normal(0, 1.5),
                15.0,
                base_growth + 4.0
            ))
        else:
            post_harvest_days = float(np.clip(1.5 + effective_days * 0.85 + (ripeness_prob * 2.5) + np.random.normal(0, 0.4), 1.0, 30.0))
            phys_age = float(np.clip(
                base_growth * (0.95 + 0.06 * ripeness_prob) + post_harvest_days,
                base_growth * 0.80,
                base_growth + 38.0
            ))
        y_phys_age.append(phys_age)

        # Target 3: Remaining Shelf Life in days (e.g. 2–4 days at ambient)
        # Max shelf life under ideal fridge conditions ~ 14 to 28 days depending on produce
        base_shelf_days = 20.0 if "apple" in produce.lower() or "onion" in produce.lower() else 10.0
        remaining_days = (base_shelf_days - effective_days) / env.thermal_acceleration_factor
        remaining_days -= (defect_area_pct * 0.18)
        remaining_shelf = float(np.clip(remaining_days + np.random.normal(0, 0.3), 0.0, 30.0))
        y_shelf_life.append(remaining_shelf)

        # Target 4: Spoilage Risk (0: Low, 1: Medium, 2: High)
        if quality_score < 48.0 or defect_area_pct > 14.0 or remaining_shelf < 2.0 or mold_ratio > 0.05:
            risk = 2  # High
        elif quality_score < 74.0 or defect_area_pct > 4.5 or remaining_shelf < 5.0:
            risk = 1  # Medium
        else:
            risk = 0  # Low
        y_spoilage_risk.append(risk)

        # Target 5: Time-to-Harvest in days (e.g. 10–14 days for growing plants)
        if is_plant:
            harvest_days = float(np.clip(18.0 - days_elapsed * 0.8 - ripeness_prob * 6.0, 1.0, 35.0))
        else:
            # For already harvested produce, time to harvest is 0.0
            harvest_days = 0.0
        y_time_to_harvest.append(harvest_days)

    X = np.array(X_list, dtype=np.float32)
    y_dict = {
        "quality_score": np.array(y_quality, dtype=np.float32),
        "physiological_age": np.array(y_phys_age, dtype=np.float32),
        "remaining_shelf_life": np.array(y_shelf_life, dtype=np.float32),
        "spoilage_risk": np.array(y_spoilage_risk, dtype=np.int32),
        "time_to_harvest": np.array(y_time_to_harvest, dtype=np.float32),
    }
    return X, y_dict


def train_structured_models(
    num_samples: int = 2400,
    output_dir: Path = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Trains XGBoost & LightGBM models on the structured produce dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  FreshAI - Stage 5: Structured Prediction Layer Training")
    print(f"  Generating {num_samples} structured multi-factor training samples...")
    X, y_dict = generate_structured_training_data(num_samples=num_samples)
    print(f"  Feature Matrix Shape: {X.shape} ({len(STRUCTURED_FEATURE_NAMES)} features)")
    print("=" * 65)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    report: Dict[str, Any] = {"models": {}}

    # ─────────────────────────────────────────────────────────────────
    # 1. Quality Score Model (XGBoost Regressor)
    # ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Training Quality Score Model (XGBoost Regressor)...")
    y_tr = y_dict["quality_score"][train_idx]
    y_te = y_dict["quality_score"][test_idx]

    quality_model = None
    if xgb is not None:
        try:
            quality_model = xgb.XGBRegressor(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=2,
            )
            quality_model.fit(X_train, y_tr)
            model_file = output_dir / "xgboost_quality.json"
            quality_model.save_model(str(model_file))
            print(f"  Saved XGBoost model to: {model_file.name}")
        except Exception as e:
            print(f"  XGBoost native training failed: {e}. Using HistGradientBoostingRegressor.")
            quality_model = None

    if quality_model is None:
        quality_model = HistGradientBoostingRegressor(max_iter=120, max_depth=5, random_state=42)
        quality_model.fit(X_train, y_tr)
        if joblib is not None:
            joblib.dump(quality_model, str(output_dir / "fallback_quality.joblib"))

    preds = quality_model.predict(X_test)
    mse = float(np.mean((preds - y_te) ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    ss_res = float(np.sum((y_te - preds) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    print(f"  Quality Score Model Evaluated: RMSE={rmse:.2f} | R²={r2:.4f}")
    report["models"]["quality_score"] = {"rmse": round(rmse, 2), "r2": round(r2, 4), "engine": "XGBoost" if xgb else "HistGradientBoosting"}

    # ─────────────────────────────────────────────────────────────────
    # 2. Physiological Age Model (LightGBM Regressor)
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Training Physiological Age Model (LightGBM Regressor)...")
    y_tr = y_dict["physiological_age"][train_idx]
    y_te = y_dict["physiological_age"][test_idx]

    age_model = None
    if lgb is not None:
        try:
            age_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.08,
                num_leaves=24,
                random_state=42,
                n_jobs=2,
                verbose=-1,
            )
            age_model.fit(X_train, y_tr)
            model_file = output_dir / "lightgbm_phys_age.txt"
            age_model.booster_.save_model(str(model_file))
            print(f"  Saved LightGBM model to: {model_file.name}")
        except Exception as e:
            print(f"  LightGBM native training failed: {e}. Using HistGradientBoostingRegressor.")
            age_model = None

    if age_model is None:
        age_model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
        age_model.fit(X_train, y_tr)
        if joblib is not None:
            joblib.dump(age_model, str(output_dir / "fallback_phys_age.joblib"))

    preds = age_model.predict(X_test)
    rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    ss_res = float(np.sum((y_te - preds) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    print(f"  Physiological Age Model Evaluated: RMSE={rmse:.2f} days | R²={r2:.4f}")
    report["models"]["physiological_age"] = {"rmse_days": round(rmse, 2), "r2": round(r2, 4), "engine": "LightGBM" if lgb else "HistGradientBoosting"}

    # ─────────────────────────────────────────────────────────────────
    # 3. Remaining Shelf Life Model (LightGBM Regressor)
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Training Remaining Shelf Life Model (LightGBM Regressor)...")
    y_tr = y_dict["remaining_shelf_life"][train_idx]
    y_te = y_dict["remaining_shelf_life"][test_idx]

    shelf_model = None
    if lgb is not None:
        try:
            shelf_model = lgb.LGBMRegressor(
                n_estimators=110,
                max_depth=5,
                learning_rate=0.08,
                num_leaves=24,
                random_state=42,
                n_jobs=2,
                verbose=-1,
            )
            shelf_model.fit(X_train, y_tr)
            model_file = output_dir / "lightgbm_shelf_life.txt"
            shelf_model.booster_.save_model(str(model_file))
            print(f"  Saved LightGBM model to: {model_file.name}")
        except Exception as e:
            print(f"  LightGBM native training failed: {e}. Using HistGradientBoostingRegressor.")
            shelf_model = None

    if shelf_model is None:
        shelf_model = HistGradientBoostingRegressor(max_iter=110, max_depth=5, random_state=42)
        shelf_model.fit(X_train, y_tr)
        if joblib is not None:
            joblib.dump(shelf_model, str(output_dir / "fallback_shelf_life.joblib"))

    preds = shelf_model.predict(X_test)
    rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
    ss_tot = float(np.sum((y_te - np.mean(y_te)) ** 2))
    ss_res = float(np.sum((y_te - preds) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    print(f"  Remaining Shelf Life Model Evaluated: RMSE={rmse:.2f} days | R²={r2:.4f}")
    report["models"]["remaining_shelf_life"] = {"rmse_days": round(rmse, 2), "r2": round(r2, 4), "engine": "LightGBM" if lgb else "HistGradientBoosting"}

    # ─────────────────────────────────────────────────────────────────
    # 4. Spoilage Risk Classifier (XGBoost Classifier)
    # ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Training Spoilage Risk Classifier (XGBoost / LightGBM Classifier)...")
    y_tr = y_dict["spoilage_risk"][train_idx]
    y_te = y_dict["spoilage_risk"][test_idx]

    risk_model = None
    if xgb is not None:
        try:
            risk_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                n_jobs=2,
            )
            risk_model.fit(X_train, y_tr)
            model_file = output_dir / "xgboost_spoilage_risk.json"
            risk_model.save_model(str(model_file))
            print(f"  Saved XGBoost classifier to: {model_file.name}")
        except Exception as e:
            print(f"  XGBoost classifier failed: {e}. Using HistGradientBoostingClassifier.")
            risk_model = None

    if risk_model is None:
        risk_model = HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
        risk_model.fit(X_train, y_tr)
        if joblib is not None:
            joblib.dump(risk_model, str(output_dir / "fallback_spoilage_risk.joblib"))

    preds = risk_model.predict(X_test)
    acc = float(np.mean(preds == y_te))
    print(f"  Spoilage Risk Classifier Evaluated: Accuracy={acc * 100:.1f}%")
    report["models"]["spoilage_risk"] = {"accuracy": round(acc, 4), "engine": "XGBoost" if xgb else "HistGradientBoosting"}

    # ─────────────────────────────────────────────────────────────────
    # 5. Time-to-Harvest Model (LightGBM Regressor)
    # ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Training Time-to-Harvest Model (LightGBM Regressor)...")
    plant_mask = X_train[:, STRUCTURED_FEATURE_NAMES.index("is_growing_plant")] > 0.5
    if np.sum(plant_mask) > 30:
        X_tr_p = X_train[plant_mask]
        y_tr_p = y_dict["time_to_harvest"][train_idx][plant_mask]
    else:
        X_tr_p = X_train
        y_tr_p = y_dict["time_to_harvest"][train_idx]

    harvest_model = None
    if lgb is not None:
        try:
            harvest_model = lgb.LGBMRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                random_state=42,
                n_jobs=2,
                verbose=-1,
            )
            harvest_model.fit(X_tr_p, y_tr_p)
            model_file = output_dir / "lightgbm_time_to_harvest.txt"
            harvest_model.booster_.save_model(str(model_file))
            print(f"  Saved LightGBM model to: {model_file.name}")
        except Exception as e:
            print(f"  LightGBM harvest training failed: {e}. Using fallback.")
            harvest_model = None

    if harvest_model is None:
        harvest_model = HistGradientBoostingRegressor(max_iter=80, max_depth=4, random_state=42)
        harvest_model.fit(X_tr_p, y_tr_p)
        if joblib is not None:
            joblib.dump(harvest_model, str(output_dir / "fallback_time_to_harvest.joblib"))

    # Feature Importance Extraction (from Quality Model)
    importances: Dict[str, float] = {}
    if hasattr(quality_model, "feature_importances_"):
        fi = quality_model.feature_importances_
        for name, score in zip(STRUCTURED_FEATURE_NAMES, fi):
            importances[name] = round(float(score), 4)
    # Sort top 10
    top_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:10])
    report["top_feature_importances"] = top_importances

    # Save summary report
    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n  All 5 Structured Prediction Models Trained & Saved to: {output_dir}")
    print(f"  Top Predictive Features: {list(top_importances.keys())[:5]}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2400, help="Number of training samples")
    args = parser.parse_args()

    train_structured_models(num_samples=args.samples)
