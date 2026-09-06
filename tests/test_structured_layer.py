"""
Unit Tests for Stage 5: Structured Prediction Layer (XGBoost & LightGBM)
"""

import unittest
import numpy as np

from freshai.fusion import ProduceFeatureVector
from freshai.structured_features import (
    STRUCTURED_FEATURE_NAMES,
    EnvironmentalData,
    TimelineStorageData,
    StructuredProduceInput,
)
from freshai.structured_predictor import (
    FreshAIStructuredPredictor,
    StructuredPredictionResult,
)


class TestStructuredLayer(unittest.TestCase):

    def setUp(self):
        self.feature_vector = ProduceFeatureVector(
            freshness_prob=0.82,
            ripeness_prob=0.91,
            defect_area_pct=7.4,
            avg_hue=12.4,
            avg_saturation=82.6,
            texture_contrast=0.42,
            texture_homogeneity=0.81,
        )
        self.env = EnvironmentalData(temperature_c=22.0, humidity_pct=60.0)
        self.timeline = TimelineStorageData(days_since_purchase=2.0, storage_condition="Pantry / Room Temperature")
        self.predictor = FreshAIStructuredPredictor()

    def test_environmental_calculations(self):
        # 4°C cold store should have thermal acceleration ~ 1.0
        cold_env = EnvironmentalData(temperature_c=4.0, humidity_pct=85.0)
        self.assertAlmostEqual(cold_env.thermal_acceleration_factor, 1.0, places=2)

        # 24°C room temp should have acceleration > 2.0x
        warm_env = EnvironmentalData(temperature_c=24.0, humidity_pct=50.0)
        self.assertGreater(warm_env.thermal_acceleration_factor, 2.0)
        self.assertGreater(warm_env.vapor_pressure_deficit, 0.0)

    def test_feature_vector_dimensions(self):
        inp = StructuredProduceInput(
            feature_vector=self.feature_vector,
            environmental=self.env,
            timeline=self.timeline,
            produce_class="Tomato",
        )
        arr = inp.to_array()
        self.assertEqual(len(arr), len(STRUCTURED_FEATURE_NAMES))
        self.assertEqual(arr.dtype, np.float32)

    def test_structured_prediction_ranges(self):
        inp = StructuredProduceInput(
            feature_vector=self.feature_vector,
            environmental=self.env,
            timeline=self.timeline,
            produce_class="Tomato",
        )
        result = self.predictor.predict(inp)

        self.assertIsInstance(result, StructuredPredictionResult)
        self.assertGreaterEqual(result.quality_score, 0.0)
        self.assertLessEqual(result.quality_score, 100.0)
        self.assertTrue(result.quality_score_str.endswith("/100"))

        self.assertGreater(result.physiological_age_days, 0.0)
        self.assertIn("–", result.physiological_age_range)

        self.assertGreaterEqual(result.remaining_shelf_life_days, 0.0)
        self.assertIn("–", result.remaining_shelf_life_range)

        self.assertIn(result.spoilage_risk, ["Low", "Medium", "High"])
        self.assertGreater(len(result.top_contributing_factors), 0)

    def test_growing_crop_time_to_harvest(self):
        crop_timeline = TimelineStorageData(
            days_since_purchase=5.0,
            storage_condition="Pantry / Room Temperature",
            is_growing_plant=True,
        )
        inp = StructuredProduceInput(
            feature_vector=self.feature_vector,
            environmental=self.env,
            timeline=crop_timeline,
            produce_class="Tomato",
        )
        result = self.predictor.predict(inp)
        self.assertTrue(result.is_growing_plant)
        self.assertGreater(result.time_to_harvest_days, 0.0)
        self.assertIn("days", result.time_to_harvest_range)

    def test_what_if_storage_comparison(self):
        inp = StructuredProduceInput(
            feature_vector=self.feature_vector,
            environmental=self.env,
            timeline=self.timeline,
            produce_class="Apple",
        )
        result = self.predictor.predict(inp)
        sc = result.what_if_shelf_life_days

        # Cold storage must preserve produce longer than warm sunlight
        self.assertIn("Refrigerated", sc)
        self.assertIn("Warm / Direct Sunlight", sc)
        self.assertGreater(sc["Refrigerated"], sc["Warm / Direct Sunlight"])


if __name__ == "__main__":
    unittest.main()
