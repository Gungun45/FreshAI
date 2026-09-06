"""
Unit tests for FreshAI:
1. Lifespan Improvement (Storage recommendations, temp/humidity, handling)
2. Produce Nutrition (USDA/WHO trusted data)
3. Result Explanation (Plain language ML diagnostics)
4. Detector Color Space Channel Verification
"""

import os
import unittest
import numpy as np
from PIL import Image

from freshai.rag_engine import FreshAIRAGEngine, PRODUCE_KNOWLEDGE_BASE, explain_predictions
from freshai.detector import FreshAIDetector


class TestNutritionStorageAndExplanation(unittest.TestCase):

    def setUp(self):
        self.rag = FreshAIRAGEngine()

    def test_result_explanation_structure(self):
        """Verify plain-language explanation contains all 4 key ML prediction components."""
        exp = explain_predictions(
            produce_name="Tomato",
            quality_score=84.0,
            physiological_age_days=6.5,
            remaining_shelf_life_days=3.0,
            spoilage_risk="Low",
            defect_area_pct=2.0,
            storage_temp_c=22.0,
        )
        self.assertIn("quality_score_explanation", exp)
        self.assertIn("estimated_age_explanation", exp)
        self.assertIn("shelf_life_explanation", exp)
        self.assertIn("spoilage_risk_explanation", exp)
        self.assertIn("summary_statement", exp)

        # Content assertions
        self.assertIn("84", exp["quality_score_explanation"])
        self.assertIn("PEAK", exp["quality_score_explanation"])
        self.assertIn("6.5", exp["estimated_age_explanation"])
        self.assertIn("3.0", exp["shelf_life_explanation"])
        self.assertIn("LOW", exp["spoilage_risk_explanation"].upper())

    def test_produce_nutrition_trusted_sources(self):
        """Verify produce nutrition facts are grounded in USDA FoodData Central and WHO."""
        for produce in ["tomato", "apple", "banana", "onion", "watermelon", "orange", "potato"]:
            self.assertIn(produce, PRODUCE_KNOWLEDGE_BASE)
            kb = PRODUCE_KNOWLEDGE_BASE[produce]
            self.assertIn("nutrition", kb)
            nutr = kb["nutrition"]
            self.assertIn("calories_kcal", nutr)
            self.assertIn("dietary_fiber_g", nutr)
            self.assertIn("vitamins", nutr)
            self.assertIn("trusted_sources", nutr)
            # Must cite USDA or WHO or Harvard
            has_trusted = any(
                ("USDA" in src or "WHO" in src or "Harvard" in src)
                for src in nutr["trusted_sources"]
            )
            self.assertTrue(has_trusted, f"Produce {produce} lacks trusted scientific source attribution.")

    def test_lifespan_improvement_guidance(self):
        """Verify storage temperature, humidity, location, and handling recommendations."""
        for produce in ["tomato", "apple", "banana", "onion", "potato"]:
            rec = self.rag.generate_recommendations(
                produce_class=produce,
                quality_score=80.0,
                physiological_age_days=4.0,
                remaining_shelf_life_days=5.0,
                defect_area_pct=1.0,
                spoilage_risk="Low",
                storage_temp_c=22.0,
            )
            strat = rec.storage_strategy
            self.assertIn("optimal_temperature_target", strat)
            self.assertIn("optimal_humidity_target", strat)
            self.assertIn("storage_location", strat)
            self.assertIn("handling_and_preservation", strat)
            self.assertGreater(len(strat["handling_and_preservation"]), 0)

    def test_detector_color_channel_consistency(self):
        """Verify detector handles PIL Image and numpy arrays consistently without color inversion."""
        # Using produce_detector-5 on sample_tomatoes.jpg
        weight_path = "runs/detect/runs/freshai_detect/produce_detector-5/weights/best.pt"
        if os.path.exists(weight_path) and os.path.exists("samples/sample_tomatoes.jpg"):
            detector = FreshAIDetector(model_path=weight_path)
            # 1. Using path
            r_path = detector.predict("samples/sample_tomatoes.jpg", conf_threshold=0.25)
            # 2. Using PIL Image
            pil_img = Image.open("samples/sample_tomatoes.jpg").convert("RGB")
            r_pil = detector.predict(pil_img, conf_threshold=0.25)
            # 3. Using RGB Numpy array
            np_rgb = np.array(pil_img)
            r_np = detector.predict(np_rgb, conf_threshold=0.25)

            # Both should detect tomatoes (>0 items)
            self.assertGreater(len(r_path.detected_objects), 0)
            self.assertGreater(len(r_pil.detected_objects), 0)
            self.assertGreater(len(r_np.detected_objects), 0)
            self.assertEqual(len(r_pil.detected_objects), len(r_path.detected_objects))

    def test_onion_watermelon_disambiguation(self):
        """Verify onion visual properties prevent false watermelon misclassifications."""
        from freshai.detector import disambiguate_produce_class
        import glob

        # 1. Test disambiguate_produce_class directly
        onion_files = glob.glob("datasets/fruits_360/train/Onion*/*.jpg")
        if onion_files:
            for p in onion_files[:5]:
                corrected = disambiguate_produce_class(Image.open(p), "Watermelon")
                self.assertEqual(corrected, "onion", f"Onion at {p} failed to disambiguate from Watermelon")

        # 2. Test that true watermelons remain watermelons
        wm_files = glob.glob("datasets/fruit_named/train/images/*semangka*")
        if wm_files:
            for p in wm_files[:3]:
                result = disambiguate_produce_class(Image.open(p), "Watermelon")
                self.assertEqual(result, "Watermelon", f"True watermelon at {p} was falsely reclassified")

        # 3. Test universal model on Onion Red
        if os.path.exists("runs/detect/freshai_universal_produce_best.pt") and onion_files:
            d = FreshAIDetector("runs/detect/freshai_universal_produce_best.pt")
            res = d.predict(onion_files[0], conf_threshold=0.25)
            class_names = [o.class_name.lower() for o in res.detected_objects]
            self.assertIn("onion", class_names)
            self.assertNotIn("watermelon", class_names)


if __name__ == "__main__":
    unittest.main()
