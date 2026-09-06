import unittest
import numpy as np
from PIL import Image, ImageDraw
from freshai.defect_detector import FreshAIDefectDetector, DefectResult

class TestDefectDetection(unittest.TestCase):
    def setUp(self):
        self.detector = FreshAIDefectDetector()

    def test_clean_produce_has_minimal_defects(self):
        # Create clean solid red tomato crop
        img = Image.new("RGB", (200, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 180, 180], fill=(220, 38, 38))
        res = self.detector.predict(img, produce_class="Tomato")
        self.assertIsInstance(res, DefectResult)
        self.assertLess(res.total_defect_area_pct, 5.0)

    def test_defective_produce_detected_and_segmented(self):
        # Create produce with distinct dark necrotic rot lesion and mold patch
        img = Image.new("RGB", (200, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Red apple body
        draw.ellipse([20, 20, 180, 180], fill=(210, 30, 30))
        # Dark rotting patch
        draw.ellipse([60, 60, 110, 110], fill=(25, 15, 10))
        # Mold patch (pale grayish-white)
        draw.ellipse([120, 120, 150, 150], fill=(230, 230, 235))

        res = self.detector.predict(img, produce_class="Apple")
        self.assertIsInstance(res, DefectResult)
        self.assertGreater(res.total_defect_area_pct, 2.0)
        self.assertGreater(res.defect_count, 0)
        self.assertIsNotNone(res.composite_mask)
        self.assertIsNotNone(res.annotated_crop)
        self.assertEqual(res.annotated_crop.shape, (200, 200, 3))

if __name__ == "__main__":
    unittest.main()
