"""
Unit tests for FreshAI Step 1 Detection Engine
"""

import os
import unittest
import numpy as np
from PIL import Image

from freshai.detector import FreshAIDetector, FreshAIDetectionResult
from freshai.dataset_utils import generate_synthetic_demo_dataset, generate_data_yaml


class TestFreshAIDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a test image with distinct shapes
        cls.test_img_path = "tests/test_produce.jpg"
        os.makedirs("tests", exist_ok=True)
        img = Image.new("RGB", (640, 640), color=(240, 240, 240))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        # Red tomato/apple shape
        draw.ellipse([150, 150, 300, 300], fill=(220, 30, 30))
        # Green plant shape
        draw.ellipse([350, 200, 500, 350], fill=(30, 180, 50))
        img.save(cls.test_img_path)

        # Initialize detector
        cls.detector = FreshAIDetector(model_path="yolov8n.pt", produce_only=False)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.model_path, "yolov8n.pt")

    def test_detector_prediction_output_structure(self):
        res = self.detector.predict(
            image=self.test_img_path,
            conf_threshold=0.1,
            iou_threshold=0.45,
            draw_annotations=True,
        )
        self.assertIsInstance(res, FreshAIDetectionResult)
        self.assertEqual(res.image_width, 640)
        self.assertEqual(res.image_height, 640)
        self.assertIsInstance(res.detected_objects, list)
        self.assertIsInstance(res.class_counts, dict)

        # Check JSON conversion
        res_dict = res.to_dict()
        self.assertIn("summary", res_dict)
        self.assertIn("detections", res_dict)
        self.assertIn("step_2_ready_payload", res_dict)

    def test_dataset_generator(self):
        demo_dir = "tests/demo_dataset"
        yaml_path = generate_synthetic_demo_dataset(
            output_dir=demo_dir,
            num_train=4,
            num_val=2,
        )
        self.assertTrue(os.path.exists(yaml_path))
        self.assertTrue(os.path.exists(os.path.join(demo_dir, "images", "train")))
        self.assertTrue(os.path.exists(os.path.join(demo_dir, "labels", "train")))

    def test_onion_detection_recall(self):
        if os.path.exists("scratch_onion.jpg") and os.path.exists("runs/detect/freshai_universal_produce_best.pt"):
            det = FreshAIDetector("runs/detect/freshai_universal_produce_best.pt")
            res = det.predict("scratch_onion.jpg", conf_threshold=0.20)
            self.assertGreater(len(res.detected_objects), 0, "Expected onion to be detected in scratch_onion.jpg")
            names = [o.class_name.lower() for o in res.detected_objects]
            self.assertIn("onion", names, "Expected 'onion' class in detected objects")
            onion_obj = next(o for o in res.detected_objects if o.class_name.lower() == "onion")
            self.assertGreaterEqual(onion_obj.confidence, 0.70, "Expected onion confidence >= 0.70")


if __name__ == "__main__":
    unittest.main()