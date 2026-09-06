import unittest
import numpy as np
from PIL import Image, ImageDraw
from freshai.freshness_detector import ConvNeXtFreshnessDetector
from freshai.defect_detector import FreshAIDefectDetector
from freshai.feature_extractor import ProduceFeatureExtractor
from freshai.fusion import assemble_feature_vector, ProduceFeatureVector
from freshai.fusion_predictor import FreshAIFusionPredictor, MultimodalFreshnessAssessment

class TestFusionPipeline(unittest.TestCase):
    def setUp(self):
        self.convnext = ConvNeXtFreshnessDetector()
        self.defect = FreshAIDefectDetector()
        self.extractor = ProduceFeatureExtractor()
        self.fusion = FreshAIFusionPredictor()

    def test_end_to_end_fusion(self):
        # Create a sample produce crop
        img = Image.new("RGB", (120, 120), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.ellipse([10, 10, 110, 110], fill=(220, 38, 38))
        # Add a defect spot
        draw.ellipse([40, 40, 60, 60], fill=(30, 20, 10))

        fr = self.convnext.predict(img, produce_class="Tomato")
        dr = self.defect.predict(img, produce_class="Tomato")
        fe = self.extractor.extract(img)

        fv = assemble_feature_vector(fr, dr, fe, produce_class="Tomato")
        self.assertIsInstance(fv, ProduceFeatureVector)
        dense = fv.to_dense_vector(include_deep=False)
        self.assertGreater(len(dense), 40)

        assessment = self.fusion.predict(fv)
        self.assertIsInstance(assessment, MultimodalFreshnessAssessment)
        self.assertIn(assessment.freshness_stage, ["Very Fresh", "Fresh", "Ripe", "Overripe", "Deteriorating", "Spoiled"])
        self.assertGreaterEqual(assessment.freshness_score, 0)
        self.assertLessEqual(assessment.freshness_score, 100)

if __name__ == "__main__":
    unittest.main()
