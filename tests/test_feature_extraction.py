import unittest
import numpy as np
from PIL import Image, ImageDraw
from freshai.feature_extractor import ProduceFeatureExtractor, ProduceExtractedFeatures

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = ProduceFeatureExtractor()

    def test_color_extraction(self):
        # Create pure green vegetable crop
        img = Image.new("RGB", (150, 150), color=(34, 197, 94))
        feats = self.extractor.extract(img)
        self.assertIsInstance(feats, ProduceExtractedFeatures)
        # Green region should be dominant
        self.assertGreater(feats.color.green_region_pct, 40.0)
        self.assertGreater(feats.color.avg_saturation, 40.0)
        self.assertGreater(len(feats.to_vector()), 20)

    def test_texture_glcm_extraction(self):
        # Create high contrast checkered pattern
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[::2, ::2] = 255
        feats = self.extractor.extract(arr)
        self.assertGreater(feats.texture.contrast, 0.0)
        self.assertGreater(feats.texture.homogeneity, 0.0)
        self.assertLessEqual(feats.texture.homogeneity, 1.0)
        self.assertGreater(feats.texture.energy, 0.0)

if __name__ == "__main__":
    unittest.main()
