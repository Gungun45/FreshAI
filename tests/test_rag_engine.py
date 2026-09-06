"""
Unit tests for Stage 6: LLM + RAG Recommendations Engine.
"""

import unittest
from freshai.rag_engine import (
    FreshAIRAGEngine,
    ProduceRAGRecommendations,
    PRODUCE_KNOWLEDGE_BASE,
)


class TestRAGEngine(unittest.TestCase):

    def setUp(self):
        self.engine = FreshAIRAGEngine()

    def test_knowledge_base_integrity(self):
        """Verify required fields exist across all produce items in knowledge base."""
        required_keys = [
            "climacteric", "optimal_storage_temp", "refrigeration_advice",
            "optimal_humidity", "ethylene_co_location", "spoilage_mitigation",
            "nutritional_evolution", "recipes"
        ]
        for produce, data in PRODUCE_KNOWLEDGE_BASE.items():
            for key in required_keys:
                self.assertIn(key, data, f"Missing key {key} in knowledge base for {produce}")
            self.assertIn("fresh_grade_a", data["recipes"])
            self.assertIn("ripe_grade_b", data["recipes"])
            self.assertIn("salvage_grade_c", data["recipes"])

    def test_tomato_rag_recommendations(self):
        """Verify tomato post-harvest advice respects chilling injury rules."""
        rec = self.engine.generate_recommendations(
            produce_class="Tomato",
            quality_score=84.0,
            physiological_age_days=6.0,
            remaining_shelf_life_days=3.5,
            defect_area_pct=5.0,
            spoilage_risk="Low",
            storage_temp_c=22.0,
        )
        self.assertEqual(rec.produce_name, "Tomato")
        self.assertEqual(rec.quality_grade, "Grade A (Peak Fresh)")
        self.assertIn("Never refrigerate", rec.storage_strategy["refrigeration_guideline"])
        self.assertIn("Heirloom Caprese", rec.chef_recipe["title"])
        self.assertIn("10°C", rec.storage_strategy.get("chilling_injury_alert", ""))

    def test_apple_high_defect_salvage(self):
        """Verify high defect area triggers urgent trimming and salvage recipe."""
        rec = self.engine.generate_recommendations(
            produce_class="Apple",
            quality_score=45.0,
            physiological_age_days=12.0,
            remaining_shelf_life_days=1.5,
            defect_area_pct=14.2,
            spoilage_risk="High",
            storage_temp_c=4.0,
        )
        self.assertEqual(rec.quality_grade, "Grade C (Discount / Salvage)")
        # Spoilage mitigation should contain high defect warning
        has_defect_warning = any("HIGH DEFECT" in act for act in rec.spoilage_mitigation)
        self.assertTrue(has_defect_warning)
        # Critical shelf life action
        has_critical_action = any("Critical shelf life" in act for act in rec.spoilage_mitigation)
        self.assertTrue(has_critical_action)
        # Recipe should be salvage grade C
        self.assertIn("Apple Butter", rec.chef_recipe["title"])

    def test_banana_ethylene_and_storage(self):
        """Verify banana chilling injury and crown wrap advice."""
        rec = self.engine.generate_recommendations(
            produce_class="Banana",
            quality_score=75.0,
            physiological_age_days=7.0,
            remaining_shelf_life_days=2.0,
            defect_area_pct=4.0,
            spoilage_risk="Medium",
            storage_temp_c=20.0,
        )
        self.assertIn("crown", rec.storage_strategy["ethylene_co_location_warning"].lower())
        self.assertIn("Banana Bread", rec.chef_recipe["title"])

    def test_fallback_general_produce(self):
        """Verify unmapped produce items fallback gracefully to general produce advice."""
        rec = self.engine.generate_recommendations(
            produce_class="Dragonfruit",
            quality_score=80.0,
            physiological_age_days=5.0,
            remaining_shelf_life_days=4.0,
            defect_area_pct=2.0,
            spoilage_risk="Low",
            storage_temp_c=18.0,
        )
        self.assertEqual(rec.produce_name, "Dragonfruit")
        self.assertIsNotNone(rec.chef_recipe["title"])
        self.assertIsNotNone(rec.ai_advisor_summary)

    def test_to_dict_serialization(self):
        """Verify full dictionary serialization for REST API and JSON clients."""
        rec = self.engine.generate_recommendations(
            produce_class="Onion",
            quality_score=88.0,
            physiological_age_days=5.0,
            remaining_shelf_life_days=10.0,
            defect_area_pct=0.5,
            spoilage_risk="Low",
            storage_temp_c=15.0,
        )
        d = rec.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["produce_name"], "Onion")
        self.assertIn("storage_strategy", d)
        self.assertIn("chef_recipe", d)


if __name__ == "__main__":
    unittest.main()
