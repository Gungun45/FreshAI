"""
Unit tests for Biological Arrhenius Degradation & Shelf-Life Engine.
"""

import unittest
from freshai.arrhenius import (
    estimate_arrhenius_shelf_life,
    calculate_arrhenius_degradation_ratio,
    get_produce_profile,
    PRODUCE_STORAGE_PROFILES,
)


class TestArrheniusEngine(unittest.TestCase):

    def test_tomato_refrigeration(self):
        """Tomato allows refrigeration and cold storage extends shelf life."""
        res_fridge = estimate_arrhenius_shelf_life("Tomato", 85, current_temp_c=4.0)
        res_counter = estimate_arrhenius_shelf_life("Tomato", 85, current_temp_c=22.0)

        self.assertTrue(res_fridge.allow_refrigeration)
        self.assertFalse(res_fridge.is_chilling_injury_active)
        self.assertGreater(res_fridge.ideal_storage_days, res_counter.ambient_shelf_days)

    def test_banana_chilling_injury(self):
        """Banana in fridge (4°C) turns black and degrades faster than at room temperature (20°C)."""
        res_fridge = estimate_arrhenius_shelf_life("Banana", 80, current_temp_c=4.0)
        res_room = estimate_arrhenius_shelf_life("Banana", 80, current_temp_c=20.0)

        self.assertTrue(res_fridge.is_chilling_injury_active)
        self.assertGreater(res_fridge.degradation_acceleration, 1.5)
        self.assertLess(res_fridge.ambient_shelf_days, res_room.ambient_shelf_days)

    def test_apple_cold_tolerance(self):
        """Apple is cold tolerant (0°C threshold) and benefits from refrigeration."""
        res_fridge = estimate_arrhenius_shelf_life("Apple", 85, current_temp_c=4.0)
        res_room = estimate_arrhenius_shelf_life("Apple", 85, current_temp_c=20.0)

        self.assertFalse(res_fridge.is_chilling_injury_active)
        self.assertTrue(res_fridge.allow_refrigeration)
        # Apple lasts longer in the fridge than at room temperature
        self.assertGreater(res_fridge.ambient_shelf_days, res_room.ambient_shelf_days)

    def test_heat_stress_acceleration(self):
        """High temperatures (35°C) must trigger heat stress acceleration."""
        res_warm = estimate_arrhenius_shelf_life("Apple", 85, current_temp_c=20.0)
        res_hot = estimate_arrhenius_shelf_life("Apple", 85, current_temp_c=35.0)

        self.assertGreater(res_hot.degradation_acceleration, 2.5)
        self.assertLess(res_hot.ambient_shelf_days, res_warm.ambient_shelf_days)

    def test_defect_cuticle_barrier_loss(self):
        """Produce with physical defects must have reduced shelf life due to cuticle rupture."""
        res_clean = estimate_arrhenius_shelf_life("Tomato", 80, current_temp_c=20.0, defect_area_pct=0.0)
        res_defective = estimate_arrhenius_shelf_life("Tomato", 80, current_temp_c=20.0, defect_area_pct=15.0)

        self.assertGreater(res_clean.ambient_shelf_days, res_defective.ambient_shelf_days)
        self.assertGreater(res_defective.defect_penalty_applied_pct, 20.0)

    def test_spoiled_produce(self):
        """Severely degraded produce returns 0 days remaining shelf life."""
        res = estimate_arrhenius_shelf_life("Onion", 8, current_temp_c=20.0)
        self.assertEqual(res.ambient_shelf_days, 0.0)
        self.assertIn("past safety threshold", res.recommendation)

    def test_watermelon_profile_and_chilling(self):
        """Watermelon must be recognized and show chilling sensitivity below 10°C."""
        res_fridge = estimate_arrhenius_shelf_life("Watermelon", 85, current_temp_c=4.0)
        self.assertEqual(res_fridge.produce_name, "Watermelon")
        self.assertTrue(res_fridge.is_chilling_injury_active)
        self.assertFalse(res_fridge.allow_refrigeration)

    def test_carrot_and_broccoli_profiles(self):
        """Root and cruciferous veggies must support refrigeration."""
        res_carrot = estimate_arrhenius_shelf_life("Carrot", 90, current_temp_c=4.0)
        res_broccoli = estimate_arrhenius_shelf_life("Broccoli", 90, current_temp_c=4.0)

        self.assertTrue(res_carrot.allow_refrigeration)
        self.assertTrue(res_broccoli.allow_refrigeration)
        self.assertFalse(res_carrot.is_chilling_injury_active)
        self.assertFalse(res_broccoli.is_chilling_injury_active)

    def test_onion_compromised_shelf_life(self):
        """A softening/compromised onion with 15% defects must have ~1-2 days shelf life, NOT 11 days."""
        res_compromised = estimate_arrhenius_shelf_life("Onion", 55, current_temp_c=22.0, defect_area_pct=15.0)
        # Shelf life must be <= 2.5 days (realistic 1-2 days remaining)
        self.assertLessEqual(res_compromised.ambient_shelf_days, 2.5)
        self.assertGreater(res_compromised.ambient_shelf_days, 0.5)

        # Fresh onion must retain long storage capability
        res_fresh = estimate_arrhenius_shelf_life("Onion", 95, current_temp_c=20.0, defect_area_pct=0.0)
        self.assertGreater(res_fresh.ambient_shelf_days, 8.0)
        self.assertGreater(res_fresh.ambient_shelf_days, res_compromised.ambient_shelf_days * 3.0)

    def test_onion_stage_terminology(self):
        """Onion is a non-climacteric bulb and must never be labeled 'Ripe'."""
        from freshai.freshness_detector import format_produce_stage_name, format_ripeness_stage_name
        # Stage index 2 (which is 'Ripe' for bananas) must be 'Aged / Softening' for onion
        onion_stage = format_produce_stage_name(2, "Onion")
        self.assertNotIn("Ripe", onion_stage)
        self.assertIn("Aged", onion_stage)

        # Ripeness stage for onion must not be 'Ripe'
        onion_ripeness = format_ripeness_stage_name(2, "Onion")
        self.assertNotIn("Ripe", onion_ripeness)


if __name__ == "__main__":
    unittest.main()
