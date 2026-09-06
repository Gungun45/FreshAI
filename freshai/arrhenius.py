"""
FreshAI — Biological Arrhenius Post-Harvest Degradation & Shelf-Life Engine
Grounded in empirical food science standards (USDA / FAO / UC Davis Postharvest Technology).

Solves the core issues of naive physical chemistry Arrhenius:
1. Biological Chilling Injury (CI):
   - For chilling-sensitive tropical & subtropical crops (Tomatoes, Bananas, Onions, Potatoes,
     Watermelons, Cucumbers), temperatures below their threshold (min_safe_temp_c) do NOT slow
     decay; instead, membrane lipid phase transitions and cell collapse accelerate spoiling,
     blackening, off-flavor development, and fungal decay.
2. Heat Stress Senescence:
   - Temperatures above 30°C–32°C trigger respiratory runaway, moisture transpiration, and rapid shriveling.
3. Defect Area & Cuticle Integrity Factor:
   - Physical defects (bruises, punctures, lesions, rot) break the protective fruit cuticle,
     opening pathways for pathogens and accelerating senescence proportional to defect_area_pct.
4. Harmonized Storage Advice:
   - Provides realistic kitchen bounds, alerting the user immediately if refrigeration is harmful.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

R_GAS_CONSTANT = 8.314  # J/(mol·K)
T_REF_KELVIN = 293.15    # 20°C Reference Temperature in Kelvin


@dataclass
class ProduceStorageProfile:
    name: str
    ea_joules: float                   # Senescence activation energy in J/mol
    base_ambient_days_20c: float       # Realistic kitchen shelf-life at 20°C for peak fresh produce
    max_shelf_cap_days: float          # Strict upper bound for home kitchen storage
    allow_refrigeration: bool          # False for Onions, Potatoes, Bananas, Tomatoes, etc.
    storage_type: str                  # Recommended physical location
    min_safe_temp_c: float             # Chilling injury threshold (°C) - temperatures below this cause damage
    optimal_temp_c: float              # Ideal storage temperature (°C)
    fridge_warning: Optional[str] = None  # Warning explanation if refrigeration causes damage


PRODUCE_STORAGE_PROFILES: Dict[str, ProduceStorageProfile] = {
    "onion": ProduceStorageProfile(
        name="Onion",
        ea_joules=54000.0,
        base_ambient_days_20c=14.0,
        max_shelf_cap_days=21.0,
        allow_refrigeration=False,
        storage_type="Pantry / Open Ventilated Basket (Cool & Dry)",
        min_safe_temp_c=10.0,
        optimal_temp_c=18.0,
        fridge_warning="⚠️ Do not refrigerate! Fridge moisture causes black mold (Aspergillus), softening, and foul sulfur odor.",
    ),
    "potato": ProduceStorageProfile(
        name="Potato",
        ea_joules=48000.0,
        base_ambient_days_20c=16.0,
        max_shelf_cap_days=25.0,
        allow_refrigeration=False,
        storage_type="Dark Ventilated Pantry (12–18°C)",
        min_safe_temp_c=8.0,
        optimal_temp_c=14.0,
        fridge_warning="⚠️ Do not refrigerate! Cold below 8°C triggers cold-induced sweetening (acrylamide risk) and rot.",
    ),
    "garlic": ProduceStorageProfile(
        name="Garlic",
        ea_joules=52000.0,
        base_ambient_days_20c=14.0,
        max_shelf_cap_days=21.0,
        allow_refrigeration=False,
        storage_type="Dry Mesh Bag / Counter Basket",
        min_safe_temp_c=10.0,
        optimal_temp_c=18.0,
        fridge_warning="⚠️ Do not refrigerate! High fridge humidity triggers rapid mold and premature sprouting.",
    ),
    "tomato": ProduceStorageProfile(
        name="Tomato",
        ea_joules=65000.0,
        base_ambient_days_20c=7.0,
        max_shelf_cap_days=14.0,
        allow_refrigeration=True,
        storage_type="Refrigerator Crisper / Cool Pantry (4–10°C)",
        min_safe_temp_c=0.0,
        optimal_temp_c=4.0,
        fridge_warning=None,
    ),
    "banana": ProduceStorageProfile(
        name="Banana",
        ea_joules=72000.0,
        base_ambient_days_20c=5.0,
        max_shelf_cap_days=7.0,
        allow_refrigeration=False,
        storage_type="Fruit Basket / Hook (Room Temp 18–22°C)",
        min_safe_temp_c=12.0,
        optimal_temp_c=18.0,
        fridge_warning="⚠️ Do not refrigerate! Chilling below 12°C triggers polyphenol oxidase, turning skin pitch black within 48h.",
    ),
    "watermelon": ProduceStorageProfile(
        name="Watermelon",
        ea_joules=60000.0,
        base_ambient_days_20c=14.0,
        max_shelf_cap_days=21.0,
        allow_refrigeration=False,
        storage_type="Cool Pantry or Countertop (15–20°C)",
        min_safe_temp_c=10.0,
        optimal_temp_c=15.0,
        fridge_warning="⚠️ Whole watermelon suffers chilling injury below 10°C (rind pitting and loss of lycopene). Only refrigerate sliced.",
    ),
    "cucumber": ProduceStorageProfile(
        name="Cucumber",
        ea_joules=58000.0,
        base_ambient_days_20c=5.0,
        max_shelf_cap_days=8.0,
        allow_refrigeration=False,
        storage_type="Cool Pantry or Top Fridge Shelf (>10°C)",
        min_safe_temp_c=10.0,
        optimal_temp_c=12.0,
        fridge_warning="⚠️ Sensitive to chilling injury below 10°C (water-soaked spots and rapid liquefaction).",
    ),
    "mango": ProduceStorageProfile(
        name="Mango",
        ea_joules=68000.0,
        base_ambient_days_20c=5.0,
        max_shelf_cap_days=8.0,
        allow_refrigeration=False,
        storage_type="Fruit Basket at Room Temperature",
        min_safe_temp_c=12.0,
        optimal_temp_c=18.0,
        fridge_warning="⚠️ Cold below 12°C halts ripening, causing gray discoloration and pitted skin.",
    ),
    "apple": ProduceStorageProfile(
        name="Apple",
        ea_joules=50000.0,
        base_ambient_days_20c=8.0,
        max_shelf_cap_days=28.0,
        allow_refrigeration=True,
        storage_type="Refrigerator Crisper Drawer (1–4°C)",
        min_safe_temp_c=0.0,
        optimal_temp_c=3.0,
        fridge_warning=None,
    ),
    "orange": ProduceStorageProfile(
        name="Orange / Citrus",
        ea_joules=56000.0,
        base_ambient_days_20c=7.0,
        max_shelf_cap_days=21.0,
        allow_refrigeration=True,
        storage_type="Cool Pantry or Fridge Crisper (4–8°C)",
        min_safe_temp_c=3.0,
        optimal_temp_c=5.0,
        fridge_warning=None,
    ),
    "lemon": ProduceStorageProfile(
        name="Lemon",
        ea_joules=55000.0,
        base_ambient_days_20c=8.0,
        max_shelf_cap_days=21.0,
        allow_refrigeration=True,
        storage_type="Cool Pantry or Fridge Crisper (4–8°C)",
        min_safe_temp_c=4.0,
        optimal_temp_c=6.0,
        fridge_warning=None,
    ),
    "pepper": ProduceStorageProfile(
        name="Bell Pepper",
        ea_joules=58000.0,
        base_ambient_days_20c=6.0,
        max_shelf_cap_days=14.0,
        allow_refrigeration=True,
        storage_type="Fridge Crisper in Perforated Paper Bag (7–10°C)",
        min_safe_temp_c=7.0,
        optimal_temp_c=8.0,
        fridge_warning="Keep in warmer part of fridge (crisper drawer) to avoid sheet pitting below 7°C.",
    ),
    "carrot": ProduceStorageProfile(
        name="Carrot",
        ea_joules=46000.0,
        base_ambient_days_20c=10.0,
        max_shelf_cap_days=30.0,
        allow_refrigeration=True,
        storage_type="Refrigerator Crisper Drawer (0–4°C in sealed bag)",
        min_safe_temp_c=0.0,
        optimal_temp_c=2.0,
        fridge_warning=None,
    ),
    "broccoli": ProduceStorageProfile(
        name="Broccoli",
        ea_joules=62000.0,
        base_ambient_days_20c=3.5,
        max_shelf_cap_days=10.0,
        allow_refrigeration=True,
        storage_type="Refrigerator Crisper Drawer (0–4°C)",
        min_safe_temp_c=0.0,
        optimal_temp_c=2.0,
        fridge_warning=None,
    ),
    "strawberry": ProduceStorageProfile(
        name="Strawberry / Berries",
        ea_joules=68000.0,
        base_ambient_days_20c=2.0,
        max_shelf_cap_days=7.0,
        allow_refrigeration=True,
        storage_type="Refrigerator Shallow Container (0–2°C, Unwashed)",
        min_safe_temp_c=0.0,
        optimal_temp_c=1.0,
        fridge_warning=None,
    ),
    "grape": ProduceStorageProfile(
        name="Grape",
        ea_joules=52000.0,
        base_ambient_days_20c=5.0,
        max_shelf_cap_days=14.0,
        allow_refrigeration=True,
        storage_type="Refrigerator Ventilated Bag (0–2°C)",
        min_safe_temp_c=0.0,
        optimal_temp_c=1.0,
        fridge_warning=None,
    ),
    "default": ProduceStorageProfile(
        name="Fresh Produce",
        ea_joules=58000.0,
        base_ambient_days_20c=5.0,
        max_shelf_cap_days=10.0,
        allow_refrigeration=True,
        storage_type="Cool Ventilated Place",
        min_safe_temp_c=5.0,
        optimal_temp_c=10.0,
        fridge_warning=None,
    ),
}


def get_produce_profile(produce_name: str) -> ProduceStorageProfile:
    p_lower = produce_name.lower().strip()
    for key, profile in PRODUCE_STORAGE_PROFILES.items():
        if key in p_lower:
            return profile
    return PRODUCE_STORAGE_PROFILES["default"]


def calculate_arrhenius_degradation_ratio(
    temp_c: float,
    ea_joules: float,
    min_safe_temp_c: float = 0.0,
    optimal_temp_c: float = 20.0,
) -> float:
    """
    Biological post-harvest Arrhenius degradation ratio:
      k_effective(T) / k(20°C)
    
    Combines:
    1. Base Arrhenius enzymatic senescence kinetics:
         k(T) / k(20°C) = exp( (Ea / R) * (1/293.15 - 1/(T+273.15)) )
    2. Chilling Injury (CI) Penalty when temp_c < min_safe_temp_c:
         For tropical/subtropical crops (Tomato, Banana, Cucumber, Watermelon, Onion, Potato),
         temperatures below min_safe_temp_c damage cellular membranes, causing rapid tissue
         collapse, dark pitting, fungal susceptibility, and off-flavors.
         Degradation rate accelerates sharply instead of dropping.
    3. Heat Stress Senescence when temp_c > 30°C:
         High ambient temperatures trigger respiratory runaway and water transpiration.
    """
    t_kelvin = max(273.15, temp_c + 273.15)
    opt_kelvin = max(273.15, optimal_temp_c + 273.15)

    # 1. Check for Chilling Injury (CI)
    if min_safe_temp_c > 0.0 and temp_c < min_safe_temp_c:
        # Reference degradation rate at optimal safe storage temperature
        k_opt = math.exp((ea_joules / R_GAS_CONSTANT) * ((1.0 / T_REF_KELVIN) - (1.0 / opt_kelvin)))
        delta_chill = min_safe_temp_c - temp_c
        # Chilling injury accelerates breakdown; severity scales with cold severity
        chilling_acceleration = 1.0 + 2.5 * ((delta_chill / max(1.0, min_safe_temp_c)) ** 1.3)
        return k_opt * chilling_acceleration

    # 2. Standard Arrhenius Chemical Senescence
    exponent = (ea_joules / R_GAS_CONSTANT) * ((1.0 / T_REF_KELVIN) - (1.0 / t_kelvin))
    base_rate = math.exp(max(-3.0, min(3.0, exponent)))

    # 3. Heat Stress Senescence Acceleration (>30°C)
    if temp_c > 30.0:
        delta_heat = temp_c - 30.0
        heat_mult = 1.0 + 0.06 * (delta_heat ** 1.2)
        base_rate *= heat_mult

    return base_rate


@dataclass
class ArrheniusShelfLifeResult:
    produce_name: str
    freshness_score: int              # 0 - 100
    ambient_temp_c: float             # Current evaluated temperature (°C)
    ambient_shelf_days: float         # Predicted shelf life at ambient_temp_c
    ideal_storage_days: float         # Shelf life under recommended optimal conditions
    ideal_storage_name: str           # e.g. "Countertop (18°C)" vs "Refrigerator Crisper (4°C)"
    allow_refrigeration: bool         # False for Onion/Potato/Tomato/Banana/etc.
    fridge_warning: Optional[str]     # Warning message explaining why cold storage is harmful
    degradation_acceleration: float   # k_effective(T) / k(20°C)
    activation_energy_kj: float       # Ea in kJ/mol
    is_chilling_injury_active: bool   # True if current temp triggers physiological chilling injury
    chilling_threshold_c: float       # Safe temperature floor (°C)
    defect_area_pct: float            # Defect area % factored in
    defect_penalty_applied_pct: float # Shelf life reduction caused by defects (%)
    recommendation: str               # Actionable user recommendation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "produce_name": self.produce_name,
            "freshness_score": self.freshness_score,
            "ambient_temp_c": round(self.ambient_temp_c, 1),
            "ambient_shelf_days": round(self.ambient_shelf_days, 1),
            "ideal_storage_days": round(self.ideal_storage_days, 1),
            "ideal_storage_location": self.ideal_storage_name,
            "allow_refrigeration": self.allow_refrigeration,
            "fridge_warning": self.fridge_warning,
            "degradation_rate_multiplier": round(self.degradation_acceleration, 2),
            "activation_energy_kj": round(self.activation_energy_kj, 1),
            "is_chilling_injury_active": self.is_chilling_injury_active,
            "chilling_threshold_c": round(self.chilling_threshold_c, 1),
            "defect_area_pct": round(self.defect_area_pct, 1),
            "defect_penalty_applied_pct": round(self.defect_penalty_applied_pct, 1),
            "recommendation": self.recommendation,
        }


def estimate_arrhenius_shelf_life(
    produce_name: str,
    freshness_score_0_100: int,
    current_temp_c: float = 22.0,
    defect_area_pct: float = 0.0,
) -> ArrheniusShelfLifeResult:
    """
    Real-world calibrated Arrhenius shelf life engine.
    Respects post-harvest storage biology, chilling injury thresholds, and defect barrier loss.
    """
    profile = get_produce_profile(produce_name)
    fresh_fraction = max(0.0, min(1.0, freshness_score_0_100 / 100.0))

    # Calculate defect penalty on protective skin/cuticle integrity
    # Lesions, cuts, and fungal spots compromise skin barriers
    defect_penalty_factor = min(0.88, (max(0.0, defect_area_pct) / 100.0) * 3.2)
    effective_fresh_fraction = fresh_fraction * (1.0 - defect_penalty_factor)

    # Completely spoiled item handling
    if effective_fresh_fraction <= 0.08 or freshness_score_0_100 <= 10:
        return ArrheniusShelfLifeResult(
            produce_name=profile.name,
            freshness_score=freshness_score_0_100,
            ambient_temp_c=current_temp_c,
            ambient_shelf_days=0.0,
            ideal_storage_days=0.0,
            ideal_storage_name=profile.storage_type,
            allow_refrigeration=profile.allow_refrigeration,
            fridge_warning=profile.fridge_warning,
            degradation_acceleration=1.0,
            activation_energy_kj=profile.ea_joules / 1000.0,
            is_chilling_injury_active=False,
            chilling_threshold_c=profile.min_safe_temp_c,
            defect_area_pct=defect_area_pct,
            defect_penalty_applied_pct=round(defect_penalty_factor * 100.0, 1),
            recommendation="Item is deteriorated / past safety threshold. Discard or compost immediately.",
        )

    # Biological Non-linear Degradation Kinetics:
    # Intact produce maintains protective cuticle membranes, but once tissue softens,
    # ages, or suffers lesions, fungal colonization and respiration follow an exponential curve.
    decay_curve = math.pow(max(0.01, effective_fresh_fraction), 1.65)
    base_nominal_days = profile.base_ambient_days_20c * decay_curve

    # Calculate degradation rate at current ambient temperature
    k_ambient = calculate_arrhenius_degradation_ratio(
        temp_c=current_temp_c,
        ea_joules=profile.ea_joules,
        min_safe_temp_c=profile.min_safe_temp_c,
        optimal_temp_c=profile.optimal_temp_c,
    )
    ambient_days = max(0.5, min(profile.max_shelf_cap_days, base_nominal_days / max(0.1, k_ambient)))

    # Calculate degradation rate at ideal optimal storage conditions
    k_ideal = calculate_arrhenius_degradation_ratio(
        temp_c=profile.optimal_temp_c,
        ea_joules=profile.ea_joules,
        min_safe_temp_c=profile.min_safe_temp_c,
        optimal_temp_c=profile.optimal_temp_c,
    )
    ideal_days = max(0.5, min(profile.max_shelf_cap_days, base_nominal_days / max(0.1, k_ideal)))

    # Determine if chilling injury is active at current_temp_c
    is_ci = profile.min_safe_temp_c > 0.0 and current_temp_c < profile.min_safe_temp_c

    # Construct actionable scientific recommendation
    if is_ci:
        rec = (
            f"⚠️ CHILLING INJURY ACTIVE! Temperature ({int(round(current_temp_c))}°C) is below safe threshold "
            f"({int(round(profile.min_safe_temp_c))}°C). Accelerates decay ({round(k_ambient, 1)}x). "
            f"Move to {profile.storage_type} immediately to preserve remaining ~{round(ambient_days, 1)} days."
        )
    elif not profile.allow_refrigeration:
        rec = (
            f"Store in {profile.storage_type}. Best consumed within {int(round(ambient_days))} days at {int(round(current_temp_c))}°C."
        )
        if profile.fridge_warning:
            rec += f" {profile.fridge_warning}"
    else:
        rec = (
            f"Lasts ~{round(ambient_days, 1)}d at {int(round(current_temp_c))}°C (or up to ~{round(ideal_days, 1)}d in {profile.storage_type})."
        )

    return ArrheniusShelfLifeResult(
        produce_name=profile.name,
        freshness_score=freshness_score_0_100,
        ambient_temp_c=current_temp_c,
        ambient_shelf_days=ambient_days,
        ideal_storage_days=ideal_days,
        ideal_storage_name=f"{profile.storage_type} ({int(round(profile.optimal_temp_c))}°C)",
        allow_refrigeration=profile.allow_refrigeration,
        fridge_warning=profile.fridge_warning,
        degradation_acceleration=k_ambient,
        activation_energy_kj=profile.ea_joules / 1000.0,
        is_chilling_injury_active=is_ci,
        chilling_threshold_c=profile.min_safe_temp_c,
        defect_area_pct=defect_area_pct,
        defect_penalty_applied_pct=round(defect_penalty_factor * 100.0, 1),
        recommendation=rec,
    )
