"""
FreshAI - Stage 6: LLM + RAG Recommendations Engine
Grounded in Post-Harvest Food Science, USDA FoodData Central, and Trusted Dietary Sources.

Architecture Principle:
ML Models predict → RAG retrieves reliable knowledge → LLM explains and recommends.
The LLM will not directly predict numerical age or shelf life; those values come from
trained ML models (ConvNeXt, XGBoost/LightGBM, Arrhenius).
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

# ─────────────────────────────────────────────────────────────────────────────
# 1. Post-Harvest Science & Produce Domain Knowledge Base (RAG Chunks)
# Grounded in USDA FoodData Central, FAO, WHO, and Post-Harvest Biology Literature
# ─────────────────────────────────────────────────────────────────────────────

PRODUCE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "tomato": {
        "scientific_name": "Solanum lycopersicum",
        "climacteric": True,
        "ethylene_production": "Moderate to High (1.0 – 10.0 µL/kg·h)",
        "ethylene_sensitivity": "High",
        "optimal_storage_temp": "4°C – 8°C (Refrigerator crisper) / 15°C – 20°C (Countertop)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Refrigerator Crisper Drawer (or Countertop if ripening)",
        "refrigeration_advice": "Store ripe tomatoes in refrigerator crisper to extend shelf life up to 10-14 days. Bring to room temperature before eating for best flavor.",
        "chilling_injury_temp": "None for ripe tomatoes (refrigeration preserves firmness and prevents spoilage)",
        "ethylene_co_location": "Keep isolated from ethylene-sensitive greens like cucumbers, broccoli, and leafy greens. Can be paired with unripe avocados to naturally hasten their ripening.",
        "handling_and_preservation": [
            "Store stem-side down to prevent moisture transpirational loss through the stem scar and block microbial entry.",
            "Do not wash until immediately prior to consumption; surface moisture promotes Cladosporium and Botrytis mold proliferation.",
            "If black lesions appear (<10% area), excise affected tissue with a 1 cm sterile margin and cook thoroughly.",
            "If overripe (softening), simmer into marinara or puree within 24–48 hours to preserve bioactive compounds."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 18,
            "carbohydrates_g": 3.9,
            "dietary_fiber_g": 1.2,
            "protein_g": 0.9,
            "fat_g": 0.2,
            "vitamins": {
                "Vitamin C": "13.7 mg (15% DV)",
                "Vitamin A (Beta-carotene)": "833 IU (17% DV)",
                "Vitamin K": "7.9 mcg (7% DV)",
                "Folate (B9)": "15 mcg (4% DV)",
            },
            "minerals": {
                "Potassium": "237 mg (5% DV)",
                "Magnesium": "11 mg",
                "Phosphorus": "24 mg",
            },
            "antioxidants": "Rich in Lycopene (3,041 µg/100g), Beta-Carotene, Chlorogenic Acid, and Naringenin.",
            "health_facts": [
                "Cardiovascular Protection: Lycopene and potassium support healthy endothelial function and blood pressure regulation (Harvard T.H. Chan School of Public Health).",
                "Cellular Defense: High lycopene concentration provides potent antioxidant defense against lipid peroxidation and UV-induced cellular stress (WHO Dietary Guidelines).",
                "Eye Health: Beta-carotene and lutein protect retinal ganglion cells and slow macular degeneration."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 170457)", "Harvard T.H. Chan School of Public Health", "WHO Fruit and Vegetable Intake Advisory"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Heirloom Caprese Salad with Cold-Pressed Olive Oil",
                "prep_time": "10 mins",
                "difficulty": "Easy",
                "instructions": "Thickly slice firm tomatoes. Layer with fresh mozzarella, sweet basil, cold-pressed olive oil, and balsamic glaze."
            },
            "ripe_grade_b": {
                "title": "Slow-Roasted Garlic Marinara Reduction",
                "prep_time": "35 mins",
                "difficulty": "Easy",
                "instructions": "Halve softened tomatoes, toss with garlic, oregano, and olive oil. Roast at 200°C for 25 mins, then puree into a rich pasta reduction."
            },
            "salvage_grade_c": {
                "title": "Andalusian Zero-Waste Gazpacho",
                "prep_time": "15 mins",
                "difficulty": "Easy",
                "instructions": "Blend overripe tomatoes with cucumber, garlic, olive oil, and stale bread. Chill and serve cold to salvage nutrients without food waste."
            }
        }
    },
    "apple": {
        "scientific_name": "Malus domestica",
        "climacteric": True,
        "ethylene_production": "Very High (10.0 – 100.0 µL/kg·h)",
        "ethylene_sensitivity": "Moderate",
        "optimal_storage_temp": "1°C – 4°C (Crisper drawer of refrigerator)",
        "optimal_humidity": "90% – 95% RH",
        "storage_location": "Refrigerator Crisper Drawer (High humidity setting)",
        "refrigeration_advice": "Refrigerate promptly. Apples respire and soften up to 10 times faster at room temperature (20°C) than in 2°C refrigeration.",
        "chilling_injury_temp": "< 0°C (Freezing tissue injury begins at -1.5°C)",
        "ethylene_co_location": "CRITICAL: Apples are prolific ethylene emitters. Isolate in a perforated bag away from potatoes, onions, carrots, and leafy greens to prevent premature sprouting and yellowing.",
        "handling_and_preservation": [
            "Maintain high humidity in crisper drawer to prevent skin wrinkling and flesh dehydration.",
            "Cut away surface bruises before Penicillium expansum (blue mold) penetrates the apple carpel core.",
            "Promptly segregate any rotting apple—the saying 'one bad apple spoils the bunch' is biochemically true due to spore aerosolization."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 52,
            "carbohydrates_g": 13.8,
            "dietary_fiber_g": 2.4,
            "protein_g": 0.3,
            "fat_g": 0.2,
            "vitamins": {
                "Vitamin C": "4.6 mg (5% DV)",
                "Vitamin K": "2.2 mcg",
                "Vitamin B6": "0.04 mg",
            },
            "minerals": {
                "Potassium": "107 mg",
                "Calcium": "6 mg",
                "Magnesium": "5 mg",
            },
            "antioxidants": "Abundant in Quercetin, Catechin, Chlorogenic Acid, and Cyanidin-3-galactoside (in red peels).",
            "health_facts": [
                "Metabolic Regulation: High soluble pectin fiber forms a gut gel that modulates postprandial glucose absorption (Harvard Health).",
                "Gut Microbiome Fuel: Pectin acts as a prebiotic substrate, feeding beneficial short-chain fatty acid (SCFA) producing gut bacteria (WHO).",
                "Cardiovascular Health: Regular apple consumption is clinically associated with lower serum LDL cholesterol and reduced arterial inflammation."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 171688)", "Harvard T.H. Chan School of Public Health", "American Heart Association"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Crisp Apple & Toasted Walnut Fennel Slaw",
                "prep_time": "10 mins",
                "difficulty": "Easy",
                "instructions": "Thinly slice crisp apples and fennel. Toss with toasted walnuts, fresh lemon juice, and Greek yogurt vinaigrette."
            },
            "ripe_grade_b": {
                "title": "Warm Cinnamon Apple Compote",
                "prep_time": "20 mins",
                "difficulty": "Easy",
                "instructions": "Dice slightly bruised apples. Simmer with cinnamon, nutmeg, a splash of lemon juice, and maple syrup until tender."
            },
            "salvage_grade_c": {
                "title": "Zero-Waste Spiced Apple Butter Preserve",
                "prep_time": "40 mins",
                "difficulty": "Medium",
                "instructions": "Simmer softened bruised apples with apple cider vinegar, cinnamon, and cloves until reduced to a spreadable jam."
            }
        }
    },
    "banana": {
        "scientific_name": "Musa acuminata",
        "climacteric": True,
        "ethylene_production": "High (5.0 – 40.0 µL/kg·h)",
        "ethylene_sensitivity": "High",
        "optimal_storage_temp": "13°C – 16°C (Ventilated countertop hanging)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Ventilated Countertop (Hanging hook)",
        "refrigeration_advice": "Do not refrigerate yellow-green or firm bananas. Sub-12°C temperatures trigger chilling injury: polyphenol oxidase enzymes oxidize peel phenols into dark melanin pigment, turning the peel black while flesh stays starch-heavy.",
        "chilling_injury_temp": "< 12°C",
        "ethylene_co_location": "Wrap the central crown cluster with beeswax or aluminum foil to throttle ethylene emission. Keep away from other produce unless ripening acceleration is intended.",
        "handling_and_preservation": [
            "Hang bananas on a designated produce hook to eliminate flat-surface contact pressure bruising.",
            "Once speckled/overripe, peel, slice, and freeze in airtight silicone bags for up to 6 months of smoothie and baking utility.",
            "Store away from direct solar radiation or oven vents to prevent accelerated enzymatic starch breakdown."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 89,
            "carbohydrates_g": 22.8,
            "dietary_fiber_g": 2.6,
            "protein_g": 1.1,
            "fat_g": 0.3,
            "vitamins": {
                "Vitamin B6": "0.37 mg (22% DV)",
                "Vitamin C": "8.7 mg (10% DV)",
                "Folate (B9)": "20 mcg (5% DV)",
            },
            "minerals": {
                "Potassium": "358 mg (8% DV)",
                "Magnesium": "27 mg (7% DV)",
                "Manganese": "0.3 mg (13% DV)",
            },
            "antioxidants": "Dopamine, Catechin, and Polyphenol antioxidants; resistant starch converts to digestible sucrose.",
            "health_facts": [
                "Electrolyte & Neuromuscular Balance: High bioavailable potassium and magnesium prevent exercise-induced muscle cramps and support cardiac rhythm (American Heart Association).",
                "Digestive Health: Green/yellow bananas supply resistant starch (prebiotic); speckled bananas produce higher accessible antioxidant fractions (Harvard Health).",
                "Cognitive & Mood Support: Vitamin B6 is a mandatory cofactor in dopamine and serotonin synthesis."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 173944)", "American Heart Association", "Harvard T.H. Chan School of Public Health"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Chia & Caramelized Banana Morning Bowl",
                "prep_time": "8 mins",
                "difficulty": "Easy",
                "instructions": "Slice firm ripe banana over warm rolled oats with chia seeds, raw honey, and creamy almond butter."
            },
            "ripe_grade_b": {
                "title": "Whole-Grain Cinnamon Banana Bread",
                "prep_time": "45 mins",
                "difficulty": "Medium",
                "instructions": "Mash sweet speckled bananas with whole wheat flour, eggs, cinnamon, and dark chocolate chips. Bake at 175°C for 45 mins."
            },
            "salvage_grade_c": {
                "title": "1-Ingredient Dairy-Free Nice Cream",
                "prep_time": "5 mins",
                "difficulty": "Easy",
                "instructions": "Freeze peeled, spotted bananas. Blitz in a food processor with a pinch of sea salt until silky soft-serve texture."
            }
        }
    },
    "onion": {
        "scientific_name": "Allium cepa",
        "climacteric": False,
        "ethylene_production": "Very Low (< 0.1 µL/kg·h)",
        "ethylene_sensitivity": "Moderate",
        "optimal_storage_temp": "10°C – 15°C (Cool, dark, dry, ventilated pantry)",
        "optimal_humidity": "65% – 70% RH (Low humidity is crucial)",
        "storage_location": "Dark Pantry / Hanging Mesh Bag (Never in closed plastic)",
        "refrigeration_advice": "Never store whole uncut onions in the refrigerator. Cold humid air causes starch conversion and encourages Aspergillus niger (black mold) and premature root sprouting.",
        "chilling_injury_temp": "Tolerates cold, but high humidity (>75%) triggers mold rot and green sprouting.",
        "ethylene_co_location": "NEVER store adjacent to potatoes. Onions release volatile sulfur and moisture that accelerate potato eyes sprouting, while potato moisture causes onions to rot.",
        "handling_and_preservation": [
            "Store in total darkness inside breathable wire baskets or mesh sacks; ambient light triggers bitter green chlorophyll sprouting.",
            "Keep dry. Avoid any water contact until peeling.",
            "Once cut, seal the remaining half in an airtight glass container and refrigerate; consume within 7 days."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 40,
            "carbohydrates_g": 9.3,
            "dietary_fiber_g": 1.7,
            "protein_g": 1.1,
            "fat_g": 0.1,
            "vitamins": {
                "Vitamin C": "7.4 mg (8% DV)",
                "Vitamin B6": "0.12 mg (7% DV)",
                "Folate (B9)": "19 mcg (5% DV)",
            },
            "minerals": {
                "Potassium": "146 mg",
                "Calcium": "23 mg",
                "Manganese": "0.13 mg",
            },
            "antioxidants": "Extremely high in Quercetin (potent anti-inflammatory flavonoid) and sulfurous Alliin.",
            "health_facts": [
                "Cardiovascular & Anti-Inflammatory: Quercetin inhibits arterial LDL oxidation and platelet aggregation (Harvard Health).",
                "Immune & Antimicrobial: Organosulfur compounds exhibit antibacterial and antifungal efficacy against common pathogens (WHO).",
                "Glycemic Control: Bioactive sulfur compounds enhance insulin sensitivity in metabolic tissues."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 170000)", "Harvard T.H. Chan School of Public Health", "WHO Botanical Monographs"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Crisp Pickled Red Onions with Peppercorn",
                "prep_time": "15 mins",
                "difficulty": "Easy",
                "instructions": "Thinly slice crisp onions. Steep in warm cider vinegar with salt, raw sugar, and peppercorns. Perfect for tacos and grain bowls."
            },
            "ripe_grade_b": {
                "title": "French Onion Caramelized Base",
                "prep_time": "40 mins",
                "difficulty": "Medium",
                "instructions": "Slowly cook sliced onions in butter with thyme for 35 mins until deep mahogany brown. Deglaze with vegetable broth."
            },
            "salvage_grade_c": {
                "title": "Balsamic Onion Savory Reduction Jam",
                "prep_time": "30 mins",
                "difficulty": "Easy",
                "instructions": "Simmer softened onions with balsamic vinegar, rosemary, and brown sugar until glossy and jammy for cheese boards."
            }
        }
    },
    "potato": {
        "scientific_name": "Solanum tuberosum",
        "climacteric": False,
        "ethylene_production": "Very Low (< 0.1 µL/kg·h)",
        "ethylene_sensitivity": "High (Ethylene triggers premature eye sprouting)",
        "optimal_storage_temp": "7°C – 10°C (Cool, dark, ventilated cellar)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Dark Pantry / Burlap Sack (Cool, ventilated, absolute darkness)",
        "refrigeration_advice": "Never refrigerate raw potatoes below 6°C. Cold-induced sweetening converts starches into reducing sugars, which form carcinogenic acrylamide when fried or roasted at high heat.",
        "chilling_injury_temp": "< 4°C (Triggers cold-induced sweetening and internal black spot browning)",
        "ethylene_co_location": "Keep well away from onions and apples. Ethylene triggers eye sprouts, which elevate toxic solanine glycoalkaloid levels.",
        "handling_and_preservation": [
            "Store in absolute darkness; light exposure produces green chlorophyll accompanied by bitter solanine neurotoxins.",
            "Cut away any small green patches or sprouts before cooking. If green extends deep into the tuber, discard.",
            "Never store in sealed plastic bags; respiratory condensation causes bacterial soft rot."
        ],
        "nutrition": {
            "serving_size": "100g (baked with skin)",
            "calories_kcal": 93,
            "carbohydrates_g": 21.2,
            "dietary_fiber_g": 2.2,
            "protein_g": 2.5,
            "fat_g": 0.1,
            "vitamins": {
                "Vitamin C": "12.8 mg (14% DV)",
                "Vitamin B6": "0.3 mg (18% DV)",
                "Folate (B9)": "28 mcg (7% DV)",
            },
            "minerals": {
                "Potassium": "535 mg (11% DV)",
                "Magnesium": "28 mg (7% DV)",
                "Iron": "1.1 mg (6% DV)",
            },
            "antioxidants": "Chlorogenic acid, carotenoids, and flavonoids concentrated predominantly in the skin.",
            "health_facts": [
                "Blood Pressure Support: Exceptionally high potassium-to-sodium ratio blunts the cardiovascular effects of dietary sodium (American Heart Association).",
                "Satiety & Gut Health: Boiled and cooled potatoes form type-3 resistant starch, boosting intestinal butyrate production (Harvard Health).",
                "Cellular Energy: Rich Vitamin B6 supports amino acid metabolism and neurotransmitter balance."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 170028)", "Harvard T.H. Chan School of Public Health", "American Heart Association"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Crispy Rosemary & Garlic Roasted Wedges",
                "prep_time": "35 mins",
                "difficulty": "Easy",
                "instructions": "Cut into wedges, parboil for 5 mins, toss with olive oil, crushed garlic, rosemary, and sea salt. Roast at 220°C until golden."
            },
            "ripe_grade_b": {
                "title": "Velvety Leek & Potato Potage",
                "prep_time": "30 mins",
                "difficulty": "Easy",
                "instructions": "Simmer cubed potatoes with sliced leeks in vegetable broth until soft. Blend smooth with a swirl of olive oil and chives."
            },
            "salvage_grade_c": {
                "title": "Golden Potato Latkes or Hash Relish",
                "prep_time": "20 mins",
                "difficulty": "Medium",
                "instructions": "Grate slightly softened potatoes, squeeze out moisture, mix with onion and egg, and fry into crispy golden patties."
            }
        }
    },
    "orange": {
        "scientific_name": "Citrus sinensis",
        "climacteric": False,
        "ethylene_production": "Very Low (< 0.1 µL/kg·h)",
        "ethylene_sensitivity": "Low to Moderate",
        "optimal_storage_temp": "3°C – 8°C (Crisper drawer) or 15°C (1–2 weeks countertop)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Crisper Drawer for maximum shelf life; countertop for immediate consumption",
        "refrigeration_advice": "Refrigeration in the crisper drawer extends shelf life up to 4 weeks. Store loose or in mesh bags; trapped moisture causes Penicillium digitatum (green mold).",
        "chilling_injury_temp": "< 2°C (Causes rind pitting and brown staining)",
        "ethylene_co_location": "Relatively tolerant to ethylene, but keep away from heavy emitters to avoid accelerated rind breakdown.",
        "handling_and_preservation": [
            "Ensure the rind stays dry; any moisture droplet on the peel invites citrus green/blue mold.",
            "Roll on countertop with light palm pressure before juicing to rupture internal juice vesicles and maximize yield.",
            "Zest the peel before juicing or discarding; freeze citrus zest for long-term culinary aromatic use."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 47,
            "carbohydrates_g": 11.8,
            "dietary_fiber_g": 2.4,
            "protein_g": 0.9,
            "fat_g": 0.1,
            "vitamins": {
                "Vitamin C": "53.2 mg (59% DV)",
                "Folate (B9)": "30 mcg (8% DV)",
                "Thiamin (B1)": "0.09 mg (7% DV)",
            },
            "minerals": {
                "Potassium": "181 mg (4% DV)",
                "Calcium": "40 mg (3% DV)",
            },
            "antioxidants": "High in Hesperidin (citrus bioflavonoid), Beta-Cryptoxanthin, and Narirutin.",
            "health_facts": [
                "Immune Function: Single 100g serving delivers over half the recommended daily intake of Vitamin C, bolstering neutrophil chemotaxis (WHO).",
                "Vascular Elasticity: Hesperidin flavonoid improves arterial microvascular reactivity and reduces diastolic blood pressure (Harvard Health).",
                "Non-Heme Iron Bioavailability: Vitamin C and citric acid significantly enhance iron absorption from plant-based foods."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 169097)", "Harvard T.H. Chan School of Public Health", "WHO Nutrition Guidelines"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Citrus Arugula & Avocado Salad",
                "prep_time": "10 mins",
                "difficulty": "Easy",
                "instructions": "Segment fresh orange supreme. Toss with peppery arugula, sliced avocado, shaved red onion, and citrus vinaigrette."
            },
            "ripe_grade_b": {
                "title": "Fresh Orange & Ginger Immunity Glaze",
                "prep_time": "15 mins",
                "difficulty": "Easy",
                "instructions": "Juice softening oranges, simmer with fresh ginger, honey, and tamari into a fragrant glaze for roasted tofu or vegetables."
            },
            "salvage_grade_c": {
                "title": "Zero-Waste Orange Marmalade or Peel Candy",
                "prep_time": "45 mins",
                "difficulty": "Medium",
                "instructions": "Julienne peels, blanch thrice, and simmer with orange pulp and raw sugar until caramelized and set."
            }
        }
    },
    "watermelon": {
        "scientific_name": "Citrullus lanatus",
        "climacteric": False,
        "ethylene_production": "Low",
        "ethylene_sensitivity": "Very High",
        "optimal_storage_temp": "12°C – 15°C (Whole); 2°C – 4°C (Once sliced)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Countertop/Pantry (Whole); Sealed Refrigeration (Cut)",
        "refrigeration_advice": "Never refrigerate whole uncut watermelon below 7°C for more than 5 days. Chilling injury degrades lycopene, causes rind pitting, and induces mealiness. Once sliced, cover tightly and refrigerate immediately.",
        "chilling_injury_temp": "< 7°C for whole watermelons",
        "ethylene_co_location": "EXTREMELY sensitive to ethylene. Exposure to apples, bananas, or tomatoes triggers rind thinning, pulp softening, and premature souring.",
        "handling_and_preservation": [
            "Store whole melons at ambient temperature until 2 hours prior to slicing if chilled fruit is desired.",
            "Once cut, cover the exposed flesh with beeswax wrap or airtight container and consume within 3–4 days.",
            "Do not discard the rind; the white inner rind is edible, highly nutritious, and ideal for quick pickling."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 30,
            "carbohydrates_g": 7.6,
            "dietary_fiber_g": 0.4,
            "protein_g": 0.6,
            "fat_g": 0.2,
            "vitamins": {
                "Vitamin C": "8.1 mg (9% DV)",
                "Vitamin A": "569 IU (11% DV)",
            },
            "minerals": {
                "Potassium": "112 mg",
                "Magnesium": "10 mg",
            },
            "antioxidants": "Extremely high in Lycopene (4,532 µg/100g — 40% higher than raw tomatoes) and L-Citrulline amino acid.",
            "health_facts": [
                "Nitric Oxide & Vascular Health: L-Citrulline converts to L-Arginine, promoting nitric oxide synthesis and endothelial vasodilation (Harvard Health).",
                "Cellular Hydration: 92% bio-structured water with balanced potassium prevents cellular dehydration (WHO).",
                "Muscle Recovery: Citrulline supplementation accelerates muscle lactic acid clearance post-exercise."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 167765)", "Harvard T.H. Chan School of Public Health", "American Journal of Hypertension"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Chilled Watermelon, Feta & Mint Salad",
                "prep_time": "10 mins",
                "difficulty": "Easy",
                "instructions": "Cube crisp cold watermelon, toss with crumbled feta cheese, fresh mint, lime juice, and flaky sea salt."
            },
            "ripe_grade_b": {
                "title": "Agua Fresca de Sandía (Hydration Cooler)",
                "prep_time": "5 mins",
                "difficulty": "Easy",
                "instructions": "Blend cubed watermelon with lime juice and ice; strain for an electrolyte-packed hydrating drink."
            },
            "salvage_grade_c": {
                "title": "Pickled Watermelon Rind & Sorbet",
                "prep_time": "25 mins",
                "difficulty": "Medium",
                "instructions": "Puree overripe flesh into freezer pops. Pickle the firm white rind in vinegar, ginger, and cloves for a crunchy zero-waste condiment."
            }
        }
    },
    "bell pepper": {
        "scientific_name": "Capsicum annuum",
        "climacteric": False,
        "ethylene_production": "Low (< 0.1 µL/kg·h)",
        "ethylene_sensitivity": "Moderate",
        "optimal_storage_temp": "7°C – 10°C (Crisper drawer)",
        "optimal_humidity": "90% – 95% RH",
        "storage_location": "Refrigerator Crisper Drawer (High humidity setting)",
        "refrigeration_advice": "Store in the refrigerator crisper drawer in a perforated bag. Chilling injury occurs below 7°C after prolonged exposure (calyx decay and sheet pitting).",
        "chilling_injury_temp": "< 7°C (Causes calyx rot and surface pitting)",
        "ethylene_co_location": "Keep separate from high ethylene emitters like apples and ripe bananas.",
        "handling_and_preservation": [
            "Keep the stem intact; removing the green calyx accelerates stem-end moisture transpiration.",
            "Ensure surface is dry; moisture on the glossy wax layer fosters Botrytis cinerea gray mold.",
            "Once cut, store seeds and strips in an airtight container lined with a dry paper towel."
        ],
        "nutrition": {
            "serving_size": "100g (Red Bell Pepper)",
            "calories_kcal": 31,
            "carbohydrates_g": 6.0,
            "dietary_fiber_g": 2.1,
            "protein_g": 1.0,
            "fat_g": 0.3,
            "vitamins": {
                "Vitamin C": "127.7 mg (142% DV — highest of common produce!)",
                "Vitamin A": "3131 IU (63% DV)",
                "Vitamin B6": "0.29 mg (15% DV)",
            },
            "minerals": {
                "Potassium": "211 mg",
                "Magnesium": "12 mg",
            },
            "antioxidants": "Capsanthin, Violaxanthin, Lutein, and Quercetin.",
            "health_facts": [
                "Unsurpassed Vitamin C Density: Contains over 200% the Vitamin C of an orange, catalyzing collagen synthesis and non-heme iron uptake (Harvard Health).",
                "Ocular Health: High concentrations of lutein and zeaxanthin filter blue light and protect the retinal macula (WHO).",
                "Metabolic Vitality: High antioxidant capacity shields mitochondrial membranes from reactive oxygen species."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 170108)", "Harvard T.H. Chan School of Public Health", "WHO Essential Nutrition Guidelines"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Rainbow Pepper & Hummus Medley",
                "prep_time": "8 mins",
                "difficulty": "Easy",
                "instructions": "Slice crisp sweet peppers into spears. Serve with spiced garlic hummus and za'atar."
            },
            "ripe_grade_b": {
                "title": "Roasted Red Pepper & Walnut Dip (Muhammara)",
                "prep_time": "25 mins",
                "difficulty": "Easy",
                "instructions": "Char peppers under broiler, peel skins, blend with toasted walnuts, pomegranate molasses, and garlic."
            },
            "salvage_grade_c": {
                "title": "Slow-Simmered Peperonata Stew",
                "prep_time": "30 mins",
                "difficulty": "Easy",
                "instructions": "Slice softening peppers, simmer slowly with onions, garlic, olive oil, and crushed tomatoes until meltingly tender."
            }
        }
    },
    "strawberry": {
        "scientific_name": "Fragaria × ananassa",
        "climacteric": False,
        "ethylene_production": "Very Low",
        "ethylene_sensitivity": "Moderate",
        "optimal_storage_temp": "0°C – 2°C (Coldest part of refrigerator)",
        "optimal_humidity": "90% – 95% RH",
        "storage_location": "Refrigerator Top Shelf / Coldest Zone in breathable container",
        "refrigeration_advice": "Refrigerate immediately upon acquisition. Strawberries are highly perishable and deteriorate rapidly at room temperature via Botrytis mold.",
        "chilling_injury_temp": "Not chilling sensitive; store as cold as possible without freezing (-0.5°C threshold).",
        "ethylene_co_location": "Low ethylene emitter; keep away from ethylene sources to prevent texture softening.",
        "handling_and_preservation": [
            "NEVER wash strawberries before storage; excess moisture acts as a mold catalyst. Wash only immediately before eating.",
            "Leave green calyx hulls intact until ready to consume.",
            "Line storage container with paper towel to absorb ambient humidity."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 32,
            "carbohydrates_g": 7.7,
            "dietary_fiber_g": 2.0,
            "protein_g": 0.7,
            "fat_g": 0.3,
            "vitamins": {
                "Vitamin C": "58.8 mg (65% DV)",
                "Folate (B9)": "24 mcg (6% DV)",
            },
            "minerals": {
                "Manganese": "0.39 mg (17% DV)",
                "Potassium": "153 mg",
            },
            "antioxidants": "Pelargonidin anthocyanins, Ellagic Acid, and Procyanidins.",
            "health_facts": [
                "Cardiovascular Vascular Tone: Anthocyanin intake is strongly correlated with lower myocardial infarction incidence in women (Harvard Health).",
                "Glycemic Regulation: Ellagitannins slow carbohydrate digestion, preventing sharp blood sugar spikes (WHO).",
                "Cognitive Longevity: Flavonoid intake is clinically linked to reduced rate of cognitive decline."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 167762)", "Harvard T.H. Chan School of Public Health", "American Journal of Clinical Nutrition"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Fresh Strawberry, Spinach & Goat Cheese Salad",
                "prep_time": "10 mins",
                "difficulty": "Easy",
                "instructions": "Halve fresh berries, toss with baby spinach, crumbled goat cheese, toasted pecans, and balsamic reduction."
            },
            "ripe_grade_b": {
                "title": "Warm Strawberry Chia Seed Jam",
                "prep_time": "15 mins",
                "difficulty": "Easy",
                "instructions": "Simmer softened berries with chia seeds and maple syrup for 10 mins until thickened without pectin."
            },
            "salvage_grade_c": {
                "title": "Balsamic Roasted Strawberry Reduction",
                "prep_time": "20 mins",
                "difficulty": "Easy",
                "instructions": "Roast bruised strawberries with balsamic glaze at 190°C for 15 mins; spoon over yogurt or porridge."
            }
        }
    },
    "lemon": {
        "scientific_name": "Citrus limon",
        "climacteric": False,
        "ethylene_production": "Very Low",
        "ethylene_sensitivity": "Moderate",
        "optimal_storage_temp": "4°C – 10°C (Crisper drawer / Pantry)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Crisper Drawer in sealed container with paper towel",
        "refrigeration_advice": "Lemons stored at room temperature dehydrate within 7 days. Stored in a sealed container or crisper drawer, they retain juiciness for up to 4 weeks.",
        "chilling_injury_temp": "< 4°C for prolonged periods causes rind pitting and discoloration.",
        "ethylene_co_location": "Moderate sensitivity; keep away from ethylene-rich apples and bananas.",
        "handling_and_preservation": [
            "Store in a breathable silicone bag in crisper to prevent moisture loss.",
            "Always zest the peel before juicing and freeze zest for baking and cooking.",
            "Freeze leftover lemon juice in ice cube trays for convenient culinary acid addition."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 29,
            "carbohydrates_g": 9.3,
            "dietary_fiber_g": 2.8,
            "protein_g": 1.1,
            "fat_g": 0.3,
            "vitamins": {
                "Vitamin C": "53.0 mg (59% DV)",
                "Vitamin B6": "0.08 mg",
            },
            "minerals": {
                "Potassium": "138 mg",
                "Calcium": "26 mg",
            },
            "antioxidants": "Eriocitrin, Hesperidin, and D-Limonene in peel essential oils.",
            "health_facts": [
                "Renal Calculi (Kidney Stone) Prevention: Citric acid increases urinary volume and pH, binding free calcium to inhibit crystal precipitation (Harvard Health).",
                "Antioxidant Defense: Eriocitrin flavonoid protects hepatic tissues from oxidative stress (WHO).",
                "Iron Absorption: Enhances dietary non-heme iron absorption up to threefold when consumed with legumes and greens."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 167746)", "Harvard Health Publishing", "National Kidney Foundation"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Classic Mediterranean Lemon-Herb Vinaigrette",
                "prep_time": "5 mins",
                "difficulty": "Easy",
                "instructions": "Whisk fresh lemon juice with Dijon mustard, minced shallot, oregano, and extra virgin olive oil."
            },
            "ripe_grade_b": {
                "title": "Warm Honey-Lemon Ginger Elixir",
                "prep_time": "8 mins",
                "difficulty": "Easy",
                "instructions": "Steep sliced lemon with fresh ginger in hot water; stir in raw honey for immune and throat support."
            },
            "salvage_grade_c": {
                "title": "Moroccan Salt-Preserved Lemons",
                "prep_time": "15 mins",
                "difficulty": "Medium",
                "instructions": "Quarter softening lemons, pack tightly in sterilized jars with kosher salt and lemon juice; ferment for 3 weeks."
            }
        }
    },
    "mango": {
        "scientific_name": "Mangifera indica",
        "climacteric": True,
        "ethylene_production": "Moderate to High",
        "ethylene_sensitivity": "High",
        "optimal_storage_temp": "12°C – 15°C (Pantry while ripening); 8°C – 10°C (Ripe)",
        "optimal_humidity": "85% – 90% RH",
        "storage_location": "Countertop while green/firm; Refrigerate only once fully ripe",
        "refrigeration_advice": "Never refrigerate unripe mangoes. Cold storage halts ripening permanently and causes chilling injury: grayish rind scald, uneven ripening, and flavor loss.",
        "chilling_injury_temp": "< 10°C for unripe fruit",
        "ethylene_co_location": "Can be placed in a paper bag with an apple or banana to accelerate ripening.",
        "handling_and_preservation": [
            "Store stem-end down on a soft surface to avoid contact bruising.",
            "Once ripe (fragrant and yields slightly to pressure), consume or refrigerate for up to 5 days.",
            "Slice and freeze ripe mango chunks for smoothies or mango lassi."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 60,
            "carbohydrates_g": 15.0,
            "dietary_fiber_g": 1.6,
            "protein_g": 0.8,
            "fat_g": 0.4,
            "vitamins": {
                "Vitamin C": "36.4 mg (40% DV)",
                "Vitamin A": "1082 IU (22% DV)",
                "Folate (B9)": "43 mcg (11% DV)",
            },
            "minerals": {
                "Potassium": "168 mg",
                "Copper": "0.11 mg (12% DV)",
            },
            "antioxidants": "Mangiferin (super-antioxidant), Beta-Carotene, and Gallotannins.",
            "health_facts": [
                "Cellular Shielding: Mangiferin possesses potent neuroprotective and anti-inflammatory properties (Harvard Health).",
                "Digestive Enzymes: Contains amylase digestive enzymes that break down large food starches into easily absorbed sugars (WHO).",
                "Ocular Support: High Vitamin A and zeaxanthin preserve corneal integrity and night vision."
            ],
            "trusted_sources": ["USDA FoodData Central (FDC ID: 169910)", "Harvard T.H. Chan School of Public Health", "WHO Fruit and Vegetable Guidelines"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Chili-Lime Fresh Mango Sticks",
                "prep_time": "5 mins",
                "difficulty": "Easy",
                "instructions": "Spear ripe mango slices, squeeze fresh lime juice, and dust with Tajín or chili powder and flaky sea salt."
            },
            "ripe_grade_b": {
                "title": "Authentic Cardamom Mango Lassi",
                "prep_time": "8 mins",
                "difficulty": "Easy",
                "instructions": "Blend sweet mango pulp with plain whole-milk yogurt, milk, a pinch of ground cardamom, and honey."
            },
            "salvage_grade_c": {
                "title": "Tangy Spiced Mango Chutney",
                "prep_time": "35 mins",
                "difficulty": "Medium",
                "instructions": "Simmer bruised or overripe mango with mustard seeds, ginger, vinegar, brown sugar, and raisins until thick and jammy."
            }
        }
    },
    "general_produce": {
        "scientific_name": "Horticultural Fresh Produce",
        "climacteric": False,
        "ethylene_production": "Low",
        "ethylene_sensitivity": "Moderate",
        "optimal_storage_temp": "4°C (Crisper drawer) or 12°C – 15°C (Cool pantry)",
        "optimal_humidity": "80% – 90% RH",
        "storage_location": "Crisper drawer for leafy/temperate produce; Pantry for tropical/root crops",
        "refrigeration_advice": "Temperate produce thrives at 4°C with high humidity. Subtropical crops prefer 12°C–15°C to avoid chilling injury.",
        "chilling_injury_temp": "Varies by species (<7°C for tropical crops)",
        "ethylene_co_location": "Keep ethylene emitters separate from sensitive vegetables to maximize shelf life.",
        "handling_and_preservation": [
            "Inspect produce regularly and segregate any decaying specimens to prevent spore cross-contamination.",
            "Do not pre-wash before storage; wash immediately prior to preparation.",
            "Maintain moderate air circulation and avoid condensation inside packaging."
        ],
        "nutrition": {
            "serving_size": "100g",
            "calories_kcal": 35,
            "carbohydrates_g": 7.5,
            "dietary_fiber_g": 2.2,
            "protein_g": 1.2,
            "fat_g": 0.2,
            "vitamins": {
                "Vitamin C": "20.0 mg (22% DV)",
                "Vitamin A": "450 IU (9% DV)",
            },
            "minerals": {
                "Potassium": "200 mg (4% DV)",
                "Magnesium": "15 mg",
            },
            "antioxidants": "Essential plant polyphenols, carotenoids, and dietary flavonoids.",
            "health_facts": [
                "Chronic Disease Mitigation: High daily intake of varied fruits and vegetables is the cornerstone of non-communicable disease reduction (WHO Guidelines).",
                "Microbiome Diversity: Varied prebiotic dietary fibers fuel gut microbiome resilience (Harvard Health)."
            ],
            "trusted_sources": ["USDA FoodData Central", "WHO Healthy Diet Guidelines", "Harvard T.H. Chan School of Public Health"]
        },
        "recipes": {
            "fresh_grade_a": {
                "title": "Garden-Fresh Seasonal Medley",
                "prep_time": "10 mins",
                "difficulty": "Easy",
                "instructions": "Wash, slice, and toss in cold-pressed vinaigrette to preserve heat-sensitive vitamins."
            },
            "ripe_grade_b": {
                "title": "Caramelized Sheet-Pan Herb Roast",
                "prep_time": "30 mins",
                "difficulty": "Easy",
                "instructions": "Toss with olive oil, sea salt, rosemary, and roast at 200°C until edges are caramelized and sweet."
            },
            "salvage_grade_c": {
                "title": "Nourishing Zero-Waste Botanical Kitchen Broth",
                "prep_time": "40 mins",
                "difficulty": "Easy",
                "instructions": "Simmer softened produce with peppercorns, garlic, and fresh herbs to extract minerals into a rich broth."
            }
        }
    }
}

# Backward-compatibility aliases for tests and external consumers
for _k, _v in PRODUCE_KNOWLEDGE_BASE.items():
    if "spoilage_mitigation" not in _v and "handling_and_preservation" in _v:
        _v["spoilage_mitigation"] = list(_v["handling_and_preservation"])
    if "nutritional_evolution" not in _v and "nutrition" in _v:
        _v["nutritional_evolution"] = _v["nutrition"].get("antioxidants", "Rich in essential dietary nutrients.")


# ─────────────────────────────────────────────────────────────────────────────
# 1B. Crop Growth Timelines & Developmental Stages (Agronomic Knowledge)
# ─────────────────────────────────────────────────────────────────────────────

CROP_GROWTH_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "tomato": {
        "crop_name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "category": "Vegetable",
        "total_duration_range": "60–85 days (transplant)",
        "average_days": 75,
        "difficulty": "Easy to Moderate",
        "stages": [
            {"stage": 1, "name": "Germination & Sprouting", "days": "6–10 days", "description": "Seed absorbs moisture, radicle sprouts, first cotyledons emerge."},
            {"stage": 2, "name": "Vegetative Growth", "days": "20–25 days", "description": "Rapid stem elongation, branching, and lush canopy establishment."},
            {"stage": 3, "name": "Flowering & Pollination", "days": "15–20 days", "description": "Bright yellow flower clusters form; self-pollination aided by wind or bees."},
            {"stage": 4, "name": "Fruit Formation & Sizing", "days": "20–30 days", "description": "Petals drop, ovaries swell into full-sized firm green tomatoes."},
            {"stage": 5, "name": "Ripening & Breaker Stage", "days": "10–15 days", "description": "Lycopene synthesis triggers color shift from yellow/orange to deep red."}
        ],
        "sunlight": "Full Sun (6–8+ hours direct sunlight daily)",
        "optimal_temperature": "21°C–29°C (70°F–85°F)",
        "water_needs": "1.5–2 inches per week; deep morning watering",
        "harvest_signs": "Uniform glossy red color, slight give to gentle touch, easily slips from vine."
    },
    "onion": {
        "crop_name": "Onion",
        "scientific_name": "Allium cepa",
        "category": "Vegetable",
        "total_duration_range": "100–175 days",
        "average_days": 135,
        "difficulty": "Easy",
        "stages": [
            {"stage": 1, "name": "Germination / Rooting", "days": "7–14 days", "description": "Thread-like green sprout pushes upward; roots anchor."},
            {"stage": 2, "name": "Vegetative Leaf Emergence", "days": "40–60 days", "description": "Tubular leaves emerge; each leaf represents a future bulb ring."},
            {"stage": 3, "name": "Bulb Swelling", "days": "30–50 days", "description": "Daylight triggers energy transfer expanding bulb layers."},
            {"stage": 4, "name": "Maturation & Neck Collapse", "days": "15–25 days", "description": "Tops yellow and fall over; tunic skin hardens into papery scales."},
            {"stage": 5, "name": "Field Curing & Harvest", "days": "10–14 days", "description": "Bulbs lifted and air-cured in dry shade until necks seal."}
        ],
        "sunlight": "Full Sun (12–16 hrs daylight depending on variety)",
        "optimal_temperature": "13°C–24°C (55°F–75°F)",
        "water_needs": "1 inch per week; stop watering 2 weeks before harvest",
        "harvest_signs": "Foliage tops yellow and flop over; bulb neck is dry and papery."
    },
    "potato": {
        "crop_name": "Potato",
        "scientific_name": "Solanum tuberosum",
        "category": "Vegetable",
        "total_duration_range": "70–120 days",
        "average_days": 95,
        "difficulty": "Easy",
        "stages": [
            {"stage": 1, "name": "Sprout Emergence", "days": "10–20 days", "description": "Sprouts emerge from seed tuber eyes through mounded soil."},
            {"stage": 2, "name": "Canopy Growth", "days": "25–35 days", "description": "Dense green canopy and subterranean stolons extend."},
            {"stage": 3, "name": "Tuber Initiation & Flowering", "days": "15–25 days", "description": "Flowers bloom; stolon tips swell underground into new tubers."},
            {"stage": 4, "name": "Tuber Bulking", "days": "25–40 days", "description": "Starch and water deposit rapidly into tubers; volume multiplies."},
            {"stage": 5, "name": "Vines Dieback & Skin Set", "days": "10–15 days", "description": "Foliage withers naturally; skins thicken and cure underground."}
        ],
        "sunlight": "Full Sun (6–8 hours/day)",
        "optimal_temperature": "15°C–21°C (60°F–70°F)",
        "water_needs": "1–2 inches per week; steady moisture prevents knobby tubers",
        "harvest_signs": "Foliage dies back; skins are thick enough that thumb pressure doesn't slip them."
    },
    "apple": {
        "crop_name": "Apple",
        "scientific_name": "Malus domestica",
        "category": "Fruit",
        "total_duration_range": "120–150 days (blossom to fruit)",
        "average_days": 135,
        "difficulty": "Challenging (Perennial Tree)",
        "stages": [
            {"stage": 1, "name": "Winter Dormancy", "days": "60–90 days", "description": "Tree rests; requires chill hours under 7°C to break dormancy."},
            {"stage": 2, "name": "Spring Blossom", "days": "15–25 days", "description": "Fragrant flowers bloom; pollinated by bees."},
            {"stage": 3, "name": "Fruit Set", "days": "15–20 days", "description": "Blossoms drop; tiny green apple fruitlets swell on spurs."},
            {"stage": 4, "name": "Summer Sizing", "days": "50–70 days", "description": "Apples absorb sunlight and nutrients; skin develops sugars and aroma."},
            {"stage": 5, "name": "Ripening & Harvest", "days": "20–30 days", "description": "Background color shifts to yellow/red; seeds turn brown."}
        ],
        "sunlight": "Full Sun (6–8 hours direct sunlight)",
        "optimal_temperature": "18°C–25°C (64°F–77°F)",
        "water_needs": "Moderate; deep soaking every 1–2 weeks during dry spells",
        "harvest_signs": "Stem detaches easily when lifted and rolled upward; seeds inside are dark brown."
    },
    "banana": {
        "crop_name": "Banana",
        "scientific_name": "Musa acuminata",
        "category": "Fruit",
        "total_duration_range": "9–12 months (planting to bunch)",
        "average_days": 300,
        "difficulty": "Moderate (Tropical)",
        "stages": [
            {"stage": 1, "name": "Corm Sprouting", "days": "30–45 days", "description": "Rhizome roots; large rolled leaves unfurl into broad paddles."},
            {"stage": 2, "name": "Pseudostem Growth", "days": "4–6 months", "description": "Massive trunk of packed leaf sheaths reaches 2–4 meters."},
            {"stage": 3, "name": "Shooting Inflorescence", "days": "30–40 days", "description": "Large purple flower bud pushes through crown and bends downward."},
            {"stage": 4, "name": "Bunch Formation", "days": "60–90 days", "description": "Female flowers reveal clusters ('hands') of green fingers."},
            {"stage": 5, "name": "Mature Green Harvest", "days": "20–30 days", "description": "Fingers become round and plump; harvested before ripening."}
        ],
        "sunlight": "Full Sun (8–12 hours direct tropical sun)",
        "optimal_temperature": "26°C–32°C (78°F–90°F)",
        "water_needs": "High (1.5–2.5 inches/week); requires excellent drainage",
        "harvest_signs": "Fingers plump up and ridges round out; top hands lighten."
    },
    "orange": {
        "crop_name": "Orange",
        "scientific_name": "Citrus sinensis",
        "category": "Fruit",
        "total_duration_range": "7–12 months (bloom to ripe fruit)",
        "average_days": 270,
        "difficulty": "Moderate",
        "stages": [
            {"stage": 1, "name": "Bud Break", "days": "20–30 days", "description": "New green foliage flushes alongside white blossom buds."},
            {"stage": 2, "name": "Citrus Blossom", "days": "20–30 days", "description": "Aromatic white blossoms open; pollinated by honeybees."},
            {"stage": 3, "name": "Fruit Set", "days": "30–45 days", "description": "Tiny green marbles set; tree self-thins surplus fruitlets."},
            {"stage": 4, "name": "Summer Sizing", "days": "90–120 days", "description": "Oranges expand, juice vesicles fill; acid diminishes."},
            {"stage": 5, "name": "Color Break & Ripening", "days": "45–75 days", "description": "Cool nights trigger chlorophyll breakdown; vibrant orange hue develops."}
        ],
        "sunlight": "Full Sun (8+ hours daily)",
        "optimal_temperature": "18°C–30°C (65°F–86°F)",
        "water_needs": "Deep watering every 7–14 days",
        "harvest_signs": "Skin is fully orange, fruit is heavy with juice, sweet flavor test."
    },
    "lemon": {
        "crop_name": "Lemon",
        "scientific_name": "Citrus limon",
        "category": "Fruit",
        "total_duration_range": "6–9 months (bloom to ripe fruit)",
        "average_days": 220,
        "difficulty": "Easy to Moderate",
        "stages": [
            {"stage": 1, "name": "Bud Formation", "days": "15–20 days", "description": "New buds show purple blush before white petals open."},
            {"stage": 2, "name": "Evergreen Flowering", "days": "20–30 days", "description": "Sweet aromatic blossoms open."},
            {"stage": 3, "name": "Fruit Set", "days": "40–60 days", "description": "Green fruitlets develop pointed nipple at blossom end."},
            {"stage": 4, "name": "Juice Sizing", "days": "60–90 days", "description": "Internal juice vesicles swell; oils concentrate in rind."},
            {"stage": 5, "name": "Degreening to Yellow", "days": "30–45 days", "description": "Chlorophyll breaks down, transforming rind into vivid sunny yellow."}
        ],
        "sunlight": "Full Sun (6–8 hours minimum)",
        "optimal_temperature": "21°C–30°C (70°F–86°F)",
        "water_needs": "Moderate; allow top 2–3 inches to dry before watering deeply",
        "harvest_signs": "Plump, heavy, vibrant bright yellow with glossy oily rind."
    },
    "mango": {
        "crop_name": "Mango",
        "scientific_name": "Mangifera indica",
        "category": "Fruit",
        "total_duration_range": "100–150 days (flowering to harvest)",
        "average_days": 120,
        "difficulty": "Moderate (Tropical)",
        "stages": [
            {"stage": 1, "name": "Panicle Emergence", "days": "15–25 days", "description": "Large branched pyramidal flower panicles emerge."},
            {"stage": 2, "name": "Blossom & Pollination", "days": "15–20 days", "description": "Thousands of tiny flowers pollinated by insects."},
            {"stage": 3, "name": "Fruit Set", "days": "25–35 days", "description": "Green fruitlets set; natural shedding leaves 1–3 mangos per panicle."},
            {"stage": 4, "name": "Fruit Development", "days": "35–50 days", "description": "Fruit swells into characteristic kidney/oval shape."},
            {"stage": 5, "name": "Shoulder Maturity", "days": "20–30 days", "description": "Shoulders rise above stem attachment; nose rounds; aroma develops."}
        ],
        "sunlight": "Full Sun (8+ hours daily intense sun)",
        "optimal_temperature": "24°C–35°C (75°F–95°F)",
        "water_needs": "Moderate while fruit swells; withhold heavy watering before harvest",
        "harvest_signs": "Shoulders fill out around stem cavity, powdery bloom on skin, sweet aroma."
    },
    "bell_pepper": {
        "crop_name": "Bell Pepper",
        "scientific_name": "Capsicum annuum",
        "category": "Vegetable",
        "total_duration_range": "60–90 days (transplant)",
        "average_days": 75,
        "difficulty": "Moderate",
        "stages": [
            {"stage": 1, "name": "Germination", "days": "8–14 days", "description": "Seeds require warm soil (25°C–30°C) to sprout."},
            {"stage": 2, "name": "Vegetative Branching", "days": "25–35 days", "description": "Bushy branching with glossy leaves and strong central stem."},
            {"stage": 3, "name": "White Flowers", "days": "15–20 days", "description": "Small white star flowers emerge at stem nodes."},
            {"stage": 4, "name": "Green Pepper Sizing", "days": "20–25 days", "description": "Fruit walls thicken and reach full blocky green size."},
            {"stage": 5, "name": "Color Shift & Sweetening", "days": "10–20 days", "description": "Sugar rises as skin turns vibrant red, yellow, or orange."}
        ],
        "sunlight": "Full Sun (6–8 hours/day)",
        "optimal_temperature": "21°C–29°C (70°F–85°F)",
        "water_needs": "1–1.5 inches per week",
        "harvest_signs": "Firm, glossy walls with good weight; snaps cleanly off plant."
    },
    "eggplant": {
        "crop_name": "Eggplant",
        "scientific_name": "Solanum melongena",
        "category": "Vegetable",
        "total_duration_range": "65–85 days (transplant)",
        "average_days": 75,
        "difficulty": "Moderate",
        "stages": [
            {"stage": 1, "name": "Germination", "days": "7–14 days", "description": "Warmth-loving seeds sprout in moist conditions."},
            {"stage": 2, "name": "Foliage & Bush Growth", "days": "25–35 days", "description": "Broad fuzzy leaves and sturdy purple-tinged stems form."},
            {"stage": 3, "name": "Violet Flowering", "days": "15–20 days", "description": "Striking purple star blossoms with golden anthers bloom."},
            {"stage": 4, "name": "Fruit Elongation", "days": "20–30 days", "description": "Fruit swells into glossy deep purple cylinder/bulb."},
            {"stage": 5, "name": "Peak Harvest", "days": "5–10 days", "description": "Skin reaches high mirror gloss before seeds harden."}
        ],
        "sunlight": "Full Sun (8+ hours daily)",
        "optimal_temperature": "22°C–32°C (72°F–90°F)",
        "water_needs": "1–2 inches weekly; consistent hydration",
        "harvest_signs": "Skin is high-gloss and shiny; thumb indent springs right back."
    },
    "strawberry": {
        "crop_name": "Strawberry",
        "scientific_name": "Fragaria × ananassa",
        "category": "Fruit",
        "total_duration_range": "90–120 days",
        "average_days": 100,
        "difficulty": "Easy",
        "stages": [
            {"stage": 1, "name": "Crown Rooting", "days": "15–20 days", "description": "Bare-root crown establishes; leaves unfold."},
            {"stage": 2, "name": "Foliage Growth", "days": "25–35 days", "description": "Compact foliage clump develops with strong crowns."},
            {"stage": 3, "name": "White Blossom", "days": "15–20 days", "description": "White blossoms with golden centers emerge."},
            {"stage": 4, "name": "Berry Sizing", "days": "20–25 days", "description": "Receptacle expands; seeds spread evenly over surface."},
            {"stage": 5, "name": "Full Red Ripening", "days": "10–15 days", "description": "Anthocyanins flood fruit from apex to stem; aroma peaks."}
        ],
        "sunlight": "Full Sun (6–10 hours/day)",
        "optimal_temperature": "15°C–26°C (60°F–80°F)",
        "water_needs": "1–1.5 inches per week",
        "harvest_signs": "Fully red from tip to shoulder with no white neck remaining."
    },
    "watermelon": {
        "crop_name": "Watermelon",
        "scientific_name": "Citrullus lanatus",
        "category": "Fruit",
        "total_duration_range": "75–100 days",
        "average_days": 85,
        "difficulty": "Easy to Moderate",
        "stages": [
            {"stage": 1, "name": "Germination", "days": "5–10 days", "description": "Seeds sprout vigorously in warm soil (22°C–32°C)."},
            {"stage": 2, "name": "Vine Run", "days": "25–35 days", "description": "Long sprawling vines spread across the ground."},
            {"stage": 3, "name": "Yellow Blossoms", "days": "15–20 days", "description": "Male and female flowers bloom for pollination."},
            {"stage": 4, "name": "Melon Bulking", "days": "25–35 days", "description": "Fruit swells rapidly with water and sweet sugars."},
            {"stage": 5, "name": "Sugar Peak & Ripening", "days": "10–15 days", "description": "Flesh turns sweet; ground spot turns rich creamy yellow."}
        ],
        "sunlight": "Full Sun (8–10 hours/day)",
        "optimal_temperature": "24°C–32°C (75°F–90°F)",
        "water_needs": "1–2 inches weekly; reduce before harvest",
        "harvest_signs": "Curled tendril nearest fruit turns brown and dry; ground spot is yellow."
    },
}



# ─────────────────────────────────────────────────────────────────────────────
# 2. Result Explanation Generator (Plain-Language ML Diagnostics)
# ─────────────────────────────────────────────────────────────────────────────

def explain_predictions(
    produce_name: str,
    quality_score: float,
    physiological_age_days: float,
    remaining_shelf_life_days: float,
    spoilage_risk: str,
    defect_area_pct: float,
    storage_temp_c: float,
) -> Dict[str, str]:
    """
    Translates raw numerical machine learning outputs into clear, everyday language.
    Strictly follows the principle: ML Models predict → RAG explains and contextualizes.
    """
    # 1. Quality Score Meaning
    if quality_score >= 82.0:
        quality_meaning = (
            f"Quality Score {quality_score:.0f}/100 indicates PEAK COMMERCIAL FRESHNESS. "
            f"The cellular walls are turgid, nutrient retention is at maximum density, "
            f"and enzymatic degradation has barely commenced."
        )
    elif quality_score >= 60.0:
        quality_meaning = (
            f"Quality Score {quality_score:.0f}/100 indicates GOOD / ACTIVE CONSUMPTION GRADE. "
            f"The produce is ripe with high flavor intensity and concentrated natural sugars, "
            f"showing early cosmetic maturity."
        )
    elif quality_score >= 35.0:
        quality_meaning = (
            f"Quality Score {quality_score:.0f}/100 indicates DISCOUNT / SALVAGE GRADE. "
            f"Cellular moisture loss and internal softening are evident ({defect_area_pct:.1f}% defect area). "
            f"Best suited for cooking, pureeing, or preserving."
        )
    else:
        quality_meaning = (
            f"Quality Score {quality_score:.0f}/100 indicates EXPIRED / ADVANCED DEGRADATION. "
            f"Severe cellular collapse or microbial contamination is present."
        )

    # 2. Estimated Age Meaning
    age_meaning = (
        f"Estimated Biological Age of {physiological_age_days:.1f} days reflects the physiological maturity "
        f"and post-harvest respiration elapsed since harvest. At this stage, natural ripening biochemical pathways "
        f"have completed their flavor development phase."
    )

    # 3. Shelf Life Meaning
    if remaining_shelf_life_days >= 5.0:
        shelf_life_meaning = (
            f"Estimated Shelf Life of {remaining_shelf_life_days:.1f} days gives an ample consumption window. "
            f"Under current ambient storage conditions ({storage_temp_c:.1f}°C), the item will maintain good quality "
            f"for almost a week."
        )
    elif remaining_shelf_life_days >= 2.0:
        shelf_life_meaning = (
            f"Estimated Shelf Life of {remaining_shelf_life_days:.1f} days indicates moderate urgency. "
            f"Consume within the next 2 to 3 days to experience peak flavor and avoid nutrient degradation."
        )
    else:
        shelf_life_meaning = (
            f"Estimated Shelf Life of {remaining_shelf_life_days:.1f} days is CRITICAL. "
            f"Enzymatic breakdown is progressing rapidly. Consume, freeze, or cook today to prevent complete food waste."
        )

    # 4. Spoilage Risk Meaning
    spoilage_meaning = (
        f"Spoilage Risk is rated {spoilage_risk.upper()} with {defect_area_pct:.1f}% defect surface involvement. "
        f"{'Microbial and mold vulnerability is minimal.' if defect_area_pct < 5.0 else 'Surface lesions increase vulnerability to fungal colonization; prioritize consumption.'}"
    )

    summary_statement = (
        f"{produce_name.capitalize()} currently scores {quality_score:.0f}/100 with an estimated age of "
        f"{physiological_age_days:.1f} days and approximately {remaining_shelf_life_days:.1f} days of safe consumption "
        f"remaining at {storage_temp_c:.1f}°C."
    )

    return {
        "quality_score_explanation": quality_meaning,
        "estimated_age_explanation": age_meaning,
        "shelf_life_explanation": shelf_life_meaning,
        "spoilage_risk_explanation": spoilage_meaning,
        "summary_statement": summary_statement,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Structured RAG Output Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProduceRAGRecommendations:
    """Structured post-harvest AI advisory output."""
    produce_name: str
    quality_score: float
    quality_grade: str
    spoilage_risk: str
    storage_temp_current: float
    # 1. Result Explanation (Simple language ML translation)
    prediction_explanation: Dict[str, str]
    # 2. Lifespan Improvement & Storage
    storage_strategy: Dict[str, Any]
    spoilage_mitigation: List[str]
    # 3. Produce Nutrition & Trusted Science
    nutrition_facts: Dict[str, Any]
    nutritional_insights: Dict[str, Any]
    # 4. Culinary Waste Reduction & Synthesis
    chef_recipe: Dict[str, Any]
    ai_advisor_summary: str
    generated_by: str   # "Gemini-LLM-RAG" or "Deterministic-Domain-RAG"
    crop_growth_timeline: Dict[str, Any] = field(default_factory=dict)
    trusted_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "produce_name": self.produce_name,
            "quality_score": round(self.quality_score, 1),
            "quality_grade": self.quality_grade,
            "spoilage_risk": self.spoilage_risk,
            "storage_temp_current": self.storage_temp_current,
            "prediction_explanation": self.prediction_explanation,
            "storage_strategy": self.storage_strategy,
            "spoilage_mitigation": self.spoilage_mitigation,
            "nutrition_facts": self.nutrition_facts,
            "nutritional_insights": self.nutritional_insights,
            "crop_growth_timeline": self.crop_growth_timeline,
            "chef_recipe": self.chef_recipe,
            "ai_advisor_summary": self.ai_advisor_summary,
            "generated_by": self.generated_by,
            "trusted_sources": self.trusted_sources,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid RAG Retrieval & Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────

class FreshAIRAGEngine:
    """
    RAG-powered Post-Harvest Advisory and Waste-Reduction Engine.
    Grounds LLM generation in verified post-harvest science and USDA FoodData Central.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.knowledge_base = PRODUCE_KNOWLEDGE_BASE

    def _retrieve_knowledge(self, produce_class: str) -> Dict[str, Any]:
        """Retrieves domain knowledge chunk matching produce class."""
        pc_clean = produce_class.strip().lower()
        for key in self.knowledge_base:
            if key in pc_clean or pc_clean in key:
                return self.knowledge_base[key]
        return self.knowledge_base["general_produce"]

    def _retrieve_crop_growth(self, produce_class: str) -> Dict[str, Any]:
        """Retrieves crop cultivation timeline & stages matching produce class."""
        pc_clean = produce_class.strip().lower().replace(" ", "_")
        for key in CROP_GROWTH_KNOWLEDGE:
            if key in pc_clean or pc_clean in key:
                return CROP_GROWTH_KNOWLEDGE[key]
        return {
            "crop_name": produce_class.capitalize(),
            "total_duration_range": "60–100 days",
            "average_days": 80,
            "difficulty": "Moderate",
            "stages": [
                {"stage": 1, "name": "Germination", "days": "7–14 days", "description": "Seed sprouting and initial root establishment."},
                {"stage": 2, "name": "Vegetative Growth", "days": "20–35 days", "description": "Canopy and stem expansion."},
                {"stage": 3, "name": "Flowering", "days": "15–20 days", "description": "Blossom set and pollination."},
                {"stage": 4, "name": "Fruit Development", "days": "20–30 days", "description": "Produce sizing and bulking."},
                {"stage": 5, "name": "Harvest Ripening", "days": "10–15 days", "description": "Full maturity and harvest readiness."}
            ],
            "sunlight": "Full Sun (6–8 hours/day)",
            "optimal_temperature": "18°C–28°C",
            "water_needs": "1–1.5 inches per week",
            "harvest_signs": "Firm texture, full varietal coloration, and pleasant aroma."
        }

    def generate_recommendations(
        self,
        produce_class: str,
        quality_score: float,
        physiological_age_days: float,
        remaining_shelf_life_days: float,
        defect_area_pct: float,
        spoilage_risk: str,
        storage_temp_c: float = 22.0,
        days_since_purchase: float = 2.0,
    ) -> ProduceRAGRecommendations:
        """
        Generates full multi-faceted post-harvest recommendations using RAG.
        Strictly enforces:
        ML Models predict → RAG retrieves reliable knowledge → LLM explains and recommends.
        """
        kb = self._retrieve_knowledge(produce_class)

        # Quality Grade Classification
        if quality_score >= 82.0:
            grade = "Grade A (Peak Fresh)"
            recipe_key = "fresh_grade_a"
        elif quality_score >= 60.0:
            grade = "Grade B (Good / Ripening)"
            recipe_key = "ripe_grade_b"
        elif quality_score >= 35.0:
            grade = "Grade C (Discount / Salvage)"
            recipe_key = "salvage_grade_c"
        else:
            grade = "Grade D (Expired / Cull)"
            recipe_key = "salvage_grade_c"

        selected_recipe = kb["recipes"].get(recipe_key, kb["recipes"]["fresh_grade_a"])

        # 1. Generate Plain-Language Prediction Explanation
        pred_explanation = explain_predictions(
            produce_name=produce_class,
            quality_score=quality_score,
            physiological_age_days=physiological_age_days,
            remaining_shelf_life_days=remaining_shelf_life_days,
            spoilage_risk=spoilage_risk,
            defect_area_pct=defect_area_pct,
            storage_temp_c=storage_temp_c,
        )

        # 2. Construct Lifespan Improvement & Storage Strategy
        storage_strategy = {
            "optimal_temperature_target": kb.get("optimal_storage_temp", "4°C – 12°C"),
            "optimal_humidity_target": kb.get("optimal_humidity", "85% – 90% RH"),
            "storage_location": kb.get("storage_location", "Pantry or Crisper Drawer"),
            "current_temp_status": (
                f"Optimal ({storage_temp_c:.1f}°C)"
                if ("refrig" in kb.get("optimal_storage_temp", "").lower() and storage_temp_c < 10.0) or
                   ("pantry" in kb.get("optimal_storage_temp", "").lower() and 10.0 <= storage_temp_c <= 18.0)
                else f"Suboptimal ({storage_temp_c:.1f}°C — adjust storage location to extend shelf life)"
            ),
            "refrigeration_guideline": kb.get("refrigeration_advice", "Follow recommended produce guidelines."),
            "chilling_injury_alert": kb.get("chilling_injury_temp", "None reported"),
            "ethylene_production_level": kb.get("ethylene_production", "Moderate"),
            "ethylene_sensitivity_level": kb.get("ethylene_sensitivity", "Moderate"),
            "ethylene_co_location_warning": kb.get("ethylene_co_location", "Keep separated from incompatible crops."),
            "handling_and_preservation": kb.get("handling_and_preservation", []),
        }

        # 3. Dynamic Spoilage Mitigation tailored to detected defect area %
        mitigation_actions = list(kb.get("handling_and_preservation", []))
        if defect_area_pct > 10.0:
            mitigation_actions.insert(0, f"🚨 HIGH DEFECT AREA ({defect_area_pct:.1f}%): Excise bruised or damaged tissue with a 1 cm margin immediately; cook within 24 hours.")
        elif defect_area_pct > 3.0:
            mitigation_actions.insert(0, f"⚠️ MODERATE BLEMISH ({defect_area_pct:.1f}%): Consume this specimen before unblemished ones.")
        else:
            mitigation_actions.insert(0, "✨ Pristine surface detected: No urgent trimming required.")

        if remaining_shelf_life_days < 2.0:
            mitigation_actions.append(f"⏱️ Critical shelf life ({remaining_shelf_life_days:.1f} days remaining): Cook, freeze, or preserve today to prevent food waste.")

        # 4. Nutrition Facts (USDA FoodData Central / WHO)
        nutrition_kb = kb.get("nutrition", {})
        nutritional_insights = {
            "serving_size": nutrition_kb.get("serving_size", "100g"),
            "calories": f"{nutrition_kb.get('calories_kcal', 30)} kcal",
            "carbohydrates": f"{nutrition_kb.get('carbohydrates_g', 5.0)}g",
            "dietary_fiber": f"{nutrition_kb.get('dietary_fiber_g', 1.5)}g",
            "protein": f"{nutrition_kb.get('protein_g', 0.8)}g",
            "key_vitamins": nutrition_kb.get("vitamins", {}),
            "key_minerals": nutrition_kb.get("minerals", {}),
            "antioxidant_profile": nutrition_kb.get("antioxidants", "Plant polyphenols"),
            "recommended_consumption_window": "Immediate (Peak nutrient density)" if quality_score >= 75.0 else "Cooked / Preserved (Nutrients best retained in stews/purees)",
        }

        trusted_sources = nutrition_kb.get("trusted_sources", [
            "USDA FoodData Central", "Harvard T.H. Chan School of Public Health", "WHO Healthy Diet Guidelines"
        ])

        # 5. Gemini LLM Advisory Synthesis or Offline Deterministic Fallback
        ai_summary = None
        generated_by = "Deterministic-Domain-RAG"

        if self.api_key:
            try:
                ai_summary = self._call_gemini_llm(
                    produce_class=produce_class,
                    quality_score=quality_score,
                    grade=grade,
                    age_days=physiological_age_days,
                    shelf_life=remaining_shelf_life_days,
                    defect_pct=defect_area_pct,
                    spoilage_risk=spoilage_risk,
                    temp=storage_temp_c,
                    kb=kb,
                    pred_explanation=pred_explanation,
                )
                if ai_summary:
                    generated_by = "Gemini-LLM-RAG"
            except Exception as e:
                print(f"[FreshAIRAGEngine] Gemini API notice: {e}. Falling back to domain RAG synthesis.")
                ai_summary = None

        if ai_summary is None:
            # Deterministic domain RAG synthesis narrative
            ai_summary = (
                f"FreshAI Assessment for {produce_class.capitalize()}:\n\n"
                f"📊 Prediction Explanation: With a Quality Score of {quality_score:.0f}/100 ({grade}) and estimated biological age "
                f"of {physiological_age_days:.1f} days, this item has {defect_area_pct:.1f}% defect involvement ({spoilage_risk} risk). "
                f"Under current ambient storage ({storage_temp_c:.1f}°C), the expected remaining shelf life is {remaining_shelf_life_days:.1f} days.\n\n"
                f"🧊 Lifespan Improvement: {kb.get('refrigeration_advice', '')} "
                f"Optimal temperature is {kb.get('optimal_storage_temp', '')} at {kb.get('optimal_humidity', '')}. "
                f"{kb.get('ethylene_co_location', '')}\n\n"
                f"🥗 Nutrition Facts: Delivers {nutrition_kb.get('calories_kcal', 30)} kcal and {nutrition_kb.get('dietary_fiber_g', 1.5)}g fiber per 100g. "
                f"{nutrition_kb.get('antioxidants', '')} (Sources: {', '.join(trusted_sources)}).\n\n"
                f"👨‍🍳 Chef's Zero-Waste Tip: Recommended for '{selected_recipe['title']}' ({selected_recipe['prep_time']} prep time)."
            )

        crop_timeline = self._retrieve_crop_growth(produce_class)

        return ProduceRAGRecommendations(
            produce_name=produce_class.capitalize(),
            quality_score=quality_score,
            quality_grade=grade,
            spoilage_risk=spoilage_risk,
            storage_temp_current=storage_temp_c,
            prediction_explanation=pred_explanation,
            storage_strategy=storage_strategy,
            spoilage_mitigation=mitigation_actions,
            nutrition_facts=nutrition_kb,
            nutritional_insights=nutritional_insights,
            crop_growth_timeline=crop_timeline,
            chef_recipe=selected_recipe,
            ai_advisor_summary=ai_summary,
            generated_by=generated_by,
            trusted_sources=trusted_sources,
        )

    def _call_gemini_llm(
        self,
        produce_class: str,
        quality_score: float,
        grade: str,
        age_days: float,
        shelf_life: float,
        defect_pct: float,
        spoilage_risk: str,
        temp: float,
        kb: Dict[str, Any],
        pred_explanation: Dict[str, str],
    ) -> Optional[str]:
        """
        Calls Google Gemini API with RAG context to produce natural language advisory.
        Enforces: The LLM will not predict numerical age or shelf life; it explains and recommends.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

        nutrition_info = kb.get("nutrition", {})
        prompt = (
            f"You are FreshAI, an expert post-harvest food scientist and culinary advisor.\n"
            f"CRITICAL ARCHITECTURE RULE: You must NOT predict or alter the numerical values. The machine learning models "
            f"have already predicted the exact numbers below. Your job is to explain the numbers in simple language, "
            f"provide storage recommendations to improve lifespan, and detail produce-specific nutrition from trusted sources.\n\n"
            f"ML Model Predictions:\n"
            f"- Produce: {produce_class.capitalize()}\n"
            f"- Quality Score: {quality_score:.0f}/100 ({grade})\n"
            f"- Estimated Age: {age_days:.1f} days\n"
            f"- Remaining Shelf Life: {shelf_life:.1f} days\n"
            f"- Defect Area: {defect_pct:.1f}%\n"
            f"- Spoilage Risk: {spoilage_risk}\n"
            f"- Current Storage Temperature: {temp:.1f}°C\n\n"
            f"RAG Knowledge Context:\n"
            f"- Optimal Temp & Humidity: {kb.get('optimal_storage_temp', '')}, {kb.get('optimal_humidity', '')}\n"
            f"- Storage Rules: {kb.get('refrigeration_advice', '')}\n"
            f"- Ethylene Guidance: {kb.get('ethylene_co_location', '')}\n"
            f"- Handling Practices: {'; '.join(kb.get('handling_and_preservation', [])[:3])}\n"
            f"- Nutrition Profile: {nutrition_info.get('calories_kcal', 30)} kcal, {nutrition_info.get('dietary_fiber_g', 1.5)}g fiber, {nutrition_info.get('antioxidants', '')}\n"
            f"- Trusted Sources: {', '.join(nutrition_info.get('trusted_sources', ['USDA FoodData Central', 'WHO', 'Harvard Health']))}\n\n"
            f"Please structure your response with these clear sections:\n"
            f"1. 🎯 Result Explanation: Explain what the Quality Score ({quality_score:.0f}/100), Estimated Age ({age_days:.1f}d), and Shelf Life ({shelf_life:.1f}d) mean in simple consumer-friendly words.\n"
            f"2. 🧊 Lifespan Improvement: Practical storage recommendations, temperature/humidity guidance, and handling practices.\n"
            f"3. 🥗 Nutrition Facts: Key produce-specific nutrients, vitamins, and health facts cited from trusted sources.\n"
            f"4. 👨‍🍳 Zero-Waste Chef Recommendation: One practical culinary suggestion to use this item before it spoils."
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 450}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
