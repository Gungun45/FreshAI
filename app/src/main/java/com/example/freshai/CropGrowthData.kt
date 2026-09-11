package com.example.freshai

/**
 * Developmental stage in a crop's lifecycle from seed/planting to harvest.
 */
data class CropStage(
    val stageNumber: Int,
    val name: String,
    val durationDays: String,
    val description: String,
    val progressPct: Int, // 0 to 100
    val iconEmoji: String = "🌱",
)

/**
 * Full agronomic cultivation profile for a fruit or vegetable crop.
 */
data class CropCultivationProfile(
    val name: String,
    val scientificName: String,
    val category: String, // "Vegetable" or "Fruit"
    val emoji: String,
    val totalDurationRange: String, // e.g., "60–85 days"
    val averageDays: Int,
    val difficulty: String, // "Easy", "Moderate", "Challenging"
    val stages: List<CropStage>,
    val sunlight: String,
    val temperature: String,
    val soilAndPh: String,
    val waterNeeds: String,
    val harvestSigns: String,
    val careTips: String,
    val npkRatio: String = "NPK 10-10-10 (Vegetative) / 5-10-15 (Fruiting)",
    val optimalPickingWindow: String = "Early Morning (6:00 AM – 9:30 AM)",
    val diseaseWatch: String = "Early blight, powdery mildew, aphids & whiteflies",
    val gddTarget: String = "1,150–1,400 GDD (Growing Degree Days)"
)

/**
 * Repository providing access to crop growth timelines and cultivation intelligence.
 * Works completely offline on-device.
 */
object CropGrowthRepository {

    val CROPS: List<CropCultivationProfile> = listOf(
        CropCultivationProfile(
            name = "Tomato",
            scientificName = "Solanum lycopersicum",
            category = "Vegetable",
            emoji = "🍅",
            totalDurationRange = "60–85 days (transplant)",
            averageDays = 75,
            difficulty = "Easy to Moderate",
            stages = listOf(
                CropStage(1, "Germination & Sprouting", "6–10 days", "Seed absorbs moisture, radicle sprouts, first cotyledons emerge.", 15, "🌱"),
                CropStage(2, "Vegetative & Foliage Growth", "20–25 days", "Rapid stem elongation, branching, and lush canopy establishment.", 35, "🌿"),
                CropStage(3, "Flowering & Pollination", "15–20 days", "Bright yellow flower clusters form; self-pollination aided by gentle breeze or bees.", 55, "🌼"),
                CropStage(4, "Fruit Formation & Ripening", "10–15 days", "Green fruit swells and enters breaker turning phase (~10–15 days to harvest).", 80, "🟢"),
                CropStage(5, "Harvest Ready", "Peak Ripe", "Deep vibrant red, slight give to gentle touch, ready for harvest now.", 100, "🍅"),
            ),
            sunlight = "Full Sun (6–8+ hours direct sunlight daily)",
            temperature = "21°C–29°C (70°F–85°F) optimal day; 16°C–20°C night",
            soilAndPh = "Rich, loamy, well-draining soil; pH 6.0–6.8",
            waterNeeds = "1.5–2 inches per week; deep morning watering at root base",
            harvestSigns = "Uniform glossy red color, slight give to gentle touch, easily detaches at the pedicel joint.",
            careTips = "Mulch around base to maintain moisture; prune lower suckers to direct energy into fruit clusters.",
        ),

        CropCultivationProfile(
            name = "Onion",
            scientificName = "Allium cepa",
            category = "Vegetable",
            emoji = "🧅",
            totalDurationRange = "100–175 days",
            averageDays = 135,
            difficulty = "Easy",
            stages = listOf(
                CropStage(1, "Germination / Set Rooting", "7–14 days", "Tiny thread-like green sprout pushes upward; fibrous roots anchor.", 15, "🌱"),
                CropStage(2, "Vegetative Leaf Emergence", "40–60 days", "Foliage stalks emerge; each green tubular leaf represents a future bulb ring.", 40, "🌿"),
                CropStage(3, "Bulb Swelling & Formation", "30–50 days", "Daylight triggers energy transfer from leaves to base, expanding the bulb layers.", 75, "🧅"),
                CropStage(4, "Maturation & Neck Collapse", "15–25 days", "Tops yellow and fall over naturally; outer tunic skin hardens into papery scales.", 95, "🍂"),
                CropStage(5, "Field Curing & Harvest", "10–14 days", "Bulbs lifted and air-cured in dry shade until necks are fully sealed and dry.", 100, "🧺"),
            ),
            sunlight = "Full Sun (needs 12–16 hrs daylight depending on short/long-day variety)",
            temperature = "13°C–24°C (55°F–75°F); prefers cool start, warm bulb finish",
            soilAndPh = "Loose, friable sandy loam; pH 6.0–7.0; free of heavy stones",
            waterNeeds = "1 inch per week; stop watering 2 weeks before harvest when tops collapse",
            harvestSigns = "Foliage tops turn yellow and fall over; bulb neck is dry and papery.",
            careTips = "Keep weeding diligent because onions have shallow roots and cannot compete with aggressive weeds.",
        ),

        CropCultivationProfile(
            name = "Potato",
            scientificName = "Solanum tuberosum",
            category = "Vegetable",
            emoji = "🥔",
            totalDurationRange = "70–120 days",
            averageDays = 95,
            difficulty = "Easy",
            stages = listOf(
                CropStage(1, "Sprout Development & Emergence", "10–20 days", "Sprouts emerge from seed tuber eyes and push through mounded soil.", 20, "🌱"),
                CropStage(2, "Vegetative Canopy Growth", "25–35 days", "Dense green canopy and subterranean stolons (underground stems) extend.", 45, "🌿"),
                CropStage(3, "Tuber Initiation & Flowering", "15–25 days", "Delicate flowers bloom; stolon tips swell underground into new potato tubers.", 70, "🌸"),
                CropStage(4, "Tuber Bulking", "25–40 days", "Starch and water deposit rapidly into tubers; volume multiplies.", 90, "🥔"),
                CropStage(5, "Vines Dieback & Skin Set", "10–15 days", "Foliage withers naturally; potato skins thicken and cure underground for harvest.", 100, "📦"),
            ),
            sunlight = "Full Sun (6–8 hours/day); protect tubers from light exposure to prevent greening",
            temperature = "15°C–21°C (60°F–70°F); cooler soil temperature promotes tuber bulking",
            soilAndPh = "Deep, loose, well-drained acidic soil; pH 5.0–6.0 (prevents scab disease)",
            waterNeeds = "1–2 inches per week; steady moisture prevents knobby or hollow tubers",
            harvestSigns = "Foliage dies back and dries; skins are thick enough that gentle thumb pressure doesn't slip them.",
            careTips = "Hill up soil around plants repeatedly as stems grow to prevent sunlight from turning tubers green and bitter.",
        ),

        CropCultivationProfile(
            name = "Bell Pepper",
            scientificName = "Capsicum annuum",
            category = "Vegetable",
            emoji = "🫑",
            totalDurationRange = "60–90 days (transplant)",
            averageDays = 75,
            difficulty = "Moderate",
            stages = listOf(
                CropStage(1, "Germination", "8–14 days", "Seeds require warm soil (25°C–30°C) to break dormancy and sprout.", 15, "🌱"),
                CropStage(2, "Vegetative Branching", "25–35 days", "Bushy branching with glossy deep green leaves and strong central stem.", 40, "🌿"),
                CropStage(3, "Budding & White Flowers", "15–20 days", "Small white star-shaped flowers emerge at stem nodes; self-fertile.", 65, "⭐"),
                CropStage(4, "Green Pepper Development", "20–25 days", "Fruit walls thicken and reach full blocky green size (edible as green pepper).", 85, "🫑"),
                CropStage(5, "Color Shift & Sweetening", "10–20 days", "Chlorophyll breaks down; sugar rises as skin turns red, yellow, or orange.", 100, "🌶️"),
            ),
            sunlight = "Full Sun (6–8 hours/day)",
            temperature = "21°C–29°C (70°F–85°F); sensitive to cold under 12°C",
            soilAndPh = "Fertile, moist, organically amended soil; pH 6.2–7.0",
            waterNeeds = "1–1.5 inches per week; avoid waterlogged roots",
            harvestSigns = "Firm, glossy walls with good weight; snaps cleanly off plant with stem intact.",
            careTips = "Stake plants early to prevent heavy fruit loads from snapping branches during summer storms.",
        ),

        CropCultivationProfile(
            name = "Eggplant",
            scientificName = "Solanum melongena",
            category = "Vegetable",
            emoji = "🍆",
            totalDurationRange = "65–85 days (transplant)",
            averageDays = 75,
            difficulty = "Moderate",
            stages = listOf(
                CropStage(1, "Germination", "7–14 days", "Warmth-loving seeds sprout slowly under warm, moist conditions.", 15, "🌱"),
                CropStage(2, "Foliage & Bushy Growth", "25–35 days", "Broad fuzzy leaves and sturdy purple-tinged woody stems form.", 45, "🌿"),
                CropStage(3, "Violet Flowering", "15–20 days", "Striking purple star blossoms with bright yellow anthers bloom.", 65, "💜"),
                CropStage(4, "Fruit Elongation & Gloss", "20–30 days", "Dark calyx expands; fruit swells into glossy deep purple cylinder/bulb.", 85, "🍆"),
                CropStage(5, "Peak Harvest Maturity", "5–10 days", "Skin reaches high mirror gloss before seeds harden.", 100, "✨"),
            ),
            sunlight = "Full Sun (8+ hours daily warmth)",
            temperature = "22°C–32°C (72°F–90°F); highly heat tolerant",
            soilAndPh = "Rich, loamy, compost-enriched soil; pH 5.8–6.8",
            waterNeeds = "1–2 inches weekly; deep consistent hydration prevents spongy bitter flesh",
            harvestSigns = "Skin is high-gloss and shiny (dull skin means overripe and seedy); thumb indent springs right back.",
            careTips = "Clip fruit off with shears leaving 1 inch of stem; never pull to avoid stripping the plant branches.",
        ),

        CropCultivationProfile(
            name = "Apple",
            scientificName = "Malus domestica",
            category = "Fruit",
            emoji = "🍎",
            totalDurationRange = "120–150 days (blossom to fruit)",
            averageDays = 135,
            difficulty = "Challenging (Perennial Tree)",
            stages = listOf(
                CropStage(1, "Winter Dormancy & Chilling", "60–90 days", "Tree rests in winter; requires 500–1000 chill hours under 7°C to break dormancy.", 15, "❄️"),
                CropStage(2, "Bud Break & Spring Blossom", "15–25 days", "Pink-to-white fragrant flowers bloom in clusters; cross-pollinated by bees.", 35, "🌸"),
                CropStage(3, "Petal Fall & Fruit Set", "15–20 days", "Blossoms drop; tiny green apple fruitlets swell on spurs.", 55, "🍏"),
                CropStage(4, "Summer Sizing & Cell Expansion", "50–70 days", "Apples absorb sunlight and nutrients; skin develops sugars and aroma.", 85, "🌳"),
                CropStage(5, "Coloring & Harvest Ripening", "20–30 days", "Background turns from green to cream/yellow; red blush intensifies; starch converts to sugar.", 100, "🍎"),
            ),
            sunlight = "Full Sun (6–8 hours direct sunlight)",
            temperature = "18°C–25°C (64°F–77°F) during growing season; cold winters essential",
            soilAndPh = "Well-drained loamy soil with clay subsoil; pH 6.0–7.0",
            waterNeeds = "Moderate; established trees need deep soaking every 1–2 weeks in dry spells",
            harvestSigns = "Stem detaches easily when lifted and rolled upward; seeds inside turn dark brown; sweet crisp taste.",
            careTips = "Thin young fruit clusters in early summer to 1 apple per cluster for larger fruit size and annual fruiting.",
        ),

        CropCultivationProfile(
            name = "Banana",
            scientificName = "Musa acuminata",
            category = "Fruit",
            emoji = "🍌",
            totalDurationRange = "9–12 months (planting to bunch)",
            averageDays = 300,
            difficulty = "Moderate (Tropical Climate)",
            stages = listOf(
                CropStage(1, "Sucker / Corm Sprouting", "30–45 days", "Rhizome corm roots; large rolled cigar leaves unfurl into broad paddles.", 20, "🌱"),
                CropStage(2, "Pseudostem Vegetative Growth", "4–6 months", "Massive trunk of tightly packed leaf sheaths reaches 2–4 meters tall.", 50, "🌴"),
                CropStage(3, "Shooting / Purple Heart Inflorescence", "30–40 days", "Large purple flower bud pushes through crown and bends downward.", 70, "🌺"),
                CropStage(4, "Hand & Finger Bunch Formation", "60–90 days", "Female flowers drop petals revealing clusters ('hands') of green upward-curving fingers.", 90, "🍌"),
                CropStage(5, "Fattening & Mature Green Harvest", "20–30 days", "Fingers become round and plump (ribs disappear); harvested green before ripening.", 100, "📦"),
            ),
            sunlight = "Full Sun (8–12 hours direct tropical sun)",
            temperature = "26°C–32°C (78°F–90°F); ceases growth below 14°C; damaged by frost",
            soilAndPh = "Deep, rich, alluvial or volcanic loam high in organic matter; pH 5.5–6.5",
            waterNeeds = "High (1.5–2.5 inches/week); requires excellent drainage to prevent corm rot",
            harvestSigns = "Banana fingers plump up and ridges become rounded; top hands begin to lighten.",
            careTips = "Remove surplus suckers, leaving only 1 main pseudostem and 1 follower sucker for continuous yield.",
        ),

        CropCultivationProfile(
            name = "Orange",
            scientificName = "Citrus sinensis",
            category = "Fruit",
            emoji = "🍊",
            totalDurationRange = "7–12 months (bloom to ripe fruit)",
            averageDays = 270,
            difficulty = "Moderate (Subtropical Tree)",
            stages = listOf(
                CropStage(1, "Spring Flushes & Bud Break", "20–30 days", "Fragrant new green foliage flushes alongside white blossom buds.", 15, "🌿"),
                CropStage(2, "Flowering & Citrus Blossom", "20–30 days", "Highly aromatic white star blossoms open; pollinated by honeybees.", 30, "🌸"),
                CropStage(3, "Fruit Set & June Drop", "30–45 days", "Tiny green marbles set; tree self-thins surplus fruitlets.", 50, "🟢"),
                CropStage(4, "Summer Sizing & Acid Reduction", "90–120 days", "Oranges expand, juice vesicles fill; citric acid gradually diminishes.", 80, "🌳"),
                CropStage(5, "Color Break & Sugar Accumulation", "45–75 days", "Cool nights trigger chlorophyll breakdown; vibrant orange hue and high Brix sugar develop.", 100, "🍊"),
            ),
            sunlight = "Full Sun (8+ hours daily)",
            temperature = "18°C–30°C (65°F–86°F); intolerant of sustained freezing temperatures",
            soilAndPh = "Well-drained sandy loam; pH 6.0–7.5; very intolerant of waterlogged roots",
            waterNeeds = "Deep watering every 7–14 days depending on heat; keep surface dry between waterings",
            harvestSigns = "Skin is fully orange, fruit is heavy with juice, sweet flavor test (oranges do not ripen after picking).",
            careTips = "Always clip fruit from tree with shears so peel is not torn near the stem button.",
        ),

        CropCultivationProfile(
            name = "Lemon",
            scientificName = "Citrus limon",
            category = "Fruit",
            emoji = "🍋",
            totalDurationRange = "6–9 months (bloom to ripe fruit)",
            averageDays = 220,
            difficulty = "Easy to Moderate",
            stages = listOf(
                CropStage(1, "Purple-Tinged Bud Formation", "15–20 days", "New buds show distinctive purple blush before white petals open.", 15, "🌱"),
                CropStage(2, "Evergreen Flowering", "20–30 days", "Sweet aromatic blossoms open; lemons can flower multiple cycles per year.", 35, "🌸"),
                CropStage(3, "Fruit Setting & Expansion", "40–60 days", "Green fruitlets develop characteristic pointed nipple at the blossom end.", 60, "🟢"),
                CropStage(4, "Juice Development & Sizing", "60–90 days", "Internal juice vesicles swell; essential citrus oils concentrate in rind.", 85, "🍋"),
                CropStage(5, "Degreening to Bright Yellow", "30–45 days", "Cooler temperatures break down chlorophyll, transforming rind into vivid sunny yellow.", 100, "✨"),
            ),
            sunlight = "Full Sun (6–8 hours minimum)",
            temperature = "21°C–30°C (70°F–86°F); sensitive to temperatures below -2°C",
            soilAndPh = "Light, fertile, well-draining soil; pH 5.5–6.5",
            waterNeeds = "Moderate; allow top 2–3 inches of soil to dry before watering deeply",
            harvestSigns = "Plump, heavy in hand, vibrant bright yellow with glossy oily rind and high juice content.",
            careTips = "Feed quarterly with a citrus-specific fertilizer containing iron, zinc, and manganese.",
        ),

        CropCultivationProfile(
            name = "Mango",
            scientificName = "Mangifera indica",
            category = "Fruit",
            emoji = "🥭",
            totalDurationRange = "100–150 days (flowering to harvest)",
            averageDays = 120,
            difficulty = "Moderate (Tropical Tree)",
            stages = listOf(
                CropStage(1, "Panicle Emergence", "15–25 days", "Large branched pyramidal flower panicles emerge at branch terminals.", 15, "🌿"),
                CropStage(2, "Blossom & Insect Pollination", "15–20 days", "Thousands of tiny reddish-white flowers pollinated by bees and flies.", 30, "🌸"),
                CropStage(3, "Fruit Set & Pea-Stage Sizing", "25–35 days", "Green fruitlets set; natural shedding leaves 1–3 mangos per panicle.", 55, "🟢"),
                CropStage(4, "Rapid Fruit Development", "35–50 days", "Fruit swells rapidly into characteristic kidney/oval shape with firm white flesh.", 80, "🥭"),
                CropStage(5, "Shoulder Fullness & Maturity", "20–30 days", "Shoulders rise above stem attachment; nose rounds; flesh turns golden yellow.", 100, "✨"),
            ),
            sunlight = "Full Sun (8+ hours daily intense sun)",
            temperature = "24°C–35°C (75°F–95°F); requires warm, dry weather during flowering",
            soilAndPh = "Deep, rich, well-drained loamy soil; pH 5.5–7.5",
            waterNeeds = "Moderate while fruit swells; withhold heavy irrigation leading up to harvest for sweeter fruit",
            harvestSigns = "Shoulders fill out around stem cavity, powdery bloom on skin, distinct sweet mango aroma.",
            careTips = "Bag young fruit on trees to protect against fruit flies and sun scald.",
        ),

        CropCultivationProfile(
            name = "Strawberry",
            scientificName = "Fragaria × ananassa",
            category = "Fruit",
            emoji = "🍓",
            totalDurationRange = "90–120 days (from runner/crown)",
            averageDays = 100,
            difficulty = "Easy",
            stages = listOf(
                CropStage(1, "Crown Awakening & Rooting", "15–20 days", "Dormant bare-root crown establishes roots; trifoliate leaves unfold.", 15, "🌱"),
                CropStage(2, "Foliage & Runner Production", "25–35 days", "Vigorous compact foliage clump develops with strong crowns.", 40, "🌿"),
                CropStage(3, "White Flower Blossom", "15–20 days", "Five-petaled white blossoms with bright golden centers emerge.", 60, "🌼"),
                CropStage(4, "Green to White Berry Sizing", "20–25 days", "Fleshy receptacle expands; achenes (seeds) spread evenly over surface.", 80, "🍓"),
                CropStage(5, "Pigment Surge & Full Red Ripening", "10–15 days", "Anthocyanins flood fruit from apex to stem; aroma and sugar peak.", 100, "🍓"),
            ),
            sunlight = "Full Sun (6–10 hours/day for maximum sweetness)",
            temperature = "15°C–26°C (60°F–80°F); prefers cool roots and sunny canopy",
            soilAndPh = "Rich, organically rich, slightly acidic loam; pH 5.5–6.5",
            waterNeeds = "1–1.5 inches per week; keep root zone moist but mulch keeps berries off wet soil",
            harvestSigns = "Fully red from tip to shoulder with no white neck remaining; pick in cool morning.",
            careTips = "Pinch off first-season runners so plant channels maximum strength into berry development.",
        ),

        CropCultivationProfile(
            name = "Watermelon",
            scientificName = "Citrullus lanatus",
            category = "Fruit",
            emoji = "🍉",
            totalDurationRange = "75–100 days",
            averageDays = 85,
            difficulty = "Easy to Moderate",
            stages = listOf(
                CropStage(1, "Germination", "5–10 days", "Seed needs warm soil (22°C–32°C) to quickly break out of hard seed coat.", 10, "🌱"),
                CropStage(2, "Vine Run & Broad Lobed Leaves", "25–35 days", "Long sprawling vines (up to 3–6 meters) spread across the ground.", 35, "🌿"),
                CropStage(3, "Yellow Blossoms & Pollination", "15–20 days", "Male flowers appear first, followed by female flowers with mini-melons at base.", 55, "🌼"),
                CropStage(4, "Melon Bulking & Stripe Definition", "25–35 days", "Fruit swells rapidly with water and sweet sugars; rind pattern sharpens.", 85, "🍉"),
                CropStage(5, "Ripening & Sugar Peak", "10–15 days", "Flesh turns sweet and crisp; ground spot changes from white to rich creamy yellow.", 100, "🍉"),
            ),
            sunlight = "Full Sun (8–10 hours/day; loves intense heat)",
            temperature = "24°C–32°C (75°F–90°F); growth halts below 15°C",
            soilAndPh = "Sandy loam, warm, deep, very well drained; pH 6.0–6.8",
            waterNeeds = "1–2 inches weekly; reduce watering 7–10 days before harvest to concentrate sugars",
            harvestSigns = "Curled tendril nearest the fruit turns brown/dried; ground spot is creamy yellow; hollow dull thump sound.",
            careTips = "Place a clean piece of cardboard or straw underneath developing melons to prevent rot from moist ground.",
        ),

        CropCultivationProfile(
            name = "Carrot",
            scientificName = "Daucus carota",
            category = "Vegetable",
            emoji = "🥕",
            totalDurationRange = "70–80 days",
            averageDays = 75,
            difficulty = "Easy",
            stages = listOf(
                CropStage(1, "Slow Germination", "10–18 days", "Tiny seeds require continuous surface moisture to germinate.", 15, "🌱"),
                CropStage(2, "Lacy Fern Foliage Emergence", "20–25 days", "Delicate feathery green tops grow upward, photosynthesizing nutrients.", 40, "🌿"),
                CropStage(3, "Taproot Lengthening", "20–25 days", "Thin taproot drills deep into soft soil before expanding outwards.", 70, "🥕"),
                CropStage(4, "Root Bulking & Sugar Storage", "15–20 days", "Carrot thickens and turns bright orange; carotene and sucrose concentrate.", 90, "🥕"),
                CropStage(5, "Crisp Harvest Maturity", "5–10 days", "Shoulders reach 3/4 to 1 inch diameter; crisp sweet crunch.", 100, "🧺"),
            ),
            sunlight = "Full Sun to Light Afternoon Shade (6+ hours)",
            temperature = "13°C–24°C (55°F–75°F); cooler weather produces sweeter carrots",
            soilAndPh = "Deep, rock-free, loose sandy loam; pH 6.0–6.8 (rocks cause forked carrots)",
            waterNeeds = "1 inch per week; consistent moisture prevents roots from cracking",
            harvestSigns = "Top of root (shoulder) visible at soil line is about 3/4 inch thick with rich orange hue.",
            careTips = "Thin seedlings early to 2–3 inches apart so each root has room to grow straight and thick.",
        ),

        CropCultivationProfile(
            name = "Cucumber",
            scientificName = "Cucumis sativus",
            category = "Vegetable",
            emoji = "🥒",
            totalDurationRange = "50–70 days",
            averageDays = 60,
            difficulty = "Easy",
            stages = listOf(
                CropStage(1, "Quick Germination", "4–8 days", "Sprouts vigorously in warm soil (21°C+).", 15, "🌱"),
                CropStage(2, "Vining & Tendril Climbing", "20–25 days", "Vines extend rapidly; curly tendrils grasp trellises or spread along soil.", 40, "🌿"),
                CropStage(3, "Yellow Blossom Flowering", "10–15 days", "Bright golden trumpet flowers attract bees for pollination.", 65, "🌼"),
                CropStage(4, "Rapid Fruit Elongation", "10–15 days", "Cucumbers grow very fast, adding up to an inch per day in warm weather.", 85, "🥒"),
                CropStage(5, "Crisp Harvest Ready", "5–8 days", "Uniform deep green, firm cylindrical shape before seeds enlarge.", 100, "🥒"),
            ),
            sunlight = "Full Sun (6–8 hours/day)",
            temperature = "21°C–30°C (70°F–86°F); frost sensitive",
            soilAndPh = "Warm, compost-rich, fertile soil; pH 6.0–6.8",
            waterNeeds = "1–2 inches weekly; consistent moisture is essential to avoid bitter cucumbers",
            harvestSigns = "Firm, bright green, 6–8 inches long (slicing) or 2–4 inches (pickling); pick before yellowing.",
            careTips = "Grow on a vertical trellis to improve airflow, prevent powdery mildew, and yield straight, clean fruit.",
        ),

        CropCultivationProfile(
            name = "Garlic",
            scientificName = "Allium sativum",
            category = "Vegetable",
            emoji = "🧄",
            totalDurationRange = "180–240 days (clove)",
            averageDays = 210,
            difficulty = "Easy",
            stages = listOf(
                CropStage(1, "Rooting & Clove Sprout", "10–20 days", "Fibrous root anchor spreads downward; emerald green sprout shoots up.", 15, "🌱"),
                CropStage(2, "Vegetative Canopy & Leafing", "60–90 days", "Sturdy upright strap-like leaves emerge to capture spring sunshine.", 40, "🌿"),
                CropStage(3, "Scape Curling & Node Extension", "30–45 days", "Curling flower scape stem spirals (snip off to enlarge bulbs).", 65, "🪱"),
                CropStage(4, "Bulb Swelling & Clove Division", "30–45 days", "Underground bulb expands and divides into distinct plump cloves.", 85, "🧄"),
                CropStage(5, "Lower Foliage Curing & Harvest", "15–20 days", "Bottom 3–4 leaves brown while top leaves stay green; ready to lift.", 100, "🧺"),
            ),
            sunlight = "Full Sun (6–8 hours/day)",
            temperature = "13°C–24°C (55°F–75°F); requires cold vernalization period to split into cloves",
            soilAndPh = "Loose, fertile, rock-free sandy loam rich in organic matter; pH 6.0–7.0",
            waterNeeds = "1 inch per week; cease watering 2–3 weeks before harvest to cure papery wrappers",
            harvestSigns = "Bottom half of foliage turns golden brown and papery; outer clove wrappers feel distinct and tight.",
            careTips = "Plant individual unpeeled cloves pointy end up 2 inches deep in autumn; mulch heavily.",
        ),

        CropCultivationProfile(
            name = "Ginger",
            scientificName = "Zingiber officinale",
            category = "Vegetable",
            emoji = "🫚",
            totalDurationRange = "240–300 days (rhizome)",
            averageDays = 260,
            difficulty = "Moderate",
            stages = listOf(
                CropStage(1, "Rhizome Eye Sprout", "15–30 days", "Green growth buds on rhizome break dormancy and push through soil.", 15, "🌱"),
                CropStage(2, "Reed-like Foliage Canopy", "60–90 days", "Graceful reed-like leafy stalks grow upward (up to 3–4 feet high).", 40, "🌿"),
                CropStage(3, "Subterranean Tillering & Branching", "60–90 days", "New subterranean rhizome fingers branch outwards continuously.", 65, "🫚"),
                CropStage(4, "Rhizome Bulking & Oil Maturation", "45–60 days", "Pungent gingerols and aromatic oils deposit into dense rhizome fingers.", 85, "🫚"),
                CropStage(5, "Leaf Yellowing & Peak Harvest", "15–30 days", "Foliage dies back naturally in cool weather; rhizomes reach peak harvest.", 100, "🧺"),
            ),
            sunlight = "Filtered / Partial Sunlight (2–5 hours; loves warm dappled shade)",
            temperature = "22°C–30°C (72°F–86°F); very sensitive to frost and dry heat",
            soilAndPh = "Rich, moist, loose loamy soil amended with plenty of compost; pH 5.5–6.5",
            waterNeeds = "Consistent moisture; never allow soil to dry out completely, but ensure sharp drainage",
            harvestSigns = "Foliage turns yellow and withers back; rhizomes develop thick tan skin and strong aroma.",
            careTips = "Plant plump rhizome segments with growth eyes pointing upward 1–2 inches deep in warm soil.",
        ),
    )

    /**
     * Find cultivation profile by produce class name or synonym.
     */
    fun getProfile(className: String): CropCultivationProfile? {
        val clean = className.trim().lowercase()
        return CROPS.firstOrNull { crop ->
            crop.name.lowercase() == clean ||
                    clean.contains(crop.name.lowercase()) ||
                    crop.name.lowercase().contains(clean) ||
                    (clean.contains("lehsun") && crop.name == "Garlic") ||
                    (clean.contains("garlic") && crop.name == "Garlic") ||
                    (clean.contains("adrak") && crop.name == "Ginger") ||
                    (clean.contains("ginger") && crop.name == "Ginger") ||
                    (clean.contains("aloo") && crop.name == "Potato") ||
                    (clean.contains("potato") && crop.name == "Potato") ||
                    (clean.contains("pyaz") && crop.name == "Onion") ||
                    (clean.contains("tamatar") && crop.name == "Tomato") ||
                    (clean.contains("kheera") && crop.name == "Cucumber") ||
                    (clean.contains("gajar") && crop.name == "Carrot") ||
                    (clean.contains("kela") && crop.name == "Banana") ||
                    (clean.contains("seb") && crop.name == "Apple") ||
                    (clean.contains("nimbu") && crop.name == "Lemon") ||
                    (clean.contains("brinjal") && crop.name == "Eggplant") ||
                    (clean.contains("capsicum") && crop.name == "Bell Pepper") ||
                    (clean.contains("pepper") && crop.name == "Bell Pepper") ||
                    (clean.contains("citrus") && crop.name == "Orange")
        }
    }

    /**
     * Search and filter crops by text query and category filter ("All", "Vegetables", "Fruits").
     */
    fun searchCrops(query: String, categoryFilter: String = "All"): List<CropCultivationProfile> {
        val q = query.trim().lowercase()
        return CROPS.filter { crop ->
            val matchesCategory = when (categoryFilter.lowercase()) {
                "vegetable", "vegetables" -> crop.category.equals("Vegetable", ignoreCase = true)
                "fruit", "fruits" -> crop.category.equals("Fruit", ignoreCase = true)
                else -> true
            }
            val matchesQuery = q.isEmpty() ||
                    crop.name.lowercase().contains(q) ||
                    crop.scientificName.lowercase().contains(q) ||
                    crop.totalDurationRange.lowercase().contains(q)

            matchesCategory && matchesQuery
        }
    }
}
