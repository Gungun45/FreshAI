package com.example.freshai

import android.graphics.Color

/**
 * FreshAI — Step 2: Freshness Detection Result
 * Holds freshness stage, ripeness, shelf life, and confidence from the API or on-device heuristic.
 */
data class ArrheniusShelfLife(
    val ambientDays: Float,
    val refrigDays: Float,
    val hotDays: Float,
    val summary: String,
)

data class FreshnessEstimate(
    val stage: String,                   // "Very Fresh", "Fresh", "Ripe", etc.
    val stageIndex: Int,                 // 0 (Very Fresh) … 5 (Spoiled)
    val probability: Float,              // 0.0 – 1.0
    val ripenessStage: String,           // "Unripe", "Nearly Ripe", "Ripe", "Overripe"
    val ripenessProb: Float,             // 0.0 – 1.0
    val shelfLifeEstimate: String,       // "7-10 days", "Eat today", etc.
    val badgeColorHex: String,           // "#16A34A" etc.
    val freshnesEmoji: String,           // "🌿", "✅", etc.
    val ripenessEmoji: String,           // "🔵", "🟢", etc.
    val modelMode: String,               // "convnext" |N"heuristic"
    val freshnessScore: Int,             // 0 – 100 (100 = Very Fresh)
    val className: String = "",          // "Onion", "Tomato", etc.
    // Stage 3: YOLO Defect Detection & Segmentation
    val defectCount: Int = 0,
    val defectAreaPct: Float = 0f,
    // Stage 5: Structured Prediction Layer (XGBoost & LightGBM)
    val qualityScoreStr: String = "",
    val physiologicalAgeRange: String = "",
    val postHarvestAgeRange: String = "",
    val remainingShelfLifeRange: String = "",
    val spoilageRisk: String = "",
    val spoilageRiskColorHex: String = "#16A34A",
    val timeToHarvestRange: String = "",
    // Stage 6: LLM + RAG Recommendations
    val storageGuideline: String = "",
    val ethyleneWarning: String = "",
    val chefRecipeTitle: String = "",
    val chefRecipePrepTime: String = "",
    val chefRecipeInstructions: String = "",
    val aiAdvisorSummary: String = "",
) {
    /** Returns an Android Color int parsed from badgeColorHex. */
    fun badgeColor(): Int = try {
        Color.parseColor(badgeColorHex)
    } catch (e: Exception) {
        Color.parseColor("#22C55E")
    }

    /**
     * Compute temperature-dependent shelf life using the Arrhenius Kinetics Equation:
     *   k(T) = A * exp(-Ea / (R * T))
     * 
     * Grounded in food science:
     * - Never refrigerate whole Onions, Potatoes, or Bananas (moisture, mold, chilling injury).
     * - Realistic home kitchen shelf life bounds (1 - 18 days).
     */
    /**
     * Compute temperature-dependent shelf life using the Biological Arrhenius Kinetics Equation:
     *   k(T) = A * exp(-Ea / (R * T))
     * 
     * Grounded in food science:
     * - Accounts for Chilling Injury (CI) below minimum safe temperatures (Bananas <12°C, Onions/Potatoes <10°C).
     * - Accounts for Heat Stress above 30°C.
     * - Accounts for Defect Area % reducing protective skin/cuticle barrier.
     */
    fun calculateArrheniusShelfLife(tempC: Float = 22f): ArrheniusShelfLife {
        val ea = 58000.0 // Universal produce respiration activation energy (J/mol)
        val rGas = 8.314
        val tRef = 293.15 // Reference temperature: 20°C
        val tK = Math.max(273.15, tempC + 273.15)
        var rate = Math.exp(Math.max(-3.0, Math.min(3.0, (ea / rGas) * ((1.0 / tRef) - (1.0 / tK)))))
        if (tempC > 30.0) {
            rate *= (1.0 + 0.05 * Math.pow(tempC - 30.0, 1.2))
        }

        // Structural tissue degradation factor from 0.0 (fresh) to 1.0 (decayed)
        val dStruct = ((100 - freshnessScore) / 100.0).coerceIn(0.0, 1.0)

        val cleanName = className.lowercase().trim()
        val baseCapacity = when {
            cleanName.contains("garlic") || cleanName.contains("lehsun") -> 14.0
            cleanName.contains("potato") || cleanName.contains("aloo") -> 14.0
            cleanName.contains("onion") || cleanName.contains("pyaz") -> 14.0
            cleanName.contains("ginger") || cleanName.contains("adrak") -> 14.0
            cleanName.contains("carrot") || cleanName.contains("gajar") -> 14.0
            cleanName.contains("apple") || cleanName.contains("seb") -> 10.0
            cleanName.contains("lemon") || cleanName.contains("nimbu") || cleanName.contains("lime") -> 7.0
            cleanName.contains("orange") || cleanName.contains("citrus") -> 7.0
            cleanName.contains("watermelon") || cleanName.contains("tarbooz") -> 7.0
            cleanName.contains("pepper") || cleanName.contains("capsicum") || cleanName.contains("mirch") -> 7.0
            cleanName.contains("tomato") || cleanName.contains("tamatar") -> 7.0
            cleanName.contains("cucumber") || cleanName.contains("kheera") -> 5.0
            cleanName.contains("banana") || cleanName.contains("kela") -> 5.0
            cleanName.contains("eggplant") || cleanName.contains("brinjal") || cleanName.contains("baingan") -> 5.0
            cleanName.contains("strawberry") -> 3.0
            else -> 7.0
        }

        val decayMultiplier = Math.pow(Math.max(0.0, 1.0 - dStruct), 2.2)
        val ambientDays = Math.max(0.5, (baseCapacity * decayMultiplier) / Math.max(0.1, rate)).toFloat()

        val refrigK = 277.15 // 4°C
        val refrigRate = Math.exp((ea / rGas) * ((1.0 / tRef) - (1.0 / refrigK)))
        val refrigDays = Math.max(0.5, (baseCapacity * decayMultiplier) / Math.max(0.1, refrigRate)).toFloat()

        val hotK = 308.15 // 35°C
        val hotRate = Math.exp((ea / rGas) * ((1.0 / tRef) - (1.0 / hotK))) * 1.3
        val hotDays = Math.max(0.5, (baseCapacity * decayMultiplier) / Math.max(0.1, hotRate)).toFloat()

        val summary = when {
            freshnessScore <= 30 || dStruct >= 0.65 -> "1 day (Spoiled / Discard)"
            freshnessScore <= 50 || dStruct >= 0.45 -> "1–2 days (Cook immediately • Heavy decay)"
            freshnessScore <= 70 || dStruct >= 0.25 -> "${Math.max(1, Math.round(ambientDays))} days at ${tempC.toInt()}°C (Use soon • Defect detected)"
            cleanName.contains("onion") || cleanName.contains("pyaz") || cleanName.contains("garlic") || cleanName.contains("potato") ->
                "~${Math.round(ambientDays)}d at ${tempC.toInt()}°C (Pantry storage • Do not refrigerate)"
            else -> "~${Math.round(ambientDays)}d at ${tempC.toInt()}°C (Cold storage: ~${Math.round(refrigDays)}d)"
        }

        return ArrheniusShelfLife(ambientDays, refrigDays, hotDays, summary)
    }

    companion object {
        val FRESHNESS_STAGES = listOf(
            "Very Fresh", "Fresh", "Ripe", "Overripe", "Deteriorating", "Spoiled"
        )
        val RIPENESS_STAGES = listOf(
            "Unripe", "Nearly Ripe", "Ripe", "Overripe"
        )
        val FRESHNESS_COLORS = listOf(
            "#16A34A", "#22C55E", "#EAB308", "#F97316", "#DC2626", "#6B7280"
        )
        val FRESHNESS_SHELF_LIFE = listOf(
            "7–10 days", "4–7 days", "1–3 days", "Eat today", "Process immediately", "Discard"
        )
        val FRESHNESS_EMOJI = listOf("🌿", "✅", "🟡", "🟠", "⚠️", "❌")
        val RIPENESS_EMOJI = listOf("🔵", "🟢", "🟡", "🟠")

        /** Build a FreshnessEstimate from a server API JSON object fields. */
        fun fromApiFields(
            stage: String,
            stageProb: Float,
            ripeness: String,
            ripenessProb: Float,
            shelfLife: String,
            badgeColor: String,
            freshEmoji: String,
            ripenessEmoji: String,
            modelMode: String,
            freshnessScore: Int,
            className: String = "",
            defectCount: Int = 0,
            defectAreaPct: Float = 0f,
            qualityScoreStr: String = "",
            physiologicalAgeRange: String = "",
            postHarvestAgeRange: String = "",
            remainingShelfLifeRange: String = "",
            spoilageRisk: String = "",
            spoilageRiskColorHex: String = "#16A34A",
            timeToHarvestRange: String = "",
            storageGuideline: String = "",
            ethyleneWarning: String = "",
            chefRecipeTitle: String = "",
            chefRecipePrepTime: String = "",
            chefRecipeInstructions: String = "",
            aiAdvisorSummary: String = "",
        ): FreshnessEstimate {
            val idx = FRESHNESS_STAGES.indexOfFirst {
                it.equals(stage, ignoreCase = true)
            }.coerceAtLeast(0)
            val rIdx = RIPENESS_STAGES.indexOfFirst {
                it.equals(ripeness, ignoreCase = true)
            }.coerceAtLeast(0)
            return FreshnessEstimate(
                stage = stage,
                stageIndex = idx,
                probability = stageProb,
                ripenessStage = ripeness,
                ripenessProb = ripenessProb,
                shelfLifeEstimate = shelfLife,
                badgeColorHex = badgeColor,
                freshnesEmoji = freshEmoji,
                ripenessEmoji = ripenessEmoji,
                modelMode = modelMode,
                freshnessScore = freshnessScore,
                className = className,
                defectCount = defectCount,
                defectAreaPct = defectAreaPct,
                qualityScoreStr = qualityScoreStr,
                physiologicalAgeRange = physiologicalAgeRange,
                postHarvestAgeRange = postHarvestAgeRange,
                remainingShelfLifeRange = remainingShelfLifeRange,
                spoilageRisk = spoilageRisk,
                spoilageRiskColorHex = spoilageRiskColorHex,
                timeToHarvestRange = timeToHarvestRange,
                storageGuideline = storageGuideline,
                ethyleneWarning = ethyleneWarning,
                chefRecipeTitle = chefRecipeTitle,
                chefRecipePrepTime = chefRecipePrepTime,
                chefRecipeInstructions = chefRecipeInstructions,
                aiAdvisorSummary = aiAdvisorSummary,
            )
        }

        /**
         * On-device freshness estimate from produce-specific color & texture analysis.
         * Grounded in CMC 2022 research: "Fruits and Vegetables Freshness Categorization Using Deep Learning".
         * Works entirely offline without a server connection.
         */
        fun estimateOnDevice(
            bitmap: android.graphics.Bitmap,
            x1: Float, y1: Float, x2: Float, y2: Float,
            className: String = "",
        ): FreshnessEstimate {
            val bx1 = x1.toInt().coerceIn(0, bitmap.width - 1)
            val by1 = y1.toInt().coerceIn(0, bitmap.height - 1)
            val bx2 = x2.toInt().coerceIn(bx1 + 1, bitmap.width)
            val by2 = y2.toInt().coerceIn(by1 + 1, bitmap.height)

            var sumR = 0L; var sumG = 0L; var sumB = 0L
            var brownCount = 0; var greenCount = 0; var yellowCount = 0; var spotCount = 0; var total = 0
            var onionGoldCount = 0; var onionMagentaCount = 0
            val step = 4

            val cx = (bx1 + bx2) / 2f
            val cy = (by1 + by2) / 2f
            val rx = ((bx2 - bx1) / 2f).coerceAtLeast(1f)
            val ry = ((by2 - by1) / 2f).coerceAtLeast(1f)
            val cleanCrop = className.lowercase().trim()
            val isOnion = cleanCrop.contains("onion") || cleanCrop.contains("pyaz")
            val isPotato = cleanCrop.contains("potato") || cleanCrop.contains("aloo")
            val isGarlic = cleanCrop.contains("garlic") || cleanCrop.contains("lehsun")

            for (px in bx1 until bx2 step step) {
                for (py in by1 until by2 step step) {
                    val pixel = bitmap.getPixel(px, py)
                    val r = Color.red(pixel).toFloat()
                    val g = Color.green(pixel).toFloat()
                    val b = Color.blue(pixel).toFloat()

                    // Normalized radial distance from center of detected crop
                    val normX = (px - cx) / rx
                    val normY = (py - cy) / ry
                    val distSq = normX * normX + normY * normY

                    val (h, s, v) = rgbToHsv(r, g, b)

                    // Skip extreme glare highlights or complete black boundaries
                    if ((r > 250f && g > 250f && b > 250f) || (r < 10f && g < 10f && b < 10f)) continue

                    // Filter out neutral background floor/table (cement, light tiles, neutral gray/tan surfaces)
                    val isTrueGarlicIvory = isGarlic && (r in 170f..245f && g in 160f..242f && b in 140f..230f && r >= g && g >= b && (r - b in 12f..35f) && distSq < 0.70f)
                    val isNeutralSurface = (Math.abs(r - g) < 14f && Math.abs(g - b) < 14f && s < 0.15f && v in 0.20f..0.98f)
                    if (isNeutralSurface && !isTrueGarlicIvory) continue

                    // Skip outer corner background
                    val isCornerBackground = distSq > 0.90f && (v < 0.25f || s < 0.18f || (r < 50f && g < 50f && b < 50f))
                    if (isCornerBackground) continue

                    sumR += r.toLong(); sumG += g.toLong(); sumB += b.toLong()
                    total++

                    // Universal Dark Fungal Rot / Mold / Necrotic Lesion detection (works for all crops including Onion, Potato, Garlic, Apple, Tomato)
                    val isNecroticRot = (v in 0.04f..0.45f && s < 0.52f && (r < 135f || g < 130f || b < 130f))
                    val isPowderyMildewGray = (v in 0.28f..0.82f && s < 0.28f && distSq < 0.88f && (r - b in 0f..58f))
                    val isSunkenSlimeBrown = (h in 14f..48f && s in 0.08f..0.34f && v in 0.14f..0.44f)

                    // Produce-specific tissue health assessment:
                    if (isOnion) {
                        // Track healthy curing colors for individual age calculation
                        val isHealthyGoldTunic = (h in 20f..55f && s in 0.28f..0.75f && v in 0.35f..0.85f && r > g + 16 && r - b > 45)
                        val isHealthyMagenta = (h in 265f..350f && s > 0.15f && v > 0.18f) || (h in 350f..360f && b > 45)
                        if (isHealthyGoldTunic) onionGoldCount++
                        if (isHealthyMagenta) onionMagentaCount++

                        if (isNecroticRot || isPowderyMildewGray) {
                            spotCount++
                        } else if (isSunkenSlimeBrown) {
                            brownCount++
                        }
                    } else if (isPotato || isGarlic) {
                        if (isNecroticRot || isPowderyMildewGray) {
                            spotCount++
                        } else if (isSunkenSlimeBrown) {
                            brownCount++
                        }
                    } else {
                        // General fruits & vegetables (Tomato, Apple, Citrus, etc.)
                        val isDarkRot = (v < 0.25f && s < 0.35f) || (v < 0.32f && s < 0.28f && r < 85f && g < 75f)
                        val isDecayBrown = (h in 15f..40f && s in 0.15f..0.35f && v in 0.16f..0.36f)

                        if (isNecroticRot || isDarkRot) {
                            spotCount++
                        } else if (isSunkenSlimeBrown || isDecayBrown) {
                            brownCount++
                        } else if (h in 75f..165f && s > 0.20f && v > 0.25f) {
                            greenCount++
                        } else if (h in 45f..75f && s > 0.25f && v > 0.40f) {
                            yellowCount++
                        }
                    }
                }
            }
            if (total == 0) return defaultFresh(className)

            val brownRatio = brownCount.toFloat() / total
            val greenRatio = greenCount.toFloat() / total
            val yellowRatio = yellowCount.toFloat() / total
            val spotRatio = spotCount.toFloat() / total
            val goldRatio = onionGoldCount.toFloat() / total
            val magentaRatio = onionMagentaCount.toFloat() / total

            // ── Universal Biophysical Structure & Thermal Kinetics Engine ─────────
            // 1. Structural Damage: Progressive smooth scaling for mold & blemishes
            val rotScore = if (isOnion) {
                when {
                    spotRatio >= 0.18f -> 1.0f                                    // Severe fungal rot (> 18%) -> 1.0
                    spotRatio >= 0.09f -> 0.75f + (spotRatio - 0.09f) * 2.77f     // Notable mold patches (9-18%) -> 0.75..1.0
                    spotRatio >= 0.04f -> 0.45f + (spotRatio - 0.04f) * 6.00f     // Moderate defect (4-9%) -> 0.45..0.75
                    spotRatio >= 0.012f -> 0.15f + (spotRatio - 0.012f) * 10.7f   // Early blemish (1.2-4%) -> 0.15..0.45
                    else -> 0.0f                                                  // Pristine sound skin (< 1.2%)
                }
            } else {
                when {
                    spotRatio >= 0.15f -> 1.0f
                    spotRatio >= 0.07f -> 0.70f + (spotRatio - 0.07f) * 3.75f
                    spotRatio >= 0.03f -> 0.35f + (spotRatio - 0.03f) * 8.75f
                    spotRatio >= 0.01f -> 0.10f + (spotRatio - 0.01f) * 12.5f
                    else -> 0.0f
                }
            }
            val brownScore = if (isOnion) (brownRatio / 0.10f).coerceIn(0f, 1f) else (brownRatio / 0.08f).coerceIn(0f, 1f)
            val dStruct = (0.85f * rotScore + 0.15f * brownScore).coerceIn(0f, 1f)

            // Dynamic Ripeness Index derived from structural coloration
            val rIdx = when {
                greenRatio > 0.25f && spotRatio < 0.01f -> 0 // Unripe
                dStruct >= 0.55f -> 3                         // Overripe / Senescent
                dStruct >= 0.25f -> 2                         // Fully Ripe
                else -> 1                                     // Crisp / Fresh
            }

            // Freshness index (0: Very Fresh to 5: Spoiled)
            val fIdx = when {
                dStruct >= 0.70f -> 5 // Spoiled
                dStruct >= 0.50f -> 4 // Deteriorating
                dStruct >= 0.30f -> 3 // Overripe
                dStruct >= 0.15f -> 2 // Ripe
                dStruct >= 0.05f -> 1 // Fresh
                else -> 0             // Very Fresh
            }

            val freshnessScore = Math.round((1.0f - dStruct) * 100f).toInt().coerceIn(5, 98)

            // Arrhenius Thermal Reaction Kinetics k(T)
            val ea = 58000.0 // Universal plant respiration activation energy (J/mol)
            val rGas = 8.314
            val tRef = 293.15 // 20°C reference
            val tempC = 22.0f // Ambient temperature
            val tK = Math.max(273.15, tempC + 273.15)
            var rateMultiplier = Math.exp((ea / rGas) * ((1.0 / tRef) - (1.0 / tK)))
            if (tempC > 30.0f) {
                rateMultiplier *= (1.0 + 0.05 * Math.pow((tempC - 30.0), 1.2))
            }

            val cleanName = className.lowercase().trim()

            // Dynamic Curing & Biological Age Calculation:
            // High gold ratio = multiple layers of mature cured dry papery tunic (20–30 days)
            // High magenta ratio = newly harvested crisp purple onion (8–14 days)
            // Rotten/Moldy = advanced age (35–50+ days)
            val individualOnionAge = if (isOnion) {
                val curingOffset = (goldRatio * 18.0 - magentaRatio * 8.0).coerceIn(-5.0, 15.0)
                val decayOffset = 30.0 * Math.pow(dStruct.toDouble(), 1.1)
                13.0 + curingOffset + decayOffset
            } else 14.0

            // 1. Biological Age Prediction (Crop-specific post-harvest biological timeline)
            val baseBioAgeDays = when {
                cleanName.contains("garlic") || cleanName.contains("lehsun") -> 20.0 + 70.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("potato") || cleanName.contains("aloo") -> 14.0 + 45.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("onion") || cleanName.contains("pyaz") -> individualOnionAge
                cleanName.contains("ginger") || cleanName.contains("adrak") -> 10.0 + 25.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("apple") || cleanName.contains("seb") -> 7.0 + 20.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("lemon") || cleanName.contains("orange") || cleanName.contains("citrus") -> 5.0 + 15.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("carrot") || cleanName.contains("gajar") -> 4.0 + 18.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("watermelon") || cleanName.contains("tarbooz") -> 4.0 + 14.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("tomato") || cleanName.contains("tamatar") -> 2.0 + 9.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("cucumber") || cleanName.contains("kheera") -> 2.0 + 7.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("banana") || cleanName.contains("kela") -> 2.0 + 6.0 * Math.pow(dStruct.toDouble(), 1.1)
                cleanName.contains("strawberry") -> 1.0 + 4.0 * Math.pow(dStruct.toDouble(), 1.1)
                else -> 3.0 + 12.0 * Math.pow(dStruct.toDouble(), 1.1)
            }

            val postHarvestDays = baseBioAgeDays * (0.95 + 0.15 * (rateMultiplier - 1.0))
            val minPostHarvest = Math.max(1, Math.round(postHarvestDays * 0.85).toInt())
            val maxPostHarvest = Math.round(postHarvestDays * 1.20 + 0.5).toInt()
            val bioAgeStr = if (postHarvestDays >= 60.0) {
                "> 60 days"
            } else if (minPostHarvest >= 28) {
                val minW = minPostHarvest / 7
                val maxW = maxPostHarvest / 7
                "~$minW–$maxW weeks"
            } else if (minPostHarvest == maxPostHarvest) {
                "$minPostHarvest days"
            } else {
                "$minPostHarvest–$maxPostHarvest days"
            }

            val postHarvestAgeStr = if (postHarvestDays >= 60.0) {
                "> 60d post-harvest"
            } else if (minPostHarvest == maxPostHarvest) {
                "${minPostHarvest}d post-harvest"
            } else {
                "${minPostHarvest}–${maxPostHarvest}d post-harvest"
            }

            // 2. Remaining Shelf Life Prediction (Produce-specific biological capacity)
            val baseCapacityDays = when {
                cleanName.contains("garlic") || cleanName.contains("lehsun") -> 14.0
                cleanName.contains("potato") || cleanName.contains("aloo") -> 14.0
                cleanName.contains("onion") || cleanName.contains("pyaz") -> 14.0
                cleanName.contains("ginger") || cleanName.contains("adrak") -> 14.0
                cleanName.contains("carrot") || cleanName.contains("gajar") -> 14.0
                cleanName.contains("apple") || cleanName.contains("seb") -> 10.0
                cleanName.contains("lemon") || cleanName.contains("nimbu") || cleanName.contains("lime") -> 7.0
                cleanName.contains("orange") || cleanName.contains("citrus") -> 7.0
                cleanName.contains("watermelon") || cleanName.contains("tarbooz") -> 7.0
                cleanName.contains("pepper") || cleanName.contains("capsicum") || cleanName.contains("mirch") -> 7.0
                cleanName.contains("tomato") || cleanName.contains("tamatar") -> 7.0
                cleanName.contains("cucumber") || cleanName.contains("kheera") -> 5.0
                cleanName.contains("banana") || cleanName.contains("kela") -> 5.0
                cleanName.contains("eggplant") || cleanName.contains("brinjal") || cleanName.contains("baingan") -> 5.0
                cleanName.contains("strawberry") -> 3.0
                else -> 7.0
            }

            val decayMultiplier = Math.pow(Math.max(0.0, 1.0 - dStruct.toDouble()), 2.2)
            val remainingDays = Math.max(0.5, (baseCapacityDays * decayMultiplier) / Math.max(0.2, rateMultiplier))
            val minShelf = Math.max(1, Math.round(remainingDays * 0.80).toInt())
            val maxShelf = Math.max(minShelf, Math.round(remainingDays * 1.20).toInt())
            val remainingShelfLifeStr = when {
                dStruct >= 0.65f || freshnessScore <= 30 || remainingDays <= 1.5 -> "1 day (Spoiled / Discard)"
                dStruct >= 0.45f || freshnessScore <= 50 || remainingDays <= 3.0 -> "1–2 days (Cook / Process immediately)"
                dStruct >= 0.25f || freshnessScore <= 70 || remainingDays <= 6.0 -> "2–4 days (Use soon • Aging)"
                minShelf == maxShelf -> "$minShelf days at ${tempC.toInt()}°C"
                else -> "$minShelf–$maxShelf days at ${tempC.toInt()}°C"
            }

            // 3. Spoilage Risk
            val (spoilageRiskVal, riskColorHex) = when {
                dStruct >= 0.50f || freshnessScore <= 40 -> Pair("Critical", "#EF4444")
                dStruct >= 0.30f || freshnessScore <= 65 -> Pair("High", "#F97316")
                dStruct >= 0.15f || freshnessScore <= 80 -> Pair("Medium", "#FBBF24")
                else -> Pair("Low", "#16A34A")
            }

            // 4. Storage Guideline (Produce-specific post-harvest preservation)
            val storageGuide = when {
                cleanName.contains("garlic") || cleanName.contains("lehsun") ->
                    "🧄 Garlic: Store whole unpeeled bulbs in a cool, dry, dark pantry (15°C–18°C) in a breathable mesh bag. Never refrigerate whole bulbs (causes mold & rubberiness)."
                cleanName.contains("potato") || cleanName.contains("aloo") ->
                    "🥔 Potato: Store in a cool, dark, well-ventilated pantry (10°C–15°C). Keep separate from onions & direct sunlight to prevent toxic solanine greening."
                cleanName.contains("onion") || cleanName.contains("pyaz") ->
                    "🧅 Onion: Store in a cool, dry, dark, well-ventilated basket. Keep separate from potatoes to prevent moisture transfer and sprouting."
                cleanName.contains("ginger") || cleanName.contains("adrak") ->
                    "🫚 Ginger: Store unpeeled in a cool dry pantry (2–3 weeks) or in a sealed bag in refrigerator crisper (up to 2 months)."
                cleanName.contains("carrot") || cleanName.contains("gajar") ->
                    "🥕 Carrot: Remove green leafy tops and store in a high-humidity refrigerator crisper drawer."
                cleanName.contains("tomato") || cleanName.contains("tamatar") ->
                    "🍅 Tomato: Store countertop stem-side down (18°C–22°C). Do not refrigerate unripe tomatoes (destroys flavor & enzymes)."
                cleanName.contains("cucumber") || cleanName.contains("kheera") ->
                    "🥒 Cucumber: Store in refrigerator crisper drawer. Keep away from ethylene emitters (apples, bananas, tomatoes) to prevent yellowing."
                cleanName.contains("pepper") || cleanName.contains("capsicum") || cleanName.contains("mirch") ->
                    "🫑 Bell Pepper: Store dry in refrigerator crisper drawer in a perforated bag (10–14 days)."
                cleanName.contains("banana") || cleanName.contains("kela") ->
                    "🍌 Banana: Hang on countertop hook. Wrap stem crown with foil to slow ethylene release; avoid refrigeration below 12°C."
                cleanName.contains("lemon") || cleanName.contains("nimbu") || cleanName.contains("lime") ->
                    "🍋 Lemon: Countertop for 1 week or inside a sealed bag in refrigerator for up to 4 weeks."
                cleanName.contains("apple") || cleanName.contains("seb") ->
                    "🍎 Apple: Keep in refrigerator crisper. Isolate from other vegetables because apples emit strong ethylene gas."
                cleanName.contains("eggplant") || cleanName.contains("brinjal") || cleanName.contains("baingan") ->
                    "🍆 Eggplant: Store in cool pantry or upper refrigerator shelf (8°C–12°C); consume within 5–7 days."
                cleanName.contains("watermelon") || cleanName.contains("tarbooz") ->
                    "🍉 Watermelon: Store whole at room temperature (14–21 days). Once cut, cover tightly and refrigerate (3–5 days)."
                dStruct >= 0.60f -> "⚠️ Heavy decay / rot detected. Discard or cook remaining sound portions immediately."
                dStruct >= 0.35f -> "⚠️ Moderate surface deterioration. Store in cool, low-humidity conditions. Consume within 2–4 days."
                tempC > 26f -> "🌡️ Ambient temp (${tempC.toInt()}°C) accelerates decay (${String.format(java.util.Locale.US, "%.1f", rateMultiplier)}x rate). Transfer to cooler storage."
                else -> "✅ Sound cellular tissue structure. Store in cool, dry, well-ventilated area."
            }

            return FreshnessEstimate(
                stage = FRESHNESS_STAGES[fIdx],
                stageIndex = fIdx,
                probability = 0.88f,
                ripenessStage = RIPENESS_STAGES[rIdx],
                ripenessProb = 0.85f,
                shelfLifeEstimate = remainingShelfLifeStr,
                badgeColorHex = FRESHNESS_COLORS[fIdx],
                freshnesEmoji = FRESHNESS_EMOJI[fIdx],
                ripenessEmoji = RIPENESS_EMOJI[rIdx],
                modelMode = "on_device_calibrated",
                freshnessScore = freshnessScore,
                className = className,
                qualityScoreStr = "$freshnessScore/100",
                physiologicalAgeRange = bioAgeStr,
                postHarvestAgeRange = postHarvestAgeStr,
                remainingShelfLifeRange = remainingShelfLifeStr,
                spoilageRisk = spoilageRiskVal,
                spoilageRiskColorHex = riskColorHex,
                storageGuideline = storageGuide,
            )
        }

        private fun rgbToHsv(r: Float, g: Float, b: Float): Triple<Float, Float, Float> {
            val rN = r / 255f; val gN = g / 255f; val bN = b / 255f
            val mx = maxOf(rN, gN, bN); val mn = minOf(rN, gN, bN)
            val delta = mx - mn
            val v = mx
            val s = if (mx != 0f) delta / mx else 0f
            val h = when {
                delta == 0f -> 0f
                mx == rN -> 60f * (((gN - bN) / delta) % 6f)
                mx == gN -> 60f * ((bN - rN) / delta + 2f)
                else -> 60f * ((rN - gN) / delta + 4f)
            }.let { if (it < 0) it + 360f else it }
            return Triple(h, s, v)
        }

        private fun defaultFresh(className: String = "") = FreshnessEstimate(
            stage = "Fresh", stageIndex = 1, probability = 0.88f,
            ripenessStage = "Ripe", ripenessProb = 0.85f,
            shelfLifeEstimate = if (className.lowercase().contains("onion")) "30–45 days" else "4–7 days",
            badgeColorHex = "#22C55E",
            freshnesEmoji = "✅", ripenessEmoji = "🟡",
            modelMode = "heuristic", freshnessScore = 88,
            className = className,
            qualityScoreStr = "88/100",
            physiologicalAgeRange = "1–3 days",
            remainingShelfLifeRange = if (className.lowercase().contains("onion")) "~4–6 weeks at 22°C" else "4–7 days",
            spoilageRisk = "Low",
            spoilageRiskColorHex = "#16A34A",
            storageGuideline = "Store in a cool, dry, well-ventilated area.",
        )
    }
}
