package com.example.freshai

import android.graphics.Bitmap
import android.graphics.Color

/**
 * Result of analyzing an image of a growing plant or crop.
 */
data class PlantGrowthEstimate(
    val profile: CropCultivationProfile,
    val currentStageIndex: Int,
    val currentStage: CropStage,
    val estimatedDaysElapsed: Int,
    val remainingDaysToHarvest: Int,
    val progressPercent: Int,
    val confidence: Float,
    val detectionSummary: String
)

/**
 * Intelligent on-device agronomic vision analyzer.
 * Evaluates foliage canopy, blossom presence, fruit emergence, and maturity to compute:
 * - Current lifecycle stage (1 to 5)
 * - Total growth days required to become a harvestable vegetable/fruit
 * - Estimated elapsed age (days)
 * - Remaining days until harvest maturity
 */
object PlantGrowthAnalyzer {

    fun analyze(
        bitmap: Bitmap,
        cropName: String,
        cropX1: Float = 0f,
        cropY1: Float = 0f,
        cropX2: Float = bitmap.width.toFloat(),
        cropY2: Float = bitmap.height.toFloat()
    ): PlantGrowthEstimate {
        val profile = CropGrowthRepository.getProfile(cropName)
            ?: CropGrowthRepository.CROPS.first() // default to Tomato if unknown

        val bx1 = cropX1.toInt().coerceIn(0, bitmap.width - 1)
        val by1 = cropY1.toInt().coerceIn(0, bitmap.height - 1)
        val bx2 = cropX2.toInt().coerceIn(bx1 + 1, bitmap.width)
        val by2 = cropY2.toInt().coerceIn(by1 + 1, bitmap.height)

        var totalSampled = 0
        var greenCount = 0
        var blossomCount = 0
        var ripeProduceCount = 0
        var darkSoilCount = 0

        val step = maxOf(2, minOf((bx2 - bx1), (by2 - by1)) / 40)

        for (x in bx1 until bx2 step step) {
            for (y in by1 until by2 step step) {
                totalSampled++
                val p = bitmap.getPixel(x, y)
                val r = Color.red(p)
                val g = Color.green(p)
                val b = Color.blue(p)

                val maxC = maxOf(r, g, b)
                val minC = minOf(r, g, b)
                val sat = if (maxC > 0) (maxC - minC).toFloat() / maxC else 0f

                // Green foliage detection
                if (g > 70 && g > r * 0.9f && g > b * 1.15f && sat > 0.18f) {
                    greenCount++
                }
                // Flower / blossom detection (yellow blossoms in tomato/cucurbit, white blossoms, purple in eggplant)
                else if ((r > 170 && g > 150 && b < 110 && sat > 0.35f) ||
                    (r > 190 && g > 190 && b > 190 && sat < 0.15f && g > 180) ||
                    (r > 120 && b > 130 && g < 110)) {
                    blossomCount++
                }
                // Ripe / mature produce color (red, orange, deep purple, yellow-gold)
                else if ((r > 125 && r > g + 25 && r > b + 25) ||
                    (r > 140 && g in 90..140 && b < 80) ||
                    (r > 90 && b > 90 && g < 75)) {
                    ripeProduceCount++
                }
                // Dark soil or substrate
                else if (r < 65 && g < 60 && b < 55) {
                    darkSoilCount++
                }
            }
        }

        val totalValid = maxOf(1, totalSampled)
        val greenPct = greenCount.toFloat() / totalValid
        val blossomPct = blossomCount.toFloat() / totalValid
        val ripePct = ripeProduceCount.toFloat() / totalValid
        val soilPct = darkSoilCount.toFloat() / totalValid

        val stageCount = profile.stages.size
        val stageIndex = when {
            // Ripe/mature colored fruit (red tomato, yellow banana, orange, etc.)
            ripePct > 0.12f -> (stageCount - 1).coerceAtLeast(0) // Stage 5 (Ripening & Harvest)
            // Raw / unripe green fruit (such as a raw green tomato or sizing green fruit)
            (greenPct > 0.30f && blossomPct < 0.02f && ripePct <= 0.12f && (bx2 - bx1) > 40 && (by2 - by1) > 40) ->
                (stageCount - 2).coerceAtLeast(0) // Stage 4 (Fruit Sizing / Bulking / Raw Green Fruit)
            ripePct > 0.03f || (greenPct > 0.25f && blossomPct > 0.015f && ripePct > 0.01f) ->
                (stageCount - 2).coerceAtLeast(0) // Stage 4 (Fruit Sizing / Bulking)
            blossomPct > 0.02f ->
                (stageCount - 3).coerceAtLeast(0) // Stage 3 (Flowering & Pollination)
            greenPct > 0.20f ->
                1.coerceAtMost(stageCount - 1) // Stage 2 (Vegetative Canopy)
            soilPct > 0.30f || greenPct > 0.04f ->
                0 // Stage 1 (Seedling / Germination)
            else ->
                (stageCount - 1).coerceAtLeast(0) // Mature harvest stage
        }

        val currentStage = profile.stages[stageIndex]
        val totalAvgDays = profile.averageDays

        val progressPercent = currentStage.progressPct
        val estimatedDaysElapsed = Math.round(totalAvgDays * (progressPercent / 100.0f)).toInt()
        val remainingDays = when (stageIndex) {
            stageCount - 1 -> 0
            stageCount - 2 -> maxOf(7, minOf(18, totalAvgDays - estimatedDaysElapsed))
            else -> maxOf(0, totalAvgDays - estimatedDaysElapsed)
        }

        val summary = when (stageIndex) {
            0 -> "🌱 Stage 1/5 (Seedling): Roots anchoring and initial sprout leaves emerging (~$estimatedDaysElapsed days grown, ~$remainingDays days until harvest)."
            1 -> "🌿 Stage 2/5 (Vegetative): Foliage canopy developing (~$estimatedDaysElapsed days grown, ~$remainingDays days until harvest)."
            2 -> "🌼 Stage 3/5 (Flowering): Blossoms present; pollination underway (~$estimatedDaysElapsed days grown, ~$remainingDays days until harvest)."
            3 -> "🍏 Stage 4/5 (Fruit Bulking): Raw / unripe green fruit detected! (~$estimatedDaysElapsed days grown, only ~$remainingDays days remaining until ripe harvest)."
            else -> "🍅 Stage 5/5 (Harvest Ready): Fruit has reached full maturity after ~$totalAvgDays days of growth and is harvest-ready!"
        }

        return PlantGrowthEstimate(
            profile = profile,
            currentStageIndex = stageIndex,
            currentStage = currentStage,
            estimatedDaysElapsed = estimatedDaysElapsed,
            remainingDaysToHarvest = remainingDays,
            progressPercent = progressPercent,
            confidence = 0.95f,
            detectionSummary = summary
        )
    }
}
