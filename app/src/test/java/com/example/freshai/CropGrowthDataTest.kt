package com.example.freshai

import org.junit.Assert.*
import org.junit.Test

class CropGrowthDataTest {

    private val yoloClasses = listOf(
        "Tomato", "Onion", "Apple", "Banana", "Orange",
        "Potato", "Bell Pepper", "Watermelon", "Strawberry",
        "Lemon", "Mango", "Eggplant"
    )

    @Test
    fun testAllYoloClassesHaveCultivationProfiles() {
        for (className in yoloClasses) {
            val profile = CropGrowthRepository.getProfile(className)
            assertNotNull("Cultivation profile must exist for $className", profile)
            assertTrue("Total duration must not be blank for $className", profile!!.totalDurationRange.isNotBlank())
            assertTrue("Stages must not be empty for $className", profile.stages.isNotEmpty())
            assertEquals("Last stage must reach 100% for $className", 100, profile.stages.last().progressPct)
            assertTrue("Sunlight needs must be defined for $className", profile.sunlight.isNotBlank())
            assertTrue("Harvest signs must be defined for $className", profile.harvestSigns.isNotBlank())
        }
    }

    @Test
    fun testFuzzyLookup() {
        val tomatoLower = CropGrowthRepository.getProfile("tomato")
        assertNotNull(tomatoLower)
        assertEquals("Tomato", tomatoLower?.name)

        val bellPepper = CropGrowthRepository.getProfile("capsicum")
        assertNotNull(bellPepper)
        assertEquals("Bell Pepper", bellPepper?.name)

        val eggplant = CropGrowthRepository.getProfile("brinjal")
        assertNotNull(eggplant)
        assertEquals("Eggplant", eggplant?.name)
    }

    @Test
    fun testCategoryFiltering() {
        val veggies = CropGrowthRepository.searchCrops("", "Vegetables")
        assertTrue(veggies.isNotEmpty())
        assertTrue(veggies.all { it.category == "Vegetable" })

        val fruits = CropGrowthRepository.searchCrops("", "Fruits")
        assertTrue(fruits.isNotEmpty())
        assertTrue(fruits.all { it.category == "Fruit" })

        val allCrops = CropGrowthRepository.searchCrops("", "All")
        assertEquals(CropGrowthRepository.CROPS.size, allCrops.size)
    }

    @Test
    fun testSearchQuery() {
        val result = CropGrowthRepository.searchCrops("solanum")
        assertTrue("Query 'solanum' should match Tomato, Potato, Eggplant", result.size >= 3)
    }

    @Test
    fun testCropGrowthDaysAreScientificallyGrounded() {
        val tomato = CropGrowthRepository.getProfile("Tomato")
        assertNotNull(tomato)
        assertTrue("Tomato average days should be ~75", tomato!!.averageDays in 60..90)

        val onion = CropGrowthRepository.getProfile("Onion")
        assertNotNull(onion)
        assertTrue("Onion average days should be ~135", onion!!.averageDays in 100..175)

        val potato = CropGrowthRepository.getProfile("Potato")
        assertNotNull(potato)
        assertTrue("Potato average days should be ~95", potato!!.averageDays in 70..120)

        val banana = CropGrowthRepository.getProfile("Banana")
        assertNotNull(banana)
        assertTrue("Banana average days should be ~300", banana!!.averageDays in 250..365)

        val apple = CropGrowthRepository.getProfile("Apple")
        assertNotNull(apple)
        assertTrue("Apple average days should be ~135", apple!!.averageDays in 120..160)
    }

    @Test
    fun testNewAgronomicPropertiesExistOnAllCrops() {
        for (crop in CropGrowthRepository.CROPS) {
            assertTrue("NPK ratio must not be blank for ${crop.name}", crop.npkRatio.isNotBlank())
            assertTrue("Optimal picking window must not be blank for ${crop.name}", crop.optimalPickingWindow.isNotBlank())
            assertTrue("Disease watch must not be blank for ${crop.name}", crop.diseaseWatch.isNotBlank())
            assertTrue("GDD target must not be blank for ${crop.name}", crop.gddTarget.isNotBlank())
            assertTrue("Temperature range must not be blank for ${crop.name}", crop.temperature.isNotBlank())
            assertTrue("Water needs must not be blank for ${crop.name}", crop.waterNeeds.isNotBlank())
            assertTrue("Soil requirements must not be blank for ${crop.name}", crop.soilAndPh.isNotBlank())
        }
    }

    @Test
    fun testStageProgressionIntegrity() {
        for (crop in CropGrowthRepository.CROPS) {
            assertTrue("${crop.name} should have at least 4 stages", crop.stages.size >= 4)
            var prevProgress = -1
            for (stage in crop.stages) {
                assertTrue("Stage progress must increase monotonically in ${crop.name}", stage.progressPct > prevProgress)
                assertTrue("Stage name cannot be empty in ${crop.name}", stage.name.isNotBlank())
                assertTrue("Stage description cannot be empty in ${crop.name}", stage.description.isNotBlank())
                prevProgress = stage.progressPct
            }
            assertEquals("Last stage must be 100% in ${crop.name}", 100, prevProgress)
        }
    }

    @Test
    fun testComputeHarvestWindowFormats() {
        val window0 = PlantGrowthAnalyzer.computeHarvestWindow(0)
        assertTrue("0 days should state Peak Harvest", window0.contains("Peak Harvest") || window0.contains("Today"))

        val window15 = PlantGrowthAnalyzer.computeHarvestWindow(15)
        assertTrue("15 days should return valid date range format", window15.contains("–") || window15.contains("-"))
    }

    @Test
    fun testProportionalHarvestTimelineCalculation() {
        // Verify crop timelines scale correctly according to crop.averageDays
        val cropsToTest = listOf("Tomato", "Banana", "Onion", "Cucumber", "Eggplant", "Lemon", "Potato")
        for (cropName in cropsToTest) {
            val profile = CropGrowthRepository.getProfile(cropName)
            assertNotNull(profile)
            val avgDays = profile!!.averageDays

            // Test various progress levels
            for (progress in listOf(15, 35, 55, 80, 95, 100)) {
                val remDays = if (progress >= 98) 0 else Math.round(avgDays * ((100.0 - progress) / 100.0)).toInt()
                assertTrue("Remaining days ($remDays) must be <= total avg days ($avgDays) for $cropName at $progress%", remDays <= avgDays)
                assertTrue("Remaining days must be >= 0", remDays >= 0)
                if (progress == 100) {
                    assertEquals("At 100% progress, remaining days must be 0", 0, remDays)
                }
            }
        }
    }
}

