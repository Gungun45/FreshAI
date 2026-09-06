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
}
