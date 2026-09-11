package com.example.freshai

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.tabs.TabLayout
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var tabLayout: TabLayout
    private lateinit var studioView: ScrollView
    private lateinit var progressBar: ProgressBar

    // Detection Studio elements
    private lateinit var ivPreview: ImageView
    private lateinit var boxOverlay: BoundingBoxOverlayView

    private lateinit var placeholderLayout: LinearLayout
    private lateinit var tvPlaceholderEmoji: TextView
    private lateinit var tvPlaceholderTitle: TextView
    private lateinit var tvPlaceholderSub: TextView
    private lateinit var btnCamera: MaterialButton
    private lateinit var btnGallery: MaterialButton

    // Quick Crop Selector
    private lateinit var layoutCropSelector: LinearLayout
    private lateinit var chipGroupCrops: ChipGroup
    private var selectedCropName: String? = null
    private var isSyncingCropChips = false

    // Studio Analytics KPI elements
    private lateinit var kpiRow: LinearLayout
    private lateinit var tvKpiTotal: TextView
    private lateinit var tvKpiClasses: TextView
    private lateinit var tvKpiConf: TextView

    // Freshness & ML Intelligence elements
    private lateinit var cardFreshness: com.google.android.material.card.MaterialCardView
    private lateinit var freshnessItemsContainer: LinearLayout
    private lateinit var layoutOverallFreshnessSummary: LinearLayout
    private lateinit var tvFreshnessModelMode: TextView
    private lateinit var tvAnalysisCardTitle: TextView
    private lateinit var tvOverallRatingLabel: TextView
    private lateinit var tvOverallFreshness: TextView
    private lateinit var progressFreshnessOverall: android.widget.ProgressBar

    // Scan Mode UI
    private lateinit var tvStudioBadge: TextView
    private lateinit var tvScanModeHint: TextView
    private var isCropGrowthMode = false
    private var lastAnalyzedBitmap: Bitmap? = null
    private var lastDetectedItems = listOf<DetectedItem>()
    private var lastEstimates = listOf<FreshnessEstimate>()

    private lateinit var btnCropLibrary: MaterialButton
    private lateinit var btnPlantDoctor: MaterialButton
    private lateinit var btnHarvestTimeline: MaterialButton
    private lateinit var btnExportReport: MaterialButton

    private lateinit var btnConfigIp: MaterialButton

    // Real-Time Hardware Sensor Integration (Light & Accelerometer)
    private lateinit var layoutSensorHud: LinearLayout
    private lateinit var tvSensorLux: TextView
    private lateinit var tvSensorTemp: TextView
    private lateinit var tvSensorStability: TextView
    private var sensorManager: android.hardware.SensorManager? = null
    private var lightSensor: android.hardware.Sensor? = null
    private var accelSensor: android.hardware.Sensor? = null
    private var currentLuxValue: Float = 480f
    private var isDeviceStable: Boolean = true

    private var baseHost = "127.0.0.1"
    private var apiUrl = "http://127.0.0.1:8088/api/detect"
    private var photoFileUri: Uri? = null

    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var onDeviceDetector: YoloOnnxDetector

    // Runtime Camera Permission Launcher
    private val requestCameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            launchCamera()
        } else {
            Toast.makeText(
                this,
                "Camera permission is required to capture produce photos. Please allow it.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    // Camera Capture Launcher with FileProvider URI
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success && photoFileUri != null) {
            try {
                val bitmap = decodeSampledBitmapFromUri(photoFileUri!!)
                if (bitmap != null) {
                    processAndAnalyzeBitmap(bitmap, "Camera Photo")
                } else {
                    Toast.makeText(this, "Could not load captured photo", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "Error processing camera photo: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Gallery Picker Launcher
    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        if (uri != null) {
            try {
                val bitmap = decodeSampledBitmapFromUri(uri)
                if (bitmap != null) {
                    processAndAnalyzeBitmap(bitmap, "Gallery Image")
                } else {
                    Toast.makeText(this, "Could not load image from gallery", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Toast.makeText(this, "Failed to load image: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Seamless green status bar matching top navbar
        window.statusBarColor = ContextCompat.getColor(this, R.color.fresh_primary)
        androidx.core.view.WindowCompat.getInsetsController(window, window.decorView)?.isAppearanceLightStatusBars = false

        setContentView(R.layout.activity_main)

        loadSavedUrls()
        bindViews()
        initHardwareSensors()
        setupListeners()
        onDeviceDetector = YoloOnnxDetector(this)
    }

    private fun initHardwareSensors() {
        try {
            sensorManager = getSystemService(Context.SENSOR_SERVICE) as? android.hardware.SensorManager
            lightSensor = sensorManager?.getDefaultSensor(android.hardware.Sensor.TYPE_LIGHT)
            accelSensor = sensorManager?.getDefaultSensor(android.hardware.Sensor.TYPE_ACCELEROMETER)
        } catch (e: Exception) {
            // Graceful fallback for devices/emulators without sensors
        }
    }

    private fun loadSavedUrls() {
        val prefs = getSharedPreferences("freshai_prefs", Context.MODE_PRIVATE)
        baseHost = prefs.getString("server_host", "127.0.0.1") ?: "127.0.0.1"
        apiUrl = "http://$baseHost:8088/api/detect"
    }

    private fun saveHost(host: String) {
        baseHost = host
        apiUrl = "http://$baseHost:8088/api/detect"
        getSharedPreferences("freshai_prefs", Context.MODE_PRIVATE)
            .edit()
            .putString("server_host", baseHost)
            .apply()
    }

    private fun bindViews() {
        tabLayout = findViewById(R.id.tabLayout)
        studioView = findViewById(R.id.studioView)
        progressBar = findViewById(R.id.progressBar)

        tvStudioBadge = findViewById(R.id.tvStudioBadge)
        tvScanModeHint = findViewById(R.id.tvScanModeHint)

        ivPreview = findViewById(R.id.ivPreview)
        boxOverlay = findViewById(R.id.boxOverlay)
        placeholderLayout = findViewById(R.id.placeholderLayout)
        tvPlaceholderEmoji = findViewById(R.id.tvPlaceholderEmoji)
        tvPlaceholderTitle = findViewById(R.id.tvPlaceholderTitle)
        tvPlaceholderSub = findViewById(R.id.tvPlaceholderSub)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        btnCropLibrary = findViewById(R.id.btnCropLibrary)
        btnPlantDoctor = findViewById(R.id.btnPlantDoctor)
        btnHarvestTimeline = findViewById(R.id.btnHarvestTimeline)
        btnExportReport = findViewById(R.id.btnExportReport)

        // Sensor HUD Views
        layoutSensorHud = findViewById(R.id.layoutSensorHud)
        tvSensorLux = findViewById(R.id.tvSensorLux)
        tvSensorTemp = findViewById(R.id.tvSensorTemp)
        tvSensorStability = findViewById(R.id.tvSensorStability)

        layoutCropSelector = findViewById(R.id.layoutCropSelector)
        chipGroupCrops = findViewById(R.id.chipGroupCrops)
        setupCropChips()

        kpiRow = findViewById(R.id.kpiRow)
        tvKpiTotal = findViewById(R.id.tvKpiTotal)
        tvKpiClasses = findViewById(R.id.tvKpiClasses)
        tvKpiConf = findViewById(R.id.tvKpiConf)
        cardFreshness = findViewById(R.id.cardFreshness)
        freshnessItemsContainer = findViewById(R.id.freshnessItemsContainer)
        layoutOverallFreshnessSummary = findViewById(R.id.layoutOverallFreshnessSummary)
        tvFreshnessModelMode = findViewById(R.id.tvFreshnessModelMode)
        tvAnalysisCardTitle = findViewById(R.id.tvAnalysisCardTitle)
        tvOverallRatingLabel = findViewById(R.id.tvOverallRatingLabel)
        tvOverallFreshness = findViewById(R.id.tvOverallFreshness)
        progressFreshnessOverall = findViewById(R.id.progressFreshnessOverall)

        btnConfigIp = findViewById(R.id.btnConfigIp)
    }

    private fun setupCropChips() {
        chipGroupCrops.removeAllViews()
        for (crop in CropGrowthRepository.CROPS) {
            val chip = Chip(this).apply {
                text = "${crop.emoji} ${crop.name}"
                isCheckable = true
                isClickable = true
                textSize = 11.5f
                setChipBackgroundColorResource(android.R.color.transparent)
                chipStrokeWidth = dpToPx(1).toFloat()
                chipStrokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#CBD5E1"))
                setTextColor(Color.parseColor("#334155"))
                isCheckedIconVisible = true

                setOnCheckedChangeListener { _, isChecked ->
                    if (isSyncingCropChips) return@setOnCheckedChangeListener
                    if (isChecked) {
                        selectedCropName = crop.name
                        chipStrokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#15803D"))
                        setTextColor(Color.parseColor("#15803D"))
                        updateFreshnessUI(lastDetectedItems, lastEstimates)
                    } else {
                        chipStrokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#CBD5E1"))
                        setTextColor(Color.parseColor("#334155"))
                    }
                }
            }
            chipGroupCrops.addView(chip)
        }
    }

    private fun clearCurrentScan() {
        lastAnalyzedBitmap = null
        lastDetectedItems = emptyList()
        lastEstimates = emptyList()
        selectedCropName = null
        ivPreview.setImageBitmap(null)
        boxOverlay.clear()
        placeholderLayout.visibility = View.VISIBLE
        kpiRow.visibility = View.GONE
        cardFreshness.visibility = View.GONE
        freshnessItemsContainer.removeAllViews()

        isSyncingCropChips = true
        try {
            for (i in 0 until chipGroupCrops.childCount) {
                val chip = chipGroupCrops.getChildAt(i) as? Chip ?: continue
                chip.isChecked = false
                chip.chipStrokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#CBD5E1"))
                chip.setTextColor(Color.parseColor("#334155"))
            }
        } finally {
            isSyncingCropChips = false
        }
    }

    private fun setupListeners() {
        tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                clearCurrentScan()
                when (tab?.position) {
                    0 -> {
                        isCropGrowthMode = false
                        layoutCropSelector.visibility = View.GONE
                        tvStudioBadge.text = "🍎 PRODUCE QUALITY & SHELF-LIFE SCANNER"
                        tvScanModeHint.text = "💡 Point camera or pick photo to evaluate freshness %, quality score, remaining shelf life, and storage strategy."
                        btnCamera.text = "📷 Camera"
                        tvPlaceholderEmoji.text = "🍅 🧄 🥔 🥒"
                        tvPlaceholderTitle.text = "Capture or Choose Produce / Plant"
                        tvPlaceholderSub.text = "AI detects produce, defect %, shelf life & growth stage"
                        tvAnalysisCardTitle.text = "🍃 Analysis & Insights"
                        tvOverallRatingLabel.text = "Overall Freshness"
                    }
                    1 -> {
                        isCropGrowthMode = true
                        layoutCropSelector.visibility = View.VISIBLE
                        tvStudioBadge.text = "🌱 CROP & PLANT GROWTH SCANNER"
                        tvScanModeHint.text = "🌱 Point camera or pick photo of plant/crop to track growing days, stage, and days until harvest."
                        btnCamera.text = "📷 Scan Plant"
                        tvPlaceholderEmoji.text = "🌱 🌿 🪴 🌾"
                        tvPlaceholderTitle.text = "Capture or Choose Plant Image"
                        tvPlaceholderSub.text = "AI determines cultivation stage, growth timeline & days to harvest"
                        tvAnalysisCardTitle.text = "🌱 Plant Growth & Harvest Insights"
                        tvOverallRatingLabel.text = "Harvest Readiness"
                    }
                }
            }

            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })

        btnCamera.setOnClickListener {
            checkAndLaunchCamera()
        }

        btnGallery.setOnClickListener {
            try {
                galleryLauncher.launch("image/*")
            } catch (e: Exception) {
                Toast.makeText(this, "Could not open gallery: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

        btnCropLibrary.setOnClickListener { showCropLibraryDialog() }
        btnPlantDoctor.setOnClickListener { showPlantDoctorDialog() }
        btnHarvestTimeline.setOnClickListener { showHarvestTimelineDialog() }
        btnExportReport.setOnClickListener { shareInspectionReport() }

        btnConfigIp.setOnClickListener { showIpConfigDialog() }
    }

    private fun checkAndLaunchCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            launchCamera()
        } else {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCamera() {
        try {
            val photoFile = File(cacheDir, "camera_freshai.jpg")
            if (photoFile.exists()) photoFile.delete()
            photoFile.createNewFile()

            photoFileUri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                photoFile
            )
            takePictureLauncher.launch(photoFileUri!!)
        } catch (e: Exception) {
            Toast.makeText(this, "Could not launch camera: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
        }
    }

    private fun decodeSampledBitmapFromUri(uri: Uri, reqWidth: Int = 1000, reqHeight: Int = 1000): Bitmap? {
        return try {
            val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            }

            var inSampleSize = 1
            if (options.outHeight > reqHeight || options.outWidth > reqWidth) {
                val halfHeight = options.outHeight / 2
                val halfWidth = options.outWidth / 2
                while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                    inSampleSize *= 2
                }
            }

            options.inJustDecodeBounds = false
            options.inSampleSize = inSampleSize
            val decoded = contentResolver.openInputStream(uri)?.use { stream ->
                BitmapFactory.decodeStream(stream, null, options)
            } ?: return null

            // Read EXIF orientation to keep camera photo upright
            val orientation = try {
                contentResolver.openInputStream(uri)?.use { stream ->
                    val exif = android.media.ExifInterface(stream)
                    exif.getAttributeInt(android.media.ExifInterface.TAG_ORIENTATION, android.media.ExifInterface.ORIENTATION_NORMAL)
                } ?: android.media.ExifInterface.ORIENTATION_NORMAL
            } catch (e: Exception) {
                android.media.ExifInterface.ORIENTATION_NORMAL
            }

            val matrix = android.graphics.Matrix()
            when (orientation) {
                android.media.ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                android.media.ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                android.media.ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                else -> return decoded
            }

            Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
        } catch (e: Exception) {
            null
        }
    }

    private fun showIpConfigDialog() {
        val input = EditText(this).apply {
            setText(baseHost)
            hint = "127.0.0.1 or 192.168.x.x"
            setPadding(40, 30, 40, 30)
        }

        AlertDialog.Builder(this)
            .setTitle("⚙️ FreshAI Backend Host IP")
            .setMessage("Connected over USB: 127.0.0.1 (Default)\nConnected over WiFi: enter your PC's IP (e.g. 192.168.1.100)")
            .setView(input)
            .setPositiveButton("Save & Connect") { _, _ ->
                val newHost = input.text.toString().trim()
                if (newHost.isNotEmpty()) {
                    saveHost(newHost)
                    Toast.makeText(this, "Saved host: $baseHost", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ==========================================
    // Real YOLO Produce Detection Engine
    // ==========================================

    private fun processAndAnalyzeBitmap(bitmap: Bitmap, sourceName: String) {
        lastAnalyzedBitmap = bitmap
        progressBar.visibility = View.VISIBLE
        placeholderLayout.visibility = View.GONE
        ivPreview.setImageBitmap(bitmap)
        boxOverlay.clear()

        // Hybrid Inference Pipeline:
        // 1. Attempts to run against Python API Server (returns YOLO + ConvNeXt-Tiny neural predictions)
        // 2. Falls back to on-device YOLO ONNX + On-device Freshness Analyzer if offline
        executor.execute {
            var (detectedItems, freshnessEstimates) = runYoloInferenceOnServer(bitmap)

            if (detectedItems.isEmpty()) {
                // Server offline -> Run locally on Android device with ONNX model
                val rawItems = onDeviceDetector.detect(bitmap, confidence = 0.14f)
                val imgW = bitmap.width.toFloat()
                val imgH = bitmap.height.toFloat()
                detectedItems = rawItems.filter { item ->
                    val boxW = item.x2 - item.x1
                    val boxH = item.y2 - item.y1
                    val areaPct = (boxW * boxH) / (imgW * imgH)
                    val isTinyEdgeArtifact = areaPct < 0.015f && (item.x1 <= 8f || item.y1 <= 8f || item.x2 >= imgW - 8f || item.y2 >= imgH - 8f)
                    !isTinyEdgeArtifact
                }
                freshnessEstimates = detectedItems.map { item ->
                    FreshnessEstimate.estimateOnDevice(bitmap, item.x1, item.y1, item.x2, item.y2, item.className)
                }
            }

            // High-recall fallback: If YOLO missed the item, run on-device color & morphology detector
            if (detectedItems.isEmpty()) {
                detectedItems = runOnDeviceProduceDetection(bitmap)
                freshnessEstimates = detectedItems.map { item ->
                    FreshnessEstimate.estimateOnDevice(bitmap, item.x1, item.y1, item.x2, item.y2, item.className)
                }
            }

            // In Plant Growth mode, if user selected a crop chip or wants to scan the plant, ensure we have a detection item
            if (isCropGrowthMode && detectedItems.isEmpty()) {
                val chosenCrop = selectedCropName ?: "Tomato"
                val defaultBox = DetectedItem(chosenCrop, 0.95f, 0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
                detectedItems = listOf(defaultBox)
                freshnessEstimates = listOf(
                    FreshnessEstimate.estimateOnDevice(bitmap, 0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat(), chosenCrop)
                )
            }

            // Client-side containment & duplicate suppression
            val (cleanItems, cleanEstimates) = suppressDuplicateAndNestedItems(detectedItems, freshnessEstimates)
            detectedItems = cleanItems.map { item -> expandAndRefineBoundingBox(item, bitmap) }
            freshnessEstimates = cleanEstimates

            mainHandler.post {
                progressBar.visibility = View.GONE
                boxOverlay.setDetections(detectedItems, bitmap.width, bitmap.height)
                updateAnalyticsUI(detectedItems, bitmap.width, bitmap.height)
                lastDetectedItems = detectedItems
                lastEstimates = freshnessEstimates
                updateFreshnessUI(detectedItems, freshnessEstimates)

                val summary = if (isCropGrowthMode) {
                    val cropName = selectedCropName ?: detectedItems.firstOrNull()?.className ?: "Tomato"
                    val p = CropGrowthRepository.getProfile(cropName) ?: CropGrowthRepository.CROPS.first()
                    val item = detectedItems.firstOrNull()
                    val est = PlantGrowthAnalyzer.analyze(
                        bitmap,
                        p.name,
                        item?.x1 ?: 0f, item?.y1 ?: 0f,
                        item?.x2 ?: bitmap.width.toFloat(), item?.y2 ?: bitmap.height.toFloat()
                    )
                    val rem = est.remainingDaysToHarvest
                    val remStr = if (rem <= 0) "Harvest Ready NOW!" else "Ready in ~${rem} days"
                    val stgName = est.currentStage.name
                    "🌱 ${p.name}: $remStr ($stgName • ${est.progressPercent}% Matured)"
                } else if (detectedItems.isNotEmpty()) {
                    val f = freshnessEstimates.firstOrNull()
                    val freshnessInfo = if (f != null) " • ${f.freshnesEmoji} ${f.stage}" else ""
                    "${detectedItems.size} item(s): ${detectedItems[0].className} (${(detectedItems[0].confidence * 100).toInt()}%)$freshnessInfo"
                } else {
                    "No produce items detected"
                }
                Toast.makeText(this@MainActivity, "🌿 FreshAI: $summary", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun runYoloInferenceOnServer(bitmap: Bitmap): Pair<List<DetectedItem>, List<FreshnessEstimate>> {
        val items = mutableListOf<DetectedItem>()
        val estimates = mutableListOf<FreshnessEstimate>()
        var connection: HttpURLConnection? = null

        try {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, 85, baos)
            val imageBytes = baos.toByteArray()

            val url = URL(apiUrl)
            connection = url.openConnection() as HttpURLConnection
            connection.requestMethod = "POST"
            connection.connectTimeout = 3000
            connection.readTimeout = 6000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "image/jpeg")
            connection.setRequestProperty("Content-Length", imageBytes.size.toString())

            val os: OutputStream = connection.outputStream
            os.write(imageBytes)
            os.flush()
            os.close()

            val responseCode = connection.responseCode
            if (responseCode == 200) {
                val responseStr = connection.inputStream.bufferedReader().use { it.readText() }
                val json = JSONObject(responseStr)
                val detectionsArray = json.optJSONArray("detections")
                if (detectionsArray != null) {
                    for (i in 0 until detectionsArray.length()) {
                        val d = detectionsArray.getJSONObject(i)
                        val rawCls = d.optString("class_name", "Produce")
                        val conf = d.optDouble("confidence", 0.90).toFloat()

                        val bboxArr = d.optJSONArray("bbox_xyxy")
                        val x1 = bboxArr?.optDouble(0, 0.0)?.toFloat() ?: d.optDouble("x1", 0.0).toFloat()
                        val y1 = bboxArr?.optDouble(1, 0.0)?.toFloat() ?: d.optDouble("y1", 0.0).toFloat()
                        val x2 = bboxArr?.optDouble(2, bitmap.width.toDouble())?.toFloat() ?: d.optDouble("x2", bitmap.width.toDouble()).toFloat()
                        val y2 = bboxArr?.optDouble(3, bitmap.height.toDouble())?.toFloat() ?: d.optDouble("y2", bitmap.height.toDouble()).toFloat()

                        val finalCls = remapToProduceName(rawCls, bitmap, x1, y1, x2, y2)
                        val item = DetectedItem(finalCls, conf, x1, y1, x2, y2)
                        items.add(item)

                        // Parse ConvNeXt freshness returned by server
                        val stage = d.optString("freshness_stage", "Fresh")
                        val stageProb = d.optDouble("freshness_prob", d.optDouble("freshness_probability", 0.90)).toFloat()
                        val ripeness = d.optString("ripeness_stage", "Ripe")
                        val ripenessProb = d.optDouble("ripeness_prob", d.optDouble("ripeness_probability", 0.85)).toFloat()
                        val shelfLife = d.optString("shelf_life_estimate", "4–7 days")
                        val badgeColor = d.optString("freshness_badge_color", "#22C55E")
                        val freshEmoji = d.optString("freshness_emoji", "✅")
                        val ripenessEmoji = d.optString("ripeness_emoji", "🟡")
                        val modelMode = d.optString("freshness_model_mode", "convnext")
                        val freshnessScore = d.optInt("freshness_score", d.optInt("multimodal_freshness_score", 80))

                        // Stage 3: Defect Detection & Segmentation
                        val defectCount = d.optInt("defect_count", 0)
                        val defectAreaPct = d.optDouble("total_defect_area_pct", 0.0).toFloat()

                        // Stage 5: Structured Predictions (XGBoost & LightGBM)
                        val structObj = d.optJSONObject("structured_predictions")
                        val qualityScoreStr = structObj?.optString("quality_score_str", "") ?: ""
                        val physAgeRange = structObj?.optString("physiological_age", "") ?: ""
                        val postHarvestAge = structObj?.optString("post_harvest_age", "") ?: ""
                        val shelfLifeRange = structObj?.optString("remaining_shelf_life", "") ?: ""
                        val spoilageRisk = structObj?.optString("spoilage_risk", "") ?: ""
                        val spoilageRiskColor = structObj?.optString("spoilage_risk_color", "#16A34A") ?: "#16A34A"
                        val timeToHarvest = structObj?.optString("time_to_harvest", "") ?: ""

                        // Stage 6: LLM + RAG Recommendations
                        val ragObj = d.optJSONObject("rag_recommendations")
                        val storageObj = ragObj?.optJSONObject("storage_strategy")
                        val storageGuideline = storageObj?.optString("refrigeration_guideline", "") ?: ""
                        val ethyleneWarning = storageObj?.optString("ethylene_co_location_warning", "") ?: ""
                        val recipeObj = ragObj?.optJSONObject("chef_recipe")
                        val chefTitle = recipeObj?.optString("title", "") ?: ""
                        val chefPrep = recipeObj?.optString("prep_time", "") ?: ""
                        val chefInstructions = recipeObj?.optString("instructions", "") ?: ""
                        val aiSummary = ragObj?.optString("ai_advisor_summary", "") ?: ""

                        estimates.add(
                            FreshnessEstimate.fromApiFields(
                                stage = stage,
                                stageProb = stageProb,
                                ripeness = ripeness,
                                ripenessProb = ripenessProb,
                                shelfLife = shelfLife,
                                badgeColor = badgeColor,
                                freshEmoji = freshEmoji,
                                ripenessEmoji = ripenessEmoji,
                                modelMode = modelMode,
                                freshnessScore = freshnessScore,
                                className = finalCls,
                                defectCount = defectCount,
                                defectAreaPct = defectAreaPct,
                                qualityScoreStr = qualityScoreStr,
                                physiologicalAgeRange = physAgeRange,
                                postHarvestAgeRange = postHarvestAge,
                                remainingShelfLifeRange = shelfLifeRange,
                                spoilageRisk = spoilageRisk,
                                spoilageRiskColorHex = spoilageRiskColor,
                                timeToHarvestRange = timeToHarvest,
                                storageGuideline = storageGuideline,
                                ethyleneWarning = ethyleneWarning,
                                chefRecipeTitle = chefTitle,
                                chefRecipePrepTime = chefPrep,
                                chefRecipeInstructions = chefInstructions,
                                aiAdvisorSummary = aiSummary,
                            )
                        )
                    }
                }
            }
        } catch (e: Exception) {
            items.clear()
            estimates.clear()
        } finally {
            connection?.disconnect()
        }

        return Pair(items, estimates)
    }

    /**
     * Refines and enlarges the bounding box so produce items are fully and generously enclosed.
     * Prevents cases where YOLO cuts off the top stem, leaves, or outer curvature of a fruit,
     * ensuring the box frames the entire physical produce item accurately and cleanly.
     */
    private fun expandAndRefineBoundingBox(item: DetectedItem, bitmap: Bitmap): DetectedItem {
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        var x1 = item.x1
        var y1 = item.y1
        var x2 = item.x2
        var y2 = item.y2

        val boxW = x2 - x1
        val boxH = y2 - y1
        if (boxW <= 10f || boxH <= 10f) return item

        val classNameLower = item.className.lowercase()
        val isRoundProduce = classNameLower.contains("tomato") ||
                classNameLower.contains("apple") ||
                classNameLower.contains("orange") ||
                classNameLower.contains("onion")

        // 1. Edge & Color Boundary Expansion (strictly capped to max 15% of box dimension so it never covers background)
        val maxUpExpand = boxH * 0.15f
        val minYTarget = (y1 - maxUpExpand).coerceAtLeast(0f)
        val cx = ((x1 + x2) / 2f).toInt().coerceIn(0, bitmap.width - 1)
        val samplePointsX = intArrayOf(
            cx,
            (cx - (boxW * 0.20f)).toInt().coerceIn(0, bitmap.width - 1),
            (cx + (boxW * 0.20f)).toInt().coerceIn(0, bitmap.width - 1)
        )

        var expandedY1 = y1
        val step = 4
        var py = (y1 - step).toInt()
        while (py >= minYTarget.toInt()) {
            var isProducePixel = false
            for (sx in samplePointsX) {
                val pixel = bitmap.getPixel(sx, py)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                // Concrete / floor is neutral low-saturation (|R-G| < 12 and |G-B| < 12)
                val isNeutralBg = Math.abs(r - g) < 12 && Math.abs(g - b) < 12
                val isWhiteBg = r > 220 && g > 220 && b > 220
                val isDarkBg = r < 30 && g < 30 && b < 30
                val hasProduceColor = (r > 115 && r > g + 18) || // Red/pink produce
                        (g > 75 && g > b && g > r - 30) || // Green stem / leaf
                        (r > 130 && g > 90 && b < 80) // Orange / yellow
                if (!isNeutralBg && !isWhiteBg && !isDarkBg && hasProduceColor) {
                    isProducePixel = true
                    break
                }
            }
            if (isProducePixel) {
                expandedY1 = py.toFloat()
                py -= step
            } else {
                break
            }
        }
        y1 = expandedY1

        // 2. Aspect Ratio Balance for Round Produce (capped to max 12% so box never grows huge)
        val currentW = x2 - x1
        val currentH = y2 - y1
        if (isRoundProduce && currentW > currentH * 1.25f) {
            val deficit = minOf(currentW * 0.90f - currentH, boxH * 0.12f)
            if (deficit > 0f) {
                y1 = (y1 - deficit * 0.60f).coerceAtLeast(0f)
                y2 = (y2 + deficit * 0.40f).coerceAtMost(h)
            }
        }

        // 3. Subtle Comfort Margin (adds ~2% padding so box doesn't cut edge pixels)
        val padX = (x2 - x1) * 0.02f
        val padY = (y2 - y1) * 0.02f
        x1 = (x1 - padX).coerceAtLeast(0f)
        y1 = (y1 - padY).coerceAtLeast(0f)
        x2 = (x2 + padX).coerceAtMost(w)
        y2 = (y2 + padY).coerceAtMost(h)

        return DetectedItem(
            className = item.className,
            confidence = item.confidence,
            x1 = x1,
            y1 = y1,
            x2 = x2,
            y2 = y2
        )
    }

    /**
     * Suppresses duplicate and nested bounding boxes for the exact same produce item.
     * Prevents cases where YOLO fires on both the whole fruit and a sub-region (e.g. skin defect/patch).
     */
    private fun suppressDuplicateAndNestedItems(
        items: List<DetectedItem>,
        estimates: List<FreshnessEstimate>
    ): Pair<List<DetectedItem>, List<FreshnessEstimate>> {
        if (items.size <= 1) return Pair(items, estimates)

        val pairs = items.indices.map { i ->
            Pair(items[i], estimates.getOrNull(i))
        }.sortedByDescending { it.first.confidence }

        val keptPairs = mutableListOf<Pair<DetectedItem, FreshnessEstimate?>>()

        for (curr in pairs) {
            val c = curr.first
            // Filter out low-confidence ghost boxes (< 12%)
            if (c.confidence < 0.12f) continue

            val cArea = maxOf(1f, (c.x2 - c.x1) * (c.y2 - c.y1))
            var isDuplicate = false

            for (i in keptPairs.indices) {
                val kept = keptPairs[i]
                val k = kept.first
                val kArea = maxOf(1f, (k.x2 - k.x1) * (k.y2 - k.y1))

                val ix1 = maxOf(c.x1, k.x1)
                val iy1 = maxOf(c.y1, k.y1)
                val ix2 = minOf(c.x2, k.x2)
                val iy2 = minOf(c.y2, k.y2)

                val interW = maxOf(0f, ix2 - ix1)
                val interH = maxOf(0f, iy2 - iy1)
                val interArea = interW * interH
                val iou = interArea / maxOf(1f, (cArea + kArea - interArea))
                val containment = interArea / maxOf(1f, minOf(cArea, kArea))

                val sameClass = c.className.equals(k.className, ignoreCase = true)
                val xOverlap = maxOf(0f, minOf(c.x2, k.x2) - maxOf(c.x1, k.x1))
                val touchesOrAdjacent = (xOverlap > 0.3f * minOf(c.x2 - c.x1, k.x2 - k.x1) &&
                        (Math.abs(c.y1 - k.y2) < 0.25f * (k.y2 - k.y1) || Math.abs(c.y2 - k.y1) < 0.25f * (k.y2 - k.y1)))

                // If IoU > 0.30 OR nested box (> 50%) OR adjacent same-class fragment (e.g. stem on top of fruit)
                if (iou > 0.30f || containment > 0.50f || (sameClass && (touchesOrAdjacent || containment > 0.25f))) {
                    isDuplicate = true
                    // Expand the kept box so that the entire fruit (stem + body) is cleanly enclosed
                    val expandedK = DetectedItem(
                        className = k.className,
                        confidence = maxOf(k.confidence, c.confidence),
                        x1 = minOf(k.x1, c.x1),
                        y1 = minOf(k.y1, c.y1),
                        x2 = maxOf(k.x2, c.x2),
                        y2 = maxOf(k.y2, c.y2)
                    )
                    keptPairs[i] = Pair(expandedK, kept.second)
                    break
                }
            }

            if (!isDuplicate) {
                keptPairs.add(curr)
            }
        }

        // If all were filtered out due to confidence, keep the highest confidence one
        if (keptPairs.isEmpty() && pairs.isNotEmpty()) {
            keptPairs.add(pairs[0])
        }

        val keptItems = keptPairs.map { it.first }
        val keptEstimates = keptPairs.mapNotNull { it.second }
        return Pair(keptItems, keptEstimates)
    }

    /**
     * Determines the exact produce type from image crop pixels using HSV & morphology analysis.
     * Accurately distinguishes Garlic (Lehsun), Ginger (Adrak), Potato (Aloo), Onion (Pyaz),
     * Carrot (Gajar), Cucumber (Kheera), Bell Pepper, Tomato, Eggplant, Lemon, Banana, Apple, etc.
     */
    private fun classifyProduceFromCrop(
        bitmap: Bitmap,
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        candidateName: String? = null
    ): String {
        val bx1 = x1.toInt().coerceIn(0, bitmap.width - 1)
        val by1 = y1.toInt().coerceIn(0, bitmap.height - 1)
        val bx2 = x2.toInt().coerceIn(bx1 + 1, bitmap.width)
        val by2 = y2.toInt().coerceIn(by1 + 1, bitmap.height)

        val boxW = (bx2 - bx1).toFloat().coerceAtLeast(1f)
        val boxH = (by2 - by1).toFloat().coerceAtLeast(1f)
        val aspectRatio = maxOf(boxW / boxH, boxH / boxW)
        val isElongated = aspectRatio > 1.75f
        val cand = candidateName?.lowercase()?.trim() ?: ""

        var count = 0
        var tomatoRedCount = 0        // Crimson / Vine-Ripe Red / Breaker Red-Orange (Tomato)
        var onionMagentaCount = 0     // Anthocyanin Violet / Magenta (Red Onion)
        var onionPaperyGoldCount = 0  // Dry papery golden-tan tunic (Yellow Onion)
        var garlicIvoryCount = 0      // Pearly ivory/white cloves (Garlic)
        var potatoEarthyTanCount = 0  // Low-saturation earthy muted brown/tan (Potato)
        var gingerBuffCount = 0       // Warm buff/tan fibrous rhizome (Ginger)
        var carrotOrangeCount = 0     // Deep carrot orange taproot (Carrot)
        var citrusOrangeCount = 0     // Citrus orange fruit
        var lemonYellowCount = 0      // Bright yellow (Lemon / Banana)
        var greenCount = 0            // Green (Cucumber / Capsicum / Raw produce)
        var eggplantPurpleCount = 0   // Deep violet / dark eggplant

        val hsvTemp = FloatArray(3)
        // Sample a stable grid density. Dividing the area directly by the
        // target count produces only a handful of pixels for large boxes.
        val sampleArea = (bx2 - bx1).toDouble() * (by2 - by1).toDouble()
        val step = maxOf(1, kotlin.math.sqrt(sampleArea / 2500.0).toInt())

        // ─── TOP-ZONE NECK SCAN (same as YoloOnnxDetector) ──────────────────────────
        val bboxHt = by2 - by1
        val topEndM = (by1 + bboxHt * 0.22f).toInt().coerceIn(by1 + 1, by2)
        var neckOchreM = 0; var neckTotalM = 0
        val topStepM = maxOf(1, (bx2 - bx1) / 14)
        for (nx in bx1 until bx2 step topStepM) {
            for (ny in by1 until topEndM step topStepM) {
                val p = bitmap.getPixel(nx, ny)
                val nr = Color.red(p); val ng = Color.green(p); val nb = Color.blue(p)
                if ((nr > 248 && ng > 248 && nb > 248) || (nr < 10 && ng < 10 && nb < 10)) { neckTotalM++; continue }
                Color.RGBToHSV(nr, ng, nb, hsvTemp)
                val nh = hsvTemp[0]; val ns = hsvTemp[1]; val nv = hsvTemp[2]
                if (nh in 20f..68f && ns in 0.10f..0.82f && nv in 0.14f..0.82f && nr > ng && ng > nb && (nr - ng) > 7 && (ng - nb) > 5) {
                    neckOchreM++
                }
                neckTotalM++
            }
        }
        val neckOchrePctM = if (neckTotalM > 2) neckOchreM.toFloat() / neckTotalM else 0f
        // ─────────────────────────────────────────────────────────────────────────────

        for (px in bx1 until bx2 step step) {
            for (py in by1 until by2 step step) {
                val pixel = bitmap.getPixel(px, py)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                // Skip glare highlights or deep black shadows
                if ((r > 248 && g > 248 && b > 248) || (r < 12 && g < 12 && b < 12)) continue

                Color.RGBToHSV(r, g, b, hsvTemp)
                val h = hsvTemp[0] // 0..360
                val s = hsvTemp[1] // 0..1
                val v = hsvTemp[2] // 0..1

                // Skip floor/table background (tiles, wood, marble, cement, peach/tan surfaces)
                val isFloor = (s < 0.24f && v > 0.45f) ||
                        (Math.abs(r - g) < 22 && Math.abs(g - b) < 22 && s < 0.25f) ||
                        v < 0.08f || v > 0.96f
                if (isFloor) continue

                // 1. Red / Maroon Onion (Pyaz): anthocyanin violet/purple OR dark maroon red onion skin
                // KEY: Checked FIRST so dark reddish onion skin (v <= 0.62, r < 195) is captured as onion,
                // and does NOT get stolen by isTomato.
                if (
                    (h in 260f..350f && s >= 0.16f && v >= 0.14f) ||
                    (h in 345f..360f && s in 0.16f..0.60f && v in 0.20f..0.80f && b > 45 && r < 140) ||
                    ((h <= 22f || h >= 340f) && s in 0.25f..0.85f && v in 0.12f..0.62f && r < 195 && r > g + 8)
                ) {
                    onionMagentaCount++
                }
                // 2. Yellow/Brown Onion: Dry papery golden/copper husk
                else if (h in 18f..50f && s in 0.18f..0.78f && v in 0.20f..0.82f &&
                         r in 100..220 && g in 68..172 && b in 14..115 &&
                         (r - g) in 14..65 && (g - b) in 25..82) {
                    onionPaperyGoldCount++
                }
                // 3. Tomato (Tamatar): Deep vivid crimson/red or genuine breaker orange
                // Only bright, vivid reds reach here (v > 0.55, r > 155, high saturation)
                else if (
                    ((h <= 18f || h >= 345f) && s >= 0.45f && v >= 0.55f &&
                     r > 155 && r > g + 40 && r > b + 40 && b < 85) ||
                    (h in 5f..28f && s >= 0.48f && v >= 0.35f &&
                     r > 155 && r > g + 30 && b < 75 &&
                     (r.toFloat() / maxOf(1, b).toFloat()) > 3.0f)
                ) {
                    tomatoRedCount++
                }
                // 4. Garlic (Lehsun): Papery ivory / pearl-white / light cream (high value, low saturation, warm tint)
                else if (s in 0.04f..0.18f && v in 0.65f..0.98f && r in 170..245 && g in 160..242 && b in 140..230 && r >= g && g >= b && (r - b in 12..35)) {
                    garlicIvoryCount++
                }
                // 5. Potato (Aloo): Earthy neutral tan/gray-brown (muted saturation, balanced)
                else if (s in 0.10f..0.28f && v in 0.28f..0.72f && r in 80..185 && g in 65..165 && b in 45..135 && Math.abs(r - g) < 26 && Math.abs(g - b) < 32) {
                    potatoEarthyTanCount++
                }
                // 6. Ginger (Adrak): Warm buff-tan / light golden fibrous rhizome
                else if (h in 24f..46f && s in 0.22f..0.54f && v in 0.42f..0.82f && r in 125..215 && g in 90..175 && b in 30..115 && (r - g in 18..52) && (g - b in 30..80)) {
                    gingerBuffCount++
                }
                // 7. Carrot (Gajar): Vivid deep orange
                else if (h in 14f..34f && s > 0.60f && r > 175 && g in 75..145 && b < 70) {
                    carrotOrangeCount++
                }
                // 8. Citrus Orange fruit
                else if (h in 16f..38f && s > 0.55f && r > 180 && g in 95..160 && b < 80) {
                    citrusOrangeCount++
                }
                // 9. Eggplant / Brinjal: Dark violet / deep purple
                else if ((h in 250f..320f && v in 0.10f..0.45f) || (r in 35..110 && b in 40..120 && g < 50)) {
                    eggplantPurpleCount++
                }
                // 10. Bright Yellow (Lemon / Banana)
                else if (h in 45f..68f && s > 0.32f && r > 160 && g > 135 && b < 130) {
                    lemonYellowCount++
                }
                // 11. Green Produce (Cucumber / Capsicum / Raw Produce / Vine)
                else if ((g > r && g > b) || (h in 65f..160f && s > 0.18f)) {
                    greenCount++
                }

                count++
            }
        }

        if (count < 8) {
            return if (candidateName != null && candidateName.isNotEmpty() && !candidateName.equals("produce", ignoreCase = true)) {
                candidateName
            } else "Tomato"
        }

        val tomatoPct = tomatoRedCount.toFloat() / count
        val mOnionPct = onionMagentaCount.toFloat() / count
        val yOnionPct = onionPaperyGoldCount.toFloat() / count
        val garlicPct = garlicIvoryCount.toFloat() / count
        val potatoPct = potatoEarthyTanCount.toFloat() / count
        val gingerPct = gingerBuffCount.toFloat() / count
        val carrotPct = carrotOrangeCount.toFloat() / count
        val orangePct = citrusOrangeCount.toFloat() / count
        val eggplantPct = eggplantPurpleCount.toFloat() / count
        val yellowPct = lemonYellowCount.toFloat() / count
        val greenPct = greenCount.toFloat() / count

        val onionTotalPct = mOnionPct + yOnionPct

        // ── Scientific Relative Decision Engine ──

        // PRIORITY 0: Neck/stalk ochre check — most reliable onion indicator
        if (neckOchrePctM > 0.25f) {
            return "Onion"
        }

        // PRIORITY 1: Onion traits clearly dominate over tomato
        if (onionTotalPct > 0.08f && onionTotalPct >= tomatoPct * 0.70f) {
            return "Onion"
        }
        if (mOnionPct > 0.12f) {
            return "Onion"
        }

        // PRIORITY 2: If candidate was labeled tomato, but pixel analysis shows onion
        if ((cand == "tomato" || cand == "tamatar") && (onionTotalPct > 0.05f && onionTotalPct >= tomatoPct * 0.50f)) {
            return "Onion"
        }
        if ((cand == "tomato" || cand == "tamatar") && neckOchrePctM > 0.15f) {
            return "Onion"
        }

        // 1. Tomato: Only when genuine strong tomato pixels clearly dominate
        if (tomatoPct > 0.08f && tomatoPct > onionTotalPct * 1.5f && neckOchrePctM < 0.18f &&
            (cand != "onion" && cand != "pyaz")) {
            return if (cand == "apple") "Apple" else "Tomato"
        }
        // Confirm Tomato if model said tomato, pixel confirms it, neck is clean
        if ((cand == "tomato" || cand == "tamatar") && tomatoPct > 0.08f &&
            tomatoPct > onionTotalPct * 1.5f && neckOchrePctM < 0.15f) {
            return "Tomato"
        }

        // 2. Onion: True anthocyanin violet, dark maroon, or dry papery copper tunic
        val hasOnionTraits = (mOnionPct > 0.05f || onionMagentaCount >= 4) ||
                             (yOnionPct > 0.12f && onionPaperyGoldCount >= 6 && tomatoPct < 0.04f)
        if ((cand == "onion" || cand == "pyaz") && tomatoPct < 0.15f) {
            return "Onion"
        }
        if (hasOnionTraits && tomatoPct < 0.15f) {
            return "Onion"
        }

        // 3. Garlic: Only when ivory/white clove scales strongly dominate and there are NO onion tunic colors
        if (garlicPct > 0.35f && garlicPct > potatoPct && !hasOnionTraits && cand != "onion") {
            return "Garlic"
        }

        // 4. Specialized Produce Shapes & Colors
        if (eggplantPct > 0.15f) return "Eggplant"
        if (isElongated && carrotPct > 0.15f) return "Carrot"
        if (isElongated && greenPct > 0.20f) return "Cucumber"
        if (isElongated && yellowPct > 0.18f) return "Banana"

        // 5. Roots & Tubers
        if (gingerPct > 0.18f && potatoPct < 0.15f) return "Ginger"
        if (potatoPct > 0.16f) return "Potato"
        if (orangePct > 0.18f) return if (cand == "carrot") "Carrot" else "Orange"
        if (yellowPct > 0.18f) return "Lemon"
        if (greenPct > 0.22f) return if (cand == "watermelon") "Watermelon" else if (cand == "bell pepper") "Bell Pepper" else "Tomato"

        // Candidate fallback
        if (candidateName != null && candidateName.isNotEmpty() && !candidateName.equals("produce", ignoreCase = true)) {
            return candidateName
        }
        if (onionTotalPct > 0.04f || mOnionPct > 0.04f) {
            return "Onion"
        }
        return "Tomato"
    }

    /**
     * Client-side safety net: if YOLO/server returns a non-produce label (Vase, Bowl, etc.),
     * use pixel color analysis of the bounding box crop to determine the correct produce name.
     */
    private fun remapToProduceName(rawClass: String, bitmap: Bitmap, x1: Float, y1: Float, x2: Float, y2: Float): String {
        val nonProduceLabels = setOf(
            "vase", "bowl", "cup", "bottle", "wine glass", "sports ball", "clock",
            "chair", "couch", "dining table", "toilet", "tv", "laptop", "mouse",
            "keyboard", "cell phone", "book", "scissors", "teddy bear", "umbrella",
            "handbag", "suitcase", "frisbee", "kite", "traffic light", "fire hydrant",
            "stop sign", "person", "car", "truck", "bicycle", "dog", "cat", "bird"
        )

        val lowerClass = rawClass.lowercase().trim()
        val validProduce = setOf(
            "tomato", "potato", "garlic", "ginger", "apple", "banana",
            "orange", "lemon", "carrot", "cucumber", "eggplant", "watermelon",
            "bell pepper", "pepper", "strawberry", "mango"
        )

        // For Onion → always verify with pixel color (might actually be a tomato)
        if (lowerClass == "onion" || lowerClass == "pyaz") {
            return classifyProduceFromCrop(bitmap, x1, y1, x2, y2, rawClass)
        }

        // For Tomato → also verify with pixel color (might actually be an onion)
        // This catches the case where the YOLO model outputs "Tomato" for a golden-brown onion
        if (lowerClass == "tomato" || lowerClass == "tamatar") {
            return classifyProduceFromCrop(bitmap, x1, y1, x2, y2, rawClass)
        }

        // If the detector already identified another valid produce type, preserve it!
        if (validProduce.contains(lowerClass)) {
            return rawClass
        }

        if (!nonProduceLabels.contains(lowerClass) && lowerClass.isNotEmpty() && lowerClass != "produce") {
            return classifyProduceFromCrop(bitmap, x1, y1, x2, y2, rawClass)
        }

        return classifyProduceFromCrop(bitmap, x1, y1, x2, y2, null)
    }

    private fun runOnDeviceProduceDetection(bitmap: Bitmap): List<DetectedItem> {
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        val items = mutableListOf<DetectedItem>()

        var minX = w
        var minY = h
        var maxX = 0f
        var maxY = 0f
        var foundCount = 0

        val step = 8
        val hsvTemp = FloatArray(3)

        for (x in (w * 0.05f).toInt() until (w * 0.95f).toInt() step step) {
            for (y in (h * 0.05f).toInt() until (h * 0.95f).toInt() step step) {
                val p = bitmap.getPixel(x, y)
                val r = Color.red(p)
                val g = Color.green(p)
                val b = Color.blue(p)

                Color.RGBToHSV(r, g, b, hsvTemp)
                val hue = hsvTemp[0]
                val sat = hsvTemp[1]
                val value = hsvTemp[2]

                // Ignore background (concrete floor, peach/tan tiles, table, shadows: low saturation or neutral gray/tan)
                val isFloor = (sat < 0.24f && value > 0.45f) ||
                        (Math.abs(r - g) < 22 && Math.abs(g - b) < 22 && sat < 0.25f) ||
                        value < 0.08f || value > 0.96f
                if (isFloor) continue

                val isProduceColor = (hue in 260f..355f && sat > 0.10f) ||   // Onion violet/magenta
                        (hue in 14f..48f && sat in 0.16f..0.85f) ||           // Onion ochre/tan or Orange
                        (hue in 45f..68f && sat > 0.25f) ||                   // Banana/Lemon yellow
                        (hue in 65f..160f && sat > 0.16f) ||                  // Green produce
                        ((hue <= 14f || hue >= 348f) && sat > 0.28f) ||      // Red produce
                        (value in 0.08f..0.42f && sat < 0.45f && (r < 115 || g < 110 || b < 110)) // Produce dark rot/mold/decay blemishes

                if (isProduceColor) {
                    foundCount++
                    if (x < minX) minX = x.toFloat()
                    if (x > maxX) maxX = x.toFloat()
                    if (y < minY) minY = y.toFloat()
                    if (y > maxY) maxY = y.toFloat()
                }
            }
        }

        val boxX1: Float
        val boxY1: Float
        val boxX2: Float
        val boxY2: Float

        if (foundCount > 15 && maxX > minX && maxY > minY) {
            val padX = ((maxX - minX) * 0.08f).coerceIn(6f, 25f)
            val padY = ((maxY - minY) * 0.08f).coerceIn(6f, 25f)
            boxX1 = (minX - padX).coerceAtLeast(0f)
            boxY1 = (minY - padY).coerceAtLeast(0f)
            boxX2 = (maxX + padX).coerceAtMost(w)
            boxY2 = (maxY + padY).coerceAtMost(h)
        } else {
            // Default center framed crop
            boxX1 = w * 0.20f
            boxY1 = h * 0.20f
            boxX2 = w * 0.80f
            boxY2 = h * 0.80f
        }

        val detectedClass = classifyProduceFromCrop(bitmap, boxX1, boxY1, boxX2, boxY2, null)
        val confidence = if (foundCount > 15) 0.94f else 0.88f
        items.add(DetectedItem(detectedClass, confidence, boxX1, boxY1, boxX2, boxY2))

        return items
    }

    private fun updateAnalyticsUI(items: List<DetectedItem>, imgW: Int, imgH: Int) {
        kpiRow.visibility = View.VISIBLE

        val total = items.size
        val uniqueClasses = items.map { it.className }.distinct().size
        val avgConf = if (items.isNotEmpty()) (items.map { it.confidence }.average() * 100).toInt() else 0

        tvKpiTotal.text = total.toString()
        tvKpiClasses.text = uniqueClasses.toString()
        tvKpiConf.text = "$avgConf%"
    }

    // ==========================================
    // Step 2: Freshness Analysis UI
    // ==========================================

    private fun updateFreshnessUI(
        items: List<DetectedItem>,
        estimates: List<FreshnessEstimate>,
    ) {
        val bitmap = lastAnalyzedBitmap
        if (!isCropGrowthMode && items.isEmpty() && bitmap == null) {
            cardFreshness.visibility = View.GONE
            return
        }

        cardFreshness.visibility = View.VISIBLE
        freshnessItemsContainer.removeAllViews()

        if (isCropGrowthMode) {
            // ==========================================
            // 🌱 CROP & PLANT GROWTH SCANNER MODE
            // ==========================================
            tvFreshnessModelMode.text = "🌱 Agronomy AI Engine"
            layoutOverallFreshnessSummary.visibility = View.GONE

            val cropToAnalyze = selectedCropName ?: items.firstOrNull()?.className ?: "Tomato"
            val p = CropGrowthRepository.getProfile(cropToAnalyze) ?: CropGrowthRepository.CROPS.first()

            // Synchronize chip selection with detected or selected crop
            isSyncingCropChips = true
            try {
                for (i in 0 until chipGroupCrops.childCount) {
                    val chip = chipGroupCrops.getChildAt(i) as? Chip ?: continue
                    val match = chip.text.contains(p.name, ignoreCase = true)
                    if (chip.isChecked != match) {
                        chip.isChecked = match
                    }
                    if (match) {
                        chip.chipStrokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#15803D"))
                        chip.setTextColor(Color.parseColor("#15803D"))
                    } else {
                        chip.chipStrokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor("#CBD5E1"))
                        chip.setTextColor(Color.parseColor("#334155"))
                    }
                }
            } finally {
                isSyncingCropChips = false
            }

            val plantEst = if (bitmap != null) {
                val item = items.firstOrNull()
                PlantGrowthAnalyzer.analyze(
                    bitmap,
                    cropToAnalyze,
                    item?.x1 ?: 0f, item?.y1 ?: 0f,
                    item?.x2 ?: bitmap.width.toFloat(), item?.y2 ?: bitmap.height.toFloat()
                )
            } else {
                PlantGrowthEstimate(
                    profile = p,
                    currentStageIndex = 3,
                    currentStage = p.stages.getOrNull(3) ?: p.stages.last(),
                    estimatedDaysElapsed = Math.round(p.averageDays * 0.80f),
                    remainingDaysToHarvest = maxOf(7, Math.round(p.averageDays * 0.20f)),
                    progressPercent = 80,
                    confidence = 1.0f,
                    detectionSummary = "Overview for ${p.name} (${p.category}): Takes ${p.totalDurationRange} (~${p.averageDays} days) from planting to harvest."
                )
            }

            val dummyItem = items.firstOrNull() ?: DetectedItem(p.name, 1.0f, 0f, 0f, 100f, 100f)
            val card = buildPlantGrowthCard(dummyItem, plantEst)
            freshnessItemsContainer.addView(card)
        } else {
            // ==========================================
            // 🍎 HARVESTED PRODUCE FRESHNESS MODE
            // ==========================================
            layoutOverallFreshnessSummary.visibility = View.VISIBLE
            val mode = estimates.firstOrNull()?.modelMode ?: "heuristic"
            tvFreshnessModelMode.text = if (mode == "convnext") "🧠 ConvNeXt-Tiny" else "🎨 Heuristic"

            val avgScore = if (estimates.isNotEmpty()) estimates.map { it.freshnessScore }.average().toInt() else 0
            val dominantStageIdx = if (estimates.isNotEmpty())
                estimates.map { it.stageIndex }.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key ?: 0
            else 0
            val overallColor = FreshnessEstimate.FRESHNESS_COLORS.getOrElse(dominantStageIdx) { "#22C55E" }
            tvOverallFreshness.text = "${FreshnessEstimate.FRESHNESS_EMOJI.getOrElse(dominantStageIdx) { "✅" }} " +
                    "${FreshnessEstimate.FRESHNESS_STAGES.getOrElse(dominantStageIdx) { "Fresh" }} • $avgScore/100"
            tvOverallFreshness.setTextColor(Color.parseColor(overallColor))
            progressFreshnessOverall.progress = avgScore
            android.graphics.PorterDuffColorFilter(
                Color.parseColor(overallColor), android.graphics.PorterDuff.Mode.SRC_IN
            ).also { filter ->
                progressFreshnessOverall.progressDrawable.colorFilter = filter
            }

            // Build per-item rows
            for ((idx, item) in items.withIndex()) {
                val fr = estimates.getOrNull(idx) ?: continue
                val itemView = buildFreshnessItemRow(item, fr)
                freshnessItemsContainer.addView(itemView)
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun buildFreshnessItemRow(item: DetectedItem, fr: FreshnessEstimate): android.view.View {
        val ctx = this
        val container = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, dpToPx(4))
        }

        // Row 1: Emoji + Name + Freshness Badge
        val row1 = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val tvEmoji = android.widget.TextView(ctx).apply {
            text = fr.freshnesEmoji
            textSize = 20f
            setPadding(0, 0, dpToPx(6), 0)
        }

        val tvName = android.widget.TextView(ctx).apply {
            text = item.className
            textSize = 12.5f
            setTextColor(Color.parseColor("#0F172A"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val tvBadge = android.widget.TextView(ctx).apply {
            text = fr.stage
            textSize = 10.5f
            setTextColor(Color.WHITE)
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(8), dpToPx(3), dpToPx(8), dpToPx(3))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(fr.badgeColor())
            }
        }

        row1.addView(tvEmoji)
        row1.addView(tvName)
        row1.addView(tvBadge)

        // Stage 3: Defect Area Badge
        if (fr.defectAreaPct > 0f) {
            val tvDefect = android.widget.TextView(ctx).apply {
                text = "Defect: ${String.format("%.1f", fr.defectAreaPct)}%"
                textSize = 9.5f
                setTextColor(Color.parseColor(if (fr.defectAreaPct > 10f) "#DC2626" else "#D97706"))
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(dpToPx(5), dpToPx(2), dpToPx(5), dpToPx(2))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { marginStart = dpToPx(4) }
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(100).toFloat()
                    setColor(Color.parseColor(if (fr.defectAreaPct > 10f) "#FEE2E2" else "#FEF3C7"))
                }
            }
            row1.addView(tvDefect)
        }
        container.addView(row1)

        // Row 2: Freshness progress bar + score
        val tvScore = android.widget.TextView(ctx).apply {
            text = "Freshness: ${fr.freshnessScore}/100  •  ${(fr.probability * 100).toInt()}% confidence"
            textSize = 10.5f
            setTextColor(Color.parseColor("#475569"))
            setPadding(0, dpToPx(3), 0, dpToPx(2))
        }
        container.addView(tvScore)

        val progressBar = android.widget.ProgressBar(ctx, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = fr.freshnessScore
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(6)
            ).apply { bottomMargin = dpToPx(4) }
            android.graphics.PorterDuffColorFilter(
                fr.badgeColor(), android.graphics.PorterDuff.Mode.SRC_IN
            ).also { filter -> progressDrawable.colorFilter = filter }
        }
        container.addView(progressBar)

        // Row 3: Ripeness + Shelf Life
        val row3 = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val tvRipeness = android.widget.TextView(ctx).apply {
            text = "${fr.ripenessEmoji} ${fr.ripenessStage}"
            textSize = 10.5f
            setTextColor(Color.parseColor("#334155"))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val arrhenius = fr.calculateArrheniusShelfLife(tempC = 22f)
        val tvShelf = android.widget.TextView(ctx).apply {
            text = "🌡️ Arrhenius: ${arrhenius.summary}"
            textSize = 10.5f
            setTextColor(Color.parseColor("#166534"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(8), dpToPx(2), dpToPx(8), dpToPx(2))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(6).toFloat()
                setColor(Color.parseColor("#F0FDF4"))
                setStroke(dpToPx(1), Color.parseColor("#86EFAC"))
            }
        }

        row3.addView(tvRipeness)
        row3.addView(tvShelf)
        container.addView(row3)

        // ── Produce Quality & Storage Intelligence Card ─────────────────────
        if (fr.qualityScoreStr.isNotEmpty() || fr.shelfLifeEstimate.isNotEmpty()) {
            val structCard = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(2) }
                setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(8).toFloat()
                    setColor(Color.parseColor("#0F172A"))
                }
            }

            val tvStructHeader = android.widget.TextView(ctx).apply {
                text = "📊 Quality & Storage Intelligence"
                textSize = 10f
                setTextColor(Color.parseColor("#38BDF8"))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            structCard.addView(tvStructHeader)

            val gridRow = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { topMargin = dpToPx(3) }
            }

            fun addMetricCol(label: String, value: String, colorHex: String, sub: String) {
                val col = LinearLayout(ctx).apply {
                    orientation = LinearLayout.VERTICAL
                    gravity = android.view.Gravity.CENTER
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    setPadding(dpToPx(1), dpToPx(1), dpToPx(1), dpToPx(1))
                }
                val tvL = android.widget.TextView(ctx).apply {
                    text = label
                    textSize = 7.5f
                    setTextColor(Color.parseColor("#94A3B8"))
                }
                val tvV = android.widget.TextView(ctx).apply {
                    text = value
                    textSize = 11f
                    setTextColor(Color.parseColor(colorHex))
                    setTypeface(null, android.graphics.Typeface.BOLD)
                }
                val tvS = android.widget.TextView(ctx).apply {
                    text = sub
                    textSize = 6.5f
                    setTextColor(Color.parseColor("#64748B"))
                }
                col.addView(tvL)
                col.addView(tvV)
                col.addView(tvS)
                gridRow.addView(col)
            }

            val qVal = if (fr.qualityScoreStr.isNotEmpty()) fr.qualityScoreStr else "${fr.freshnessScore}/100"
            val bioAgeVal = if (fr.postHarvestAgeRange.isNotEmpty()) {
                fr.postHarvestAgeRange.replace(" post-harvest", "").replace("d", " days")
            } else if (fr.physiologicalAgeRange.isNotEmpty()) {
                fr.physiologicalAgeRange
            } else {
                val postH = Math.max(1, (100 - fr.freshnessScore) / 12 + 1)
                "${postH}–${postH + 2} days"
            }
            val shelfVal = if (fr.remainingShelfLifeRange.isNotEmpty()) fr.remainingShelfLifeRange else fr.shelfLifeEstimate
            val riskVal = if (fr.spoilageRisk.isNotEmpty()) fr.spoilageRisk else if (fr.freshnessScore < 35) "High" else if (fr.freshnessScore < 65) "Medium" else "Low"
            val riskColor = if (fr.spoilageRiskColorHex.isNotEmpty()) fr.spoilageRiskColorHex else if (fr.freshnessScore < 35) "#EF4444" else if (fr.freshnessScore < 65) "#FBBF24" else "#16A34A"

            addMetricCol("QUALITY", qVal, "#38BDF8", "Score")
            addMetricCol("BIO AGE", bioAgeVal, "#FBBF24", "Post-Harvest")
            addMetricCol("SHELF LIFE", shelfVal, "#34D399", "Storage")
            addMetricCol("RISK", riskVal, riskColor, "Hazard")

            structCard.addView(gridRow)

            val tvAgeDetail = android.widget.TextView(ctx).apply {
                text = "🧬 Bio Age: $bioAgeVal • State: ${fr.ripenessStage} (${fr.stage})"
                textSize = 9f
                setTextColor(Color.parseColor("#94A3B8"))
                setPadding(dpToPx(2), dpToPx(3), dpToPx(2), 0)
            }
            structCard.addView(tvAgeDetail)
            container.addView(structCard)
        }

        // ── AI Chef & Post-Harvest Advisor Card ──────────────────────────────
        if (fr.storageGuideline.isNotEmpty() || fr.chefRecipeTitle.isNotEmpty() || fr.aiAdvisorSummary.isNotEmpty()) {
            val ragCard = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(2) }
                setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(8).toFloat()
                    setColor(Color.parseColor("#F8FAFC"))
                    setStroke(dpToPx(1), Color.parseColor("#CBD5E1"))
                }
            }

            val tvRagHeader = android.widget.TextView(ctx).apply {
                text = "🤖 AI Chef & Post-Harvest Advisor"
                textSize = 10f
                setTextColor(Color.parseColor("#15803D"))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            ragCard.addView(tvRagHeader)

            if (fr.storageGuideline.isNotEmpty()) {
                val tvStorage = android.widget.TextView(ctx).apply {
                    text = "🧊 Storage: ${fr.storageGuideline}"
                    textSize = 9.5f
                    setTextColor(Color.parseColor("#334155"))
                    setPadding(0, dpToPx(2), 0, dpToPx(1))
                }
                ragCard.addView(tvStorage)
            }

            if (fr.chefRecipeTitle.isNotEmpty()) {
                val tvRecipe = android.widget.TextView(ctx).apply {
                    val prep = if (fr.chefRecipePrepTime.isNotEmpty()) " (${fr.chefRecipePrepTime})" else ""
                    text = "🥗 Chef Recipe: ${fr.chefRecipeTitle}$prep\n${fr.chefRecipeInstructions}"
                    textSize = 9.5f
                    setTextColor(Color.parseColor("#1E293B"))
                    setTypeface(null, android.graphics.Typeface.ITALIC)
                    setPadding(0, dpToPx(1), 0, dpToPx(1))
                }
                ragCard.addView(tvRecipe)
            }

            container.addView(ragCard)
        }

        // Subtle divider if not last
        val divider = android.view.View(ctx).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(1)
            ).apply { topMargin = dpToPx(6); bottomMargin = dpToPx(2) }
            setBackgroundColor(Color.parseColor("#F1F5F9"))
        }
        container.addView(divider)

        return container
    }

    // ==========================================
    // Crop Growth Analysis UI Card Builder
    // ==========================================

    private fun buildPlantGrowthCard(item: DetectedItem, est: PlantGrowthEstimate): View {
        val ctx = this
        val crop = est.profile
        val isFruit = crop.category.equals("Fruit", ignoreCase = true)

        val card = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(2); bottomMargin = dpToPx(4) }
            setPadding(dpToPx(10), dpToPx(8), dpToPx(10), dpToPx(10))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(12).toFloat()
                setColor(Color.parseColor("#F8FAFC"))
                setStroke(dpToPx(1), Color.parseColor("#E2E8F0"))
            }
        }

        // 1. Header Banner: Emoji + Name + Scientific + Category Badge + Difficulty Badge
        val headerRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val nameCol = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val tvTitle = TextView(ctx).apply {
            text = "${crop.emoji} ${crop.name} Plant"
            textSize = 14.5f
            setTextColor(Color.parseColor("#0F172A"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }

        val tvSci = TextView(ctx).apply {
            text = "${crop.scientificName} • ${crop.totalDurationRange}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#64748B"))
            setTypeface(null, android.graphics.Typeface.ITALIC)
        }

        nameCol.addView(tvTitle)
        nameCol.addView(tvSci)
        headerRow.addView(nameCol)

        val badgeCol = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
        }

        val tvCategoryBadge = TextView(ctx).apply {
            text = crop.category.uppercase()
            textSize = 8.5f
            setTextColor(Color.parseColor(if (isFruit) "#C2410C" else "#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(6), dpToPx(2), dpToPx(6), dpToPx(2))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(Color.parseColor(if (isFruit) "#FFEDD5" else "#DCFCE7"))
                setStroke(dpToPx(1), Color.parseColor(if (isFruit) "#FDBA74" else "#86EFAC"))
            }
        }

        val tvDiffBadge = TextView(ctx).apply {
            text = crop.difficulty
            textSize = 8.5f
            setTextColor(Color.parseColor("#475569"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(6), dpToPx(2), dpToPx(6), dpToPx(2))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { marginStart = dpToPx(4) }
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(Color.parseColor("#F1F5F9"))
                setStroke(dpToPx(1), Color.parseColor("#CBD5E1"))
            }
        }

        badgeCol.addView(tvCategoryBadge)
        badgeCol.addView(tvDiffBadge)
        headerRow.addView(badgeCol)
        card.addView(headerRow)

        // 2. Real-Time Vision Biometrics 3-Tile Row
        val bioRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(5); bottomMargin = dpToPx(2) }
        }

        fun addBioCard(container: LinearLayout, label: String, value: String, sub: String, colorHex: String) {
            val tile = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                gravity = android.view.Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    setMargins(dpToPx(1), 0, dpToPx(1), 0)
                }
                setPadding(dpToPx(4), dpToPx(4), dpToPx(4), dpToPx(4))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(6).toFloat()
                    setColor(Color.parseColor("#FFFFFF"))
                    setStroke(dpToPx(1), Color.parseColor("#E2E8F0"))
                }
            }
            val tvL = TextView(ctx).apply {
                text = label
                textSize = 7f
                setTextColor(Color.parseColor("#64748B"))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            val tvV = TextView(ctx).apply {
                text = value
                textSize = 10f
                setTextColor(Color.parseColor(colorHex))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            val tvS = TextView(ctx).apply {
                text = sub
                textSize = 6.5f
                setTextColor(Color.parseColor("#94A3B8"))
            }
            tile.addView(tvL)
            tile.addView(tvV)
            tile.addView(tvS)
            container.addView(tile)
        }

        val variStr = "VARI +${String.format(java.util.Locale.US, "%.2f", est.vegetationIndexVARI)}"
        addBioCard(bioRow, "🌿 CANOPY VIGOUR", variStr, "${String.format(java.util.Locale.US, "%.1f", est.canopyCoveragePct)}% Coverage", "#15803D")
        addBioCard(bioRow, "📊 MATURITY BRIX", "${est.maturityIndexPct}%", est.currentStage.name, "#0284C7")
        addBioCard(bioRow, "🌡️ THERMAL GDD", "${est.accumulatedGDD} GDD", "Target ${est.targetGDD}", "#D97706")
        card.addView(bioRow)

        // 3. Hero Harvest Countdown Banner (High contrast emerald card)
        val harvestHeroBox = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(6) }
            setPadding(dpToPx(10), dpToPx(8), dpToPx(10), dpToPx(8))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.parseColor("#F0FDF4"))
                setStroke(dpToPx(1.5f), Color.parseColor("#86EFAC"))
            }
        }

        val heroTopRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val tvHeroCountdown = TextView(ctx).apply {
            text = if (est.remainingDaysToHarvest <= 0) "🧺 HARVEST READY NOW!" else "🧺 Ready in ~${est.remainingDaysToHarvest} Days"
            textSize = 12.5f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val tvHeroProgressBadge = TextView(ctx).apply {
            text = "${est.progressPercent}% Matured"
            textSize = 9f
            setTextColor(Color.parseColor("#166534"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(6), dpToPx(2), dpToPx(6), dpToPx(2))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(Color.parseColor("#DCFCE7"))
            }
        }

        heroTopRow.addView(tvHeroCountdown)
        heroTopRow.addView(tvHeroProgressBadge)
        harvestHeroBox.addView(heroTopRow)

        // Progress bar inside Hero Box
        val heroProgressBar = ProgressBar(ctx, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = est.progressPercent
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(6)
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(4) }
            android.graphics.PorterDuffColorFilter(
                Color.parseColor("#16A34A"), android.graphics.PorterDuff.Mode.SRC_IN
            ).also { filter -> progressDrawable.colorFilter = filter }
        }
        harvestHeroBox.addView(heroProgressBar)

        val tvCalendarWindow = TextView(ctx).apply {
            text = "📅 Estimated Harvest Window: ${est.harvestCalendarWindow}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        harvestHeroBox.addView(tvCalendarWindow)

        val tvHeroSummary = TextView(ctx).apply {
            text = "${est.detectionSummary}\n🌱 ${est.chlorophyllIndex}"
            textSize = 9f
            setTextColor(Color.parseColor("#374151"))
            setPadding(0, dpToPx(2), 0, 0)
        }
        harvestHeroBox.addView(tvHeroSummary)
        card.addView(harvestHeroBox)

        // 4. 5-Stage Developmental Growth Steps Visualizer
        val tvStagesHeader = TextView(ctx).apply {
            text = "🌿 5 Developmental Lifecycle Stages:"
            textSize = 10f
            setTextColor(Color.parseColor("#0F172A"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(0, dpToPx(2), 0, dpToPx(3))
        }
        card.addView(tvStagesHeader)

        val stageRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
            setPadding(0, 0, 0, dpToPx(4))
        }

        for ((idx, stg) in crop.stages.withIndex()) {
            val isCurrent = (idx == est.currentStageIndex)
            val isPast = (idx < est.currentStageIndex)

            val stageCol = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                gravity = android.view.Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    setMargins(dpToPx(1), 0, dpToPx(1), 0)
                }
                setPadding(dpToPx(2), dpToPx(3), dpToPx(2), dpToPx(3))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(6).toFloat()
                    if (isCurrent) {
                        setColor(Color.parseColor("#DCFCE7"))
                        setStroke(dpToPx(1), Color.parseColor("#16A34A"))
                    } else if (isPast) {
                        setColor(Color.parseColor("#F0FDF4"))
                        setStroke(dpToPx(1), Color.parseColor("#BBF7D0"))
                    } else {
                        setColor(Color.parseColor("#FFFFFF"))
                        setStroke(dpToPx(1), Color.parseColor("#E2E8F0"))
                    }
                }
            }

            val tvStatusTag = TextView(ctx).apply {
                text = when {
                    isCurrent -> "🎯 Active"
                    isPast -> "✅ Done"
                    else -> "⚪ ${idx + 1}"
                }
                textSize = 6.5f
                setTextColor(Color.parseColor(if (isCurrent) "#15803D" else if (isPast) "#16A34A" else "#94A3B8"))
                setTypeface(null, if (isCurrent) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
            }

            val tvIcon = TextView(ctx).apply {
                text = stg.iconEmoji
                textSize = 12f
                gravity = android.view.Gravity.CENTER
            }

            val tvName = TextView(ctx).apply {
                val shortName = when {
                    stg.name.contains("Germination", ignoreCase = true) || stg.name.contains("Sprout", ignoreCase = true) -> "Sprout"
                    stg.name.contains("Vegetative", ignoreCase = true) || stg.name.contains("Foliage", ignoreCase = true) -> "Foliage"
                    stg.name.contains("Flower", ignoreCase = true) || stg.name.contains("Blossom", ignoreCase = true) -> "Bloom"
                    stg.name.contains("Bulking", ignoreCase = true) || stg.name.contains("Sizing", ignoreCase = true) || stg.name.contains("Fruit", ignoreCase = true) || stg.name.contains("Formation", ignoreCase = true) -> "Fruiting"
                    else -> "Harvest"
                }
                text = shortName
                textSize = 7.5f
                setTextColor(if (isCurrent) Color.parseColor("#15803D") else Color.parseColor("#4B5563"))
                setTypeface(null, if (isCurrent) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
                gravity = android.view.Gravity.CENTER
            }

            val tvDays = TextView(ctx).apply {
                text = stg.durationDays.replace(" days", "d").replace(" months", "m")
                textSize = 7f
                setTextColor(Color.parseColor(if (isCurrent) "#15803D" else "#059669"))
                gravity = android.view.Gravity.CENTER
            }

            stageCol.addView(tvStatusTag)
            stageCol.addView(tvIcon)
            stageCol.addView(tvName)
            stageCol.addView(tvDays)
            stageRow.addView(stageCol)
        }
        card.addView(stageRow)

        // 5. Cultivation Vitals 2x2 Grid (Modern Light Instrument Panel)
        val vitalsCard = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(3); bottomMargin = dpToPx(4) }
            setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.parseColor("#F8FAFC"))
                setStroke(dpToPx(1), Color.parseColor("#E2E8F0"))
            }
        }

        val tvVitalsTitle = TextView(ctx).apply {
            text = "📊 Ideal Growing Environment & Vitals"
            textSize = 9.5f
            setTextColor(Color.parseColor("#0284C7"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        vitalsCard.addView(tvVitalsTitle)

        val vitalsGrid = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(3) }
        }

        fun addVitalTile(container: LinearLayout, label: String, value: String, icon: String) {
            val col = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                setPadding(dpToPx(2), dpToPx(1), dpToPx(2), dpToPx(1))
            }
            val tvL = TextView(ctx).apply {
                text = "$icon $label"
                textSize = 7.5f
                setTextColor(Color.parseColor("#64748B"))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            val tvV = TextView(ctx).apply {
                text = value
                textSize = 8.5f
                setTextColor(Color.parseColor("#0F172A"))
                setLineSpacing(0f, 0.95f)
            }
            col.addView(tvL)
            col.addView(tvV)
            container.addView(col)
        }

        addVitalTile(vitalsGrid, "SUNLIGHT", if (crop.sunlight.length > 25) crop.sunlight.take(23) + "..." else crop.sunlight, "☀️")
        addVitalTile(vitalsGrid, "TEMP", if (crop.temperature.length > 25) crop.temperature.take(23) + "..." else crop.temperature, "🌡️")
        addVitalTile(vitalsGrid, "WATER", if (crop.waterNeeds.length > 25) crop.waterNeeds.take(23) + "..." else crop.waterNeeds, "💧")
        addVitalTile(vitalsGrid, "SOIL/pH", if (crop.soilAndPh.length > 25) crop.soilAndPh.take(23) + "..." else crop.soilAndPh, "🌱")

        vitalsCard.addView(vitalsGrid)
        card.addView(vitalsCard)

        // 6. Actionable Harvest Guide & Agronomist Care Card
        val agronomyCard = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(2); bottomMargin = dpToPx(4) }
            setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.parseColor("#FFFFFF"))
                setStroke(dpToPx(1), Color.parseColor("#CBD5E1"))
            }
        }

        val tvAgroHeader = TextView(ctx).apply {
            text = "👨‍🌾 AI Agronomist Field Prescription"
            textSize = 10f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        agronomyCard.addView(tvAgroHeader)

        val tvFeed = TextView(ctx).apply {
            text = "🧪 Feeding: ${crop.npkRatio}"
            textSize = 9f
            setTextColor(Color.parseColor("#0369A1"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(0, dpToPx(2), 0, dpToPx(1))
        }
        agronomyCard.addView(tvFeed)

        val tvRx = TextView(ctx).apply {
            text = "🌿 Prescription: ${est.agronomicPrescription}"
            textSize = 9f
            setTextColor(Color.parseColor("#1E293B"))
            setPadding(0, 0, 0, dpToPx(1))
        }
        agronomyCard.addView(tvRx)

        val tvHarvest = TextView(ctx).apply {
            text = "🧺 When to Pick: ${crop.harvestSigns}"
            textSize = 9f
            setTextColor(Color.parseColor("#334155"))
            setPadding(0, 0, 0, dpToPx(1))
        }
        agronomyCard.addView(tvHarvest)

        val tvTiming = TextView(ctx).apply {
            text = "⏰ Optimal Time: ${crop.optimalPickingWindow} • 🛡️ Disease Watch: ${crop.diseaseWatch}"
            textSize = 8.5f
            setTextColor(Color.parseColor("#475569"))
            setPadding(0, dpToPx(1), 0, 0)
        }
        agronomyCard.addView(tvTiming)
        card.addView(agronomyCard)

        // 7. Interactive Quick Agronomy Actions Bar
        val actionsRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(2) }
        }

        fun addActionBtn(title: String, colorHex: String, onClick: () -> Unit) {
            val btn = MaterialButton(ctx, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                text = title
                textSize = 9f
                setTextColor(Color.parseColor(colorHex))
                strokeColor = android.content.res.ColorStateList.valueOf(Color.parseColor(colorHex))
                strokeWidth = dpToPx(1)
                cornerRadius = dpToPx(6)
                insetTop = 0
                insetBottom = 0
                layoutParams = LinearLayout.LayoutParams(0, dpToPx(34), 1f).apply {
                    setMargins(dpToPx(2), 0, dpToPx(2), 0)
                }
                setPadding(dpToPx(2), 0, dpToPx(2), 0)
                setOnClickListener { onClick() }
            }
            actionsRow.addView(btn)
        }

        addActionBtn("📅 Calendar", "#15803D") {
            AlertDialog.Builder(ctx)
                .setTitle("📅 ${crop.name} Harvest Calendar")
                .setMessage("🎯 Target Window: ${est.harvestCalendarWindow}\n\n⏱️ Days Remaining: ~${est.remainingDaysToHarvest} days\n🌱 Current Phase: ${est.currentStage.name}\n🌿 Total Duration: ${crop.totalDurationRange}\n\n🔔 Add to your device calendar to receive peak harvest alerts.")
                .setPositiveButton("Set Reminder") { _, _ ->
                    Toast.makeText(ctx, "✅ Harvest reminder logged for ${est.harvestCalendarWindow}", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("Close", null)
                .show()
        }

        addActionBtn("💧 Nutrition", "#0284C7") {
            AlertDialog.Builder(ctx)
                .setTitle("🧪 ${crop.name} Agronomy Nutrition")
                .setMessage("💧 Water Strategy: ${crop.waterNeeds}\n\n🧪 NPK Formula: ${crop.npkRatio}\n\n💡 Care Guide: ${crop.careTips}\n\n🛡️ Disease Watch: ${crop.diseaseWatch}")
                .setPositiveButton("Got It", null)
                .show()
        }

        addActionBtn("🔬 Biometrics", "#7C3AED") {
            AlertDialog.Builder(ctx)
                .setTitle("🔬 Computer Vision Biometrics")
                .setMessage("🌿 Vegetation Index (VARI): +${String.format(java.util.Locale.US, "%.2f", est.vegetationIndexVARI)}\n🌱 Canopy Biomass Coverage: ${String.format(java.util.Locale.US, "%.1f", est.canopyCoveragePct)}%\n📊 Maturity / Brix Index: ${est.maturityIndexPct}%\n🌡️ Thermal GDD: ${est.accumulatedGDD} / ${est.targetGDD}\n🧬 Chlorophyll Rating: ${est.chlorophyllIndex}\n🎯 AI Classification Confidence: ${(est.confidence * 100).toInt()}%")
                .setPositiveButton("Done", null)
                .show()
        }

        card.addView(actionsRow)

        return card
    }

    private fun dpToPx(dp: Int): Int =
        (dp * resources.displayMetrics.density + 0.5f).toInt()

    private fun dpToPx(dp: Float): Int =
        (dp * resources.displayMetrics.density + 0.5f).toInt()

    // ==========================================
    // Sample Generator Functions
    // ==========================================

    private fun loadSampleTomatoes() {
        val bmp = Bitmap.createBitmap(600, 450, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.parseColor("#F8FAFC"))

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.parseColor("#E2E8F0")
        canvas.drawRect(0f, 260f, 600f, 450f, paint)

        paint.color = Color.parseColor("#15803D")
        paint.strokeWidth = 10f
        canvas.drawLine(80f, 160f, 520f, 220f, paint)

        paint.color = Color.parseColor("#EF4444")
        canvas.drawOval(RectF(100f, 130f, 280f, 310f), paint)
        canvas.drawOval(RectF(320f, 140f, 500f, 320f), paint)

        paint.color = Color.parseColor("#22C55E")
        canvas.drawOval(RectF(175f, 120f, 205f, 145f), paint)
        canvas.drawOval(RectF(395f, 130f, 425f, 155f), paint)

        processAndAnalyzeBitmap(bmp, "Tomatoes on Vine")
    }

    private fun loadSampleApples() {
        val bmp = Bitmap.createBitmap(600, 450, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.parseColor("#F8FAFC"))

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.parseColor("#E2E8F0")
        canvas.drawRect(0f, 260f, 600f, 450f, paint)

        paint.color = Color.parseColor("#DC2626")
        canvas.drawOval(RectF(120f, 140f, 290f, 310f), paint)
        canvas.drawOval(RectF(320f, 150f, 490f, 320f), paint)

        paint.color = Color.parseColor("#78350F")
        paint.strokeWidth = 6f
        canvas.drawLine(205f, 140f, 195f, 110f, paint)
        canvas.drawLine(405f, 150f, 395f, 120f, paint)

        processAndAnalyzeBitmap(bmp, "Fresh Apples")
    }

    private fun loadSampleBananas() {
        val bmp = Bitmap.createBitmap(600, 450, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.parseColor("#F8FAFC"))

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.parseColor("#E2E8F0")
        canvas.drawRect(0f, 260f, 600f, 450f, paint)

        paint.color = Color.parseColor("#EAB308")
        canvas.drawRoundRect(RectF(140f, 150f, 460f, 300f), 40f, 40f, paint)

        paint.color = Color.parseColor("#854D0E")
        canvas.drawOval(RectF(130f, 200f, 160f, 230f), paint)
        canvas.drawOval(RectF(440f, 220f, 470f, 250f), paint)

        processAndAnalyzeBitmap(bmp, "Banana Bunch")
    }

    private fun loadSampleMixed() {
        val bmp = Bitmap.createBitmap(600, 450, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.parseColor("#F8FAFC"))

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.parseColor("#E2E8F0")
        canvas.drawRect(0f, 260f, 600f, 450f, paint)

        paint.color = Color.parseColor("#10B981")
        canvas.drawRoundRect(RectF(100f, 140f, 260f, 310f), 30f, 30f, paint)

        paint.color = Color.parseColor("#EF4444")
        canvas.drawOval(RectF(320f, 150f, 480f, 310f), paint)

        processAndAnalyzeBitmap(bmp, "Mixed Produce")
    }

    // ==========================================
    // Web Dashboard Management
    // ==========================================

    // ==========================================
    // Advanced Feature Dialogs & Utilities
    // ==========================================

    private fun showCropLibraryDialog() {
        val cropNames = CropGrowthRepository.CROPS.map { "${it.emoji} ${it.name} (${it.category})" }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("📚 Crop Cultivation Library")
            .setItems(cropNames) { _, which ->
                val selected = CropGrowthRepository.CROPS[which]
                showCropDetailDialog(selected)
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showCropDetailDialog(crop: CropCultivationProfile) {
        val stagesSummary = crop.stages.joinToString("\n") {
            "  ${it.iconEmoji} Stage ${it.stageNumber}: ${it.name} (${it.durationDays}) - ${it.progressPct}%"
        }
        val detailMsg = """
            🌱 Scientific Name: ${crop.scientificName}
            📊 Category: ${crop.category} • Difficulty: ${crop.difficulty}
            ⏳ Total Growth Cycle: ${crop.totalDurationRange} (~${crop.averageDays} days)
            
            ☀️ Sunlight: ${crop.sunlight}
            🌡️ Temperature: ${crop.temperature}
            💧 Watering: ${crop.waterNeeds}
            🌱 Soil & pH: ${crop.soilAndPh}
            🧪 Fertilizer Formula: ${crop.npkRatio}
            ⏰ Best Picking Window: ${crop.optimalPickingWindow}
            🛡️ Common Diseases: ${crop.diseaseWatch}
            
            📈 Phenological Stages:
            $stagesSummary
            
            💡 Expert Care Tips:
            ${crop.careTips}
        """.trimIndent()

        AlertDialog.Builder(this)
            .setTitle("${crop.emoji} ${crop.name} Cultivation Guide")
            .setMessage(detailMsg)
            .setPositiveButton("Select This Crop") { _, _ ->
                selectedCropName = crop.name
                if (!isCropGrowthMode) {
                    tabLayout.getTabAt(1)?.select()
                } else {
                    updateFreshnessUI(lastDetectedItems, lastEstimates)
                }
                Toast.makeText(this, "🌱 Active crop set to ${crop.name}", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showPlantDoctorDialog() {
        val crop = CropGrowthRepository.getProfile(selectedCropName ?: "Tomato") ?: CropGrowthRepository.CROPS.first()
        val doctorMsg = """
            🩺 Plant Pathology & Foliage Scouting
            Active Profile: ${crop.emoji} ${crop.name}
            
            🛡️ High-Risk Pathogens & Pests:
            ${crop.diseaseWatch}
            
            🧪 Recommended Action Protocol:
            • Foliage Health: Inspect leaf undersides for fungal mycelium or sap-sucking pests.
            • Chlorophyll Stress: If yellowing (chlorosis) occurs, supplement with magnesium or chelated iron.
            • Humidity & Airflow: Prune lower yellowing canopy foliage to reduce fungal spore dampness.
            • NPK Feeding Schedule: ${crop.npkRatio}
            
            💧 Irrigation Check: ${crop.waterNeeds}
        """.trimIndent()

        AlertDialog.Builder(this)
            .setTitle("🩺 AI Plant Doctor & Agronomist")
            .setMessage(doctorMsg)
            .setPositiveButton("Take Foliage Photo") { _, _ ->
                checkAndLaunchCamera()
            }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun showHarvestTimelineDialog() {
        val crop = CropGrowthRepository.getProfile(selectedCropName ?: "Tomato") ?: CropGrowthRepository.CROPS.first()
        val timelineMsg = """
            🗓️ Harvest Countdown & Growing Timeline
            Crop: ${crop.emoji} ${crop.name} (${crop.totalDurationRange})
            
            🌱 Lifecycle Milestones:
            ${crop.stages.joinToString("\n") { "• ${it.iconEmoji} ${it.name}: ${it.durationDays} (${it.progressPct}% complete)" }}
            
            🧺 Peak Harvest Indicators:
            ${crop.harvestSigns}
            
            ⏰ Optimal Picking Time:
            ${crop.optimalPickingWindow}
        """.trimIndent()

        AlertDialog.Builder(this)
            .setTitle("🗓️ ${crop.name} Growth Timeline")
            .setMessage(timelineMsg)
            .setPositiveButton("Set Harvest Calendar") { _, _ ->
                Toast.makeText(this, "📅 Harvest timeline synced to ${crop.name}", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Done", null)
            .show()
    }

    private fun shareInspectionReport() {
        val isPlantMode = isCropGrowthMode
        val title = if (isPlantMode) "🌱 FreshAI Plant Growth Report" else "🍎 FreshAI Produce Quality & Freshness Report"
        val activeCrop = selectedCropName ?: lastDetectedItems.firstOrNull()?.className ?: "Tomato"
        val p = CropGrowthRepository.getProfile(activeCrop) ?: CropGrowthRepository.CROPS.first()

        val textReport = if (isPlantMode) {
            """
                $title
                ==============================
                Crop: ${p.emoji} ${p.name} (${p.scientificName})
                Category: ${p.category} | Average Lifecycle: ${p.averageDays} days
                Current Focus: ${p.totalDurationRange}
                
                ☀️ Sunlight Needs: ${p.sunlight}
                🌡️ Ideal Temp: ${p.temperature}
                💧 Water Schedule: ${p.waterNeeds}
                🧪 Nutrient Formulation: ${p.npkRatio}
                🧺 Harvest Readiness: ${p.harvestSigns}
                
                Generated by FreshAI Precision Agronomy Suite
            """.trimIndent()
        } else {
            val totalDetected = lastDetectedItems.size
            val itemsSummary = lastDetectedItems.joinToString("\n") { "• ${it.className} (${(it.confidence * 100).toInt()}% confidence)" }
            """
                $title
                ==============================
                Scanned Items Count: $totalDetected
                Detected Produce:
                $itemsSummary
                
                Generated by FreshAI Quality & Shelf-Life Suite
            """.trimIndent()
        }

        val shareIntent = android.content.Intent(android.content.Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(android.content.Intent.EXTRA_SUBJECT, title)
            putExtra(android.content.Intent.EXTRA_TEXT, textReport)
        }
        startActivity(android.content.Intent.createChooser(shareIntent, "Share FreshAI Inspection Report"))
    }

    private val sensorListener = object : android.hardware.SensorEventListener {
        override fun onSensorChanged(event: android.hardware.SensorEvent?) {
            if (event == null) return
            when (event.sensor.type) {
                android.hardware.Sensor.TYPE_LIGHT -> {
                    val lux = event.values[0]
                    currentLuxValue = lux
                    val luxStatus = when {
                        lux < 100f -> "Low Light (Use Flash)"
                        lux in 100f..1000f -> "Optimal Daylight"
                        else -> "Bright Direct Sun"
                    }
                    tvSensorLux.text = "☀️ ${lux.toInt()} Lux ($luxStatus)"
                }
                android.hardware.Sensor.TYPE_ACCELEROMETER -> {
                    val ax = event.values[0]
                    val ay = event.values[1]
                    val az = event.values[2]
                    val gMagnitude = Math.sqrt((ax * ax + ay * ay + az * az).toDouble())
                    val isSteady = Math.abs(gMagnitude - 9.81) < 1.2
                    isDeviceStable = isSteady
                    if (isSteady) {
                        tvSensorStability.text = "🎯 Lens: Stable"
                        tvSensorStability.setTextColor(Color.parseColor("#4ADE80"))
                    } else {
                        tvSensorStability.text = "⚠️ Lens: Shaky"
                        tvSensorStability.setTextColor(Color.parseColor("#F87171"))
                    }
                }
            }
        }

        override fun onAccuracyChanged(sensor: android.hardware.Sensor?, accuracy: Int) {}
    }

    override fun onResume() {
        super.onResume()
        try {
            lightSensor?.let { sensorManager?.registerListener(sensorListener, it, android.hardware.SensorManager.SENSOR_DELAY_UI) }
            accelSensor?.let { sensorManager?.registerListener(sensorListener, it, android.hardware.SensorManager.SENSOR_DELAY_UI) }
        } catch (e: Exception) {}
    }

    override fun onPause() {
        super.onPause()
        try {
            sensorManager?.unregisterListener(sensorListener)
        } catch (e: Exception) {}
    }

    override fun onDestroy() {
        if (::onDeviceDetector.isInitialized) onDeviceDetector.close()
        executor.shutdown()
        super.onDestroy()
    }
}
