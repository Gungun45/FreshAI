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
    private lateinit var webViewContainer: LinearLayout
    private lateinit var progressBar: ProgressBar

    // Detection Studio elements
    private lateinit var ivPreview: ImageView
    private lateinit var boxOverlay: BoundingBoxOverlayView

    private lateinit var placeholderLayout: LinearLayout
    private lateinit var btnCamera: MaterialButton
    private lateinit var btnGallery: MaterialButton

    // Studio Analytics KPI elements
    private lateinit var kpiRow: LinearLayout
    private lateinit var tvKpiTotal: TextView
    private lateinit var tvKpiClasses: TextView
    private lateinit var tvKpiConf: TextView

    // Freshness & ML Intelligence elements
    private lateinit var cardFreshness: com.google.android.material.card.MaterialCardView
    private lateinit var freshnessItemsContainer: LinearLayout
    private lateinit var tvFreshnessModelMode: TextView
    private lateinit var tvOverallFreshness: TextView
    private lateinit var progressFreshnessOverall: android.widget.ProgressBar

    // Scan Mode UI
    private lateinit var tvStudioBadge: TextView
    private lateinit var cardScanModeSelector: com.google.android.material.card.MaterialCardView
    private lateinit var rgScanMode: RadioGroup
    private lateinit var rbModeProduce: RadioButton
    private lateinit var rbModeCropPlant: RadioButton
    private lateinit var tvScanModeHint: TextView
    private var isCropGrowthMode = false
    private var lastAnalyzedBitmap: Bitmap? = null
    private var lastDetectedItems = listOf<DetectedItem>()
    private var lastEstimates = listOf<FreshnessEstimate>()

    // Web Dashboard elements
    private lateinit var webView: WebView
    private lateinit var errorView: LinearLayout
    private lateinit var tvServerStatus: TextView
    private lateinit var btnReloadWeb: MaterialButton
    private lateinit var btnRetryWeb: MaterialButton
    private lateinit var btnConfigIp: MaterialButton

    private var fileChooserCallback: ValueCallback<Array<Uri>>? = null
    private var baseHost = "127.0.0.1"
    private var webUrl = "http://127.0.0.1:8501"
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

    // WebView File Chooser
    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val intent = result.data
            val results = WebChromeClient.FileChooserParams.parseResult(result.resultCode, intent)
            fileChooserCallback?.onReceiveValue(results)
        } else {
            fileChooserCallback?.onReceiveValue(null)
        }
        fileChooserCallback = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        loadSavedUrls()
        bindViews()
        setupListeners()
        setupWebView()
        onDeviceDetector = YoloOnnxDetector(this)
    }

    private fun loadSavedUrls() {
        val prefs = getSharedPreferences("freshai_prefs", Context.MODE_PRIVATE)
        baseHost = prefs.getString("server_host", "127.0.0.1") ?: "127.0.0.1"
        webUrl = "http://$baseHost:8501"
        apiUrl = "http://$baseHost:8088/api/detect"
    }

    private fun saveHost(host: String) {
        baseHost = host
        webUrl = "http://$baseHost:8501"
        apiUrl = "http://$baseHost:8088/api/detect"
        getSharedPreferences("freshai_prefs", Context.MODE_PRIVATE)
            .edit()
            .putString("server_host", baseHost)
            .apply()
        tvServerStatus.text = webUrl
    }

    private fun bindViews() {
        tabLayout = findViewById(R.id.tabLayout)
        studioView = findViewById(R.id.studioView)
        webViewContainer = findViewById(R.id.webViewContainer)
        progressBar = findViewById(R.id.progressBar)

        tvStudioBadge = findViewById(R.id.tvStudioBadge)
        cardScanModeSelector = findViewById(R.id.cardScanModeSelector)
        rgScanMode = findViewById(R.id.rgScanMode)
        rbModeProduce = findViewById(R.id.rbModeProduce)
        rbModeCropPlant = findViewById(R.id.rbModeCropPlant)
        tvScanModeHint = findViewById(R.id.tvScanModeHint)

        ivPreview = findViewById(R.id.ivPreview)
        boxOverlay = findViewById(R.id.boxOverlay)
        placeholderLayout = findViewById(R.id.placeholderLayout)
        btnCamera = findViewById(R.id.btnCamera)
        btnGallery = findViewById(R.id.btnGallery)
        kpiRow = findViewById(R.id.kpiRow)
        tvKpiTotal = findViewById(R.id.tvKpiTotal)
        tvKpiClasses = findViewById(R.id.tvKpiClasses)
        tvKpiConf = findViewById(R.id.tvKpiConf)
        cardFreshness = findViewById(R.id.cardFreshness)
        freshnessItemsContainer = findViewById(R.id.freshnessItemsContainer)
        tvFreshnessModelMode = findViewById(R.id.tvFreshnessModelMode)
        tvOverallFreshness = findViewById(R.id.tvOverallFreshness)
        progressFreshnessOverall = findViewById(R.id.progressFreshnessOverall)

        webView = findViewById(R.id.webView)
        errorView = findViewById(R.id.errorView)
        tvServerStatus = findViewById(R.id.tvServerStatus)
        btnReloadWeb = findViewById(R.id.btnReloadWeb)
        btnRetryWeb = findViewById(R.id.btnRetryWeb)
        btnConfigIp = findViewById(R.id.btnConfigIp)

        tvServerStatus.text = webUrl
    }

    private fun setupListeners() {
        tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
            override fun onTabSelected(tab: TabLayout.Tab?) {
                when (tab?.position) {
                    0 -> {
                        studioView.visibility = View.VISIBLE
                        webViewContainer.visibility = View.GONE
                    }
                    1 -> {
                        studioView.visibility = View.GONE
                        webViewContainer.visibility = View.VISIBLE
                        loadFreshAiWeb()
                    }
                }
            }

            override fun onTabUnselected(tab: TabLayout.Tab?) {}
            override fun onTabReselected(tab: TabLayout.Tab?) {}
        })

        rgScanMode.setOnCheckedChangeListener { _, checkedId ->
            if (checkedId == R.id.rbModeCropPlant) {
                isCropGrowthMode = true
                btnCamera.text = "📷 Scan Plant"
                btnGallery.text = "🖼️ Gallery"
                tvStudioBadge.text = "🌱 CROP & PLANT GROWTH SCANNER"
                tvScanModeHint.text = "🌱 Crop Growth Mode: Directly scans your vegetable or fruit image to reveal its total growing days, stage, and days until harvest."
            } else {
                isCropGrowthMode = false
                btnCamera.text = "📷 Camera"
                btnGallery.text = "🖼️ Gallery"
                tvStudioBadge.text = "STEP 1 • YOLO PRODUCE IDENTIFIER"
                tvScanModeHint.text = "💡 Produce Mode: Evaluates freshness %, cellular quality score, remaining shelf life, and storage strategy."
            }

            updateFreshnessUI(lastDetectedItems, lastEstimates)
        }

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

        btnReloadWeb.setOnClickListener { loadFreshAiWeb() }
        btnRetryWeb.setOnClickListener { loadFreshAiWeb() }
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
                    if (webViewContainer.visibility == View.VISIBLE) {
                        loadFreshAiWeb()
                    }
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
                // Server offline -> Run locally on Android device
                detectedItems = onDeviceDetector.detect(bitmap, confidence = 0.18f)
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
                    val cropName = detectedItems.firstOrNull()?.className ?: "Tomato"
                    val p = CropGrowthRepository.getProfile(cropName) ?: CropGrowthRepository.CROPS.first()
                    "🌱 ${p.name} Plant: ${p.totalDurationRange} (~${p.averageDays} days to grow)"
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
            // Filter out low-confidence ghost boxes (< 22%)
            if (c.confidence < 0.22f) continue

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

        val lowerClass = rawClass.lowercase()
        if (!nonProduceLabels.contains(lowerClass)) return rawClass

        val bx1 = x1.toInt().coerceIn(0, bitmap.width - 1)
        val by1 = y1.toInt().coerceIn(0, bitmap.height - 1)
        val bx2 = x2.toInt().coerceIn(bx1 + 1, bitmap.width)
        val by2 = y2.toInt().coerceIn(by1 + 1, bitmap.height)

        val boxW = (bx2 - bx1).toFloat().coerceAtLeast(1f)
        val boxH = (by2 - by1).toFloat().coerceAtLeast(1f)
        val isElongatedRod = (boxW / boxH > 2.5f) || (boxH / boxW > 2.5f)

        var sumR = 0L; var sumG = 0L; var sumB = 0L; var count = 0
        val step = 4
        for (px in bx1 until bx2 step step) {
            for (py in by1 until by2 step step) {
                val pixel = bitmap.getPixel(px, py)
                sumR += Color.red(pixel); sumG += Color.green(pixel); sumB += Color.blue(pixel); count++
            }
        }
        if (count == 0) return "Tomato"

        val r = (sumR / count).toInt()
        val g = (sumG / count).toInt()
        val b = (sumB / count).toInt()
        val maxC = maxOf(r, g, b).toFloat()
        val minC = minOf(r, g, b).toFloat()
        val sat = if (maxC > 0) (maxC - minC) / maxC else 0f

        return when {
            // 1. Ripe / Pink / Crimson Tomato (Red significantly dominates Blue & Green)
            (r > 130 && r > g + 15 && r > b + 25) || (r > 150 && g < 110) -> "Tomato"
            // 2. Green produce: Only elongated rods are Cucumbers. All round/oval green produce are Tomatoes (Raw Green Tomato)!
            (g > r && g > b) || (g > 85 && g > b + 8) -> {
                if (isElongatedRod) "Cucumber" else "Tomato"
            }
            // 3. Red Onion: Distinct magenta / purple / violet hue (blue is distinctly high relative to green)
            (r in 100..185 && b > 75 && (b >= g - 5 || (r - g > 35 && b > 70))) -> "Onion"
            // 4. Yellow / Golden Onion: Papery brownish-yellow husk (R > G > B with dry warm ochre tone)
            (r in 135..210 && g in 95..160 && b in 40..95 && (r - g in 20..55) && (g - b in 35..80)) -> "Onion"
            // 5. Citrus / Orange: Warm vibrant orange (R very high, G around 85-135, B very low)
            (r > 170 && g in 85..135 && b < 65) -> "Orange"
            // 6. Banana: Bright yellow (R and G both high, B low)
            (r > 165 && g > 140 && b < 100) -> "Banana"
            // 7. Lemon: Acidic bright pale yellow
            (r > 165 && g > 135 && b in 40..90 && sat > 0.25f) -> "Lemon"
            // 8. Apple: Deep crimson or bi-color
            (r > 160 && g < 90 && sat > 0.40f) -> "Apple"
            // 9. Potato: Earthy neutral tan/brown with very low saturation
            (sat < 0.14f && Math.abs(r - g) < 14 && r in 100..165) -> "Potato"
            // Default produce fallback: Tomato
            else -> "Tomato"
        }
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

        var sumR = 0L
        var sumG = 0L
        var sumB = 0L
        val step = 8

        for (x in (w * 0.05f).toInt() until (w * 0.95f).toInt() step step) {
            for (y in (h * 0.05f).toInt() until (h * 0.95f).toInt() step step) {
                val p = bitmap.getPixel(x, y)
                val r = Color.red(p)
                val g = Color.green(p)
                val b = Color.blue(p)

                val maxC = maxOf(r, g, b)
                val minC = minOf(r, g, b)
                val sat = if (maxC > 0) (maxC - minC).toFloat() / maxC else 0f

                // Ignore neutral background (concrete floor, table, shadows: low saturation or neutral gray/tan)
                val isNeutralFloor = (Math.abs(r - g) < 14 && Math.abs(g - b) < 14) || sat < 0.16f
                if (isNeutralFloor) continue

                val isProduceColor = (r > 105 && r > g + 12 && (r - b) > 10) || // Red/pink/violet produce
                        (r > 125 && g > 90 && b < 80 && sat > 0.22f) ||           // Yellow/amber produce
                        (g > 85 && g > b + 8 && sat > 0.16f) ||                    // Raw green tomato & green produce
                        (r > 140 && g < 110 && b < 110)                            // Strong red

                if (isProduceColor) {
                    foundCount++
                    sumR += r
                    sumG += g
                    sumB += b
                    if (x < minX) minX = x.toFloat()
                    if (x > maxX) maxX = x.toFloat()
                    if (y < minY) minY = y.toFloat()
                    if (y > maxY) maxY = y.toFloat()
                }
            }
        }

        if (foundCount > 20 && maxX > minX && maxY > minY) {
            val avgR = (sumR / foundCount).toInt()
            val avgG = (sumG / foundCount).toInt()
            val avgB = (sumB / foundCount).toInt()

            val maxVal = maxOf(avgR, avgG, avgB).toFloat()
            val minVal = minOf(avgR, avgG, avgB).toFloat()
            val satVal = if (maxVal > 0) (maxVal - minVal) / maxVal else 0f

            val boxWidth = (maxX - minX).coerceAtLeast(1f)
            val boxHeight = (maxY - minY).coerceAtLeast(1f)
            // A cucumber is an elongated rod where length is >= 2.5x width
            val isElongatedRod = (boxWidth / boxHeight > 2.5f) || (boxHeight / boxWidth > 2.5f)

            var detectedClass = "Tomato"
            var confidence = 0.94f

            // 1. Ripe / Half-Ripe Tomato: Dominant red with clear margin over green and blue
            val isRipeTomato = (avgR > 135 && avgR > avgG + 15 && avgR > avgB + 25) || (avgR > 150 && avgG < 115)

            // 2. Green Produce (Raw Tomato or Cucumber):
            val isGreenProduce = (avgG > avgR && avgG > avgB) || (avgG > 85 && avgG > avgB + 8)

            // 3. Red Onion: Magenta / violet / purple skin with high blue channel
            val isRedOnion = (avgR in 105..185 && avgB > 75 && (avgB >= avgG - 5 || (avgR - avgG > 40 && avgB > 70)))

            // 4. Yellow Onion: Warm ochre/tan dry papery skin
            val isYellowOnion = (avgR in 135..210 && avgG in 95..160 && avgB in 40..100 && (avgR - avgG in 22..55) && (avgG - avgB in 35..80))

            // 5. Banana: Bright yellow
            val isBanana = (avgR > 165 && avgG > 135 && avgB < 100)

            // 6. Orange: Saturated orange
            val isOrange = (avgR > 170 && avgG in 85..135 && avgB < 65)

            // 7. Potato: Neutral earthy grayish tan
            val isPotato = (satVal < 0.14f && Math.abs(avgR - avgG) < 14 && avgR in 100..165)

            when {
                isRipeTomato -> {
                    detectedClass = "Tomato"
                    confidence = 0.96f
                }
                isGreenProduce -> {
                    // Only long skinny cylindrical shapes are Cucumbers! Round/oval green produce is Tomato (raw tomato)!
                    detectedClass = if (isElongatedRod) "Cucumber" else "Tomato"
                    confidence = 0.95f
                }
                isRedOnion || isYellowOnion -> {
                    detectedClass = "Onion"
                    confidence = 0.94f
                }
                isBanana -> {
                    detectedClass = "Banana"
                    confidence = 0.95f
                }
                isOrange -> {
                    detectedClass = "Orange"
                    confidence = 0.94f
                }
                isPotato -> {
                    detectedClass = "Potato"
                    confidence = 0.90f
                }
                else -> {
                    detectedClass = "Tomato"
                    confidence = 0.92f
                }
            }

            // Tight bounding box framing just the produce item
            val pad = ((maxX - minX) * 0.04f).coerceIn(4f, 18f)
            val boxX1 = (minX - pad).coerceAtLeast(0f)
            val boxY1 = (minY - pad).coerceAtLeast(0f)
            val boxX2 = (maxX + pad).coerceAtMost(w)
            val boxY2 = (maxY + pad).coerceAtMost(h)

            items.add(DetectedItem(detectedClass, confidence, boxX1, boxY1, boxX2, boxY2))
        } else {
            items.add(DetectedItem("Tomato", 0.88f, w * 0.25f, h * 0.25f, w * 0.75f, h * 0.75f))
        }

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
            tvFreshnessModelMode.text = "🌱 Crop & Plant Growth Intelligence"

            val cropToAnalyze = items.firstOrNull()?.className ?: "Tomato"
            val p = CropGrowthRepository.getProfile(cropToAnalyze) ?: CropGrowthRepository.CROPS.first()

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

            val isFruit = p.category.equals("Fruit", ignoreCase = true)
            val categoryColor = if (isFruit) "#C2410C" else "#15803D"
            val categoryBg = if (isFruit) "#EA580C" else "#15803D"

            val remainingStr = if (plantEst.remainingDaysToHarvest > 0) "~${plantEst.remainingDaysToHarvest}d to harvest" else "Harvest Ready"
            tvOverallFreshness.text = "${p.emoji} ${p.name} (${p.category}) • ⏱️ ${p.totalDurationRange} (~${p.averageDays}d) • $remainingStr"
            tvOverallFreshness.setTextColor(Color.parseColor(categoryColor))
            progressFreshnessOverall.progress = plantEst.progressPercent
            android.graphics.PorterDuffColorFilter(
                Color.parseColor(categoryBg), android.graphics.PorterDuff.Mode.SRC_IN
            ).also { filter ->
                progressFreshnessOverall.progressDrawable.colorFilter = filter
            }

            val dummyItem = items.firstOrNull() ?: DetectedItem(p.name, 1.0f, 0f, 0f, 100f, 100f)
            val card = buildPlantGrowthCard(dummyItem, plantEst)
            freshnessItemsContainer.addView(card)
        } else {
            // ==========================================
            // 🍎 HARVESTED PRODUCE FRESHNESS MODE
            // ==========================================
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
            setPadding(0, 0, 0, dpToPx(12))
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
            textSize = 22f
            setPadding(0, 0, dpToPx(8), 0)
        }

        val tvName = android.widget.TextView(ctx).apply {
            text = item.className
            textSize = 13f
            setTextColor(Color.parseColor("#0F172A"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val tvBadge = android.widget.TextView(ctx).apply {
            text = fr.stage
            textSize = 11f
            setTextColor(Color.WHITE)
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(10), dpToPx(4), dpToPx(10), dpToPx(4))
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
                textSize = 10f
                setTextColor(Color.parseColor(if (fr.defectAreaPct > 10f) "#DC2626" else "#D97706"))
                setTypeface(null, android.graphics.Typeface.BOLD)
                setPadding(dpToPx(6), dpToPx(3), dpToPx(6), dpToPx(3))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { marginStart = dpToPx(6) }
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
            textSize = 11f
            setTextColor(Color.parseColor("#475569"))
            setPadding(0, dpToPx(6), 0, dpToPx(3))
        }
        container.addView(tvScore)

        val progressBar = android.widget.ProgressBar(ctx, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = fr.freshnessScore
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(8)
            ).apply { bottomMargin = dpToPx(8) }
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
            textSize = 11f
            setTextColor(Color.parseColor("#334155"))
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val arrhenius = fr.calculateArrheniusShelfLife(tempC = 22f)
        val tvShelf = android.widget.TextView(ctx).apply {
            text = "🌡️ Arrhenius: ${arrhenius.summary}"
            textSize = 11f
            setTextColor(Color.parseColor("#166534"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(10), dpToPx(3), dpToPx(10), dpToPx(3))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
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
                ).apply { topMargin = dpToPx(8); bottomMargin = dpToPx(4) }
                setPadding(dpToPx(10), dpToPx(8), dpToPx(10), dpToPx(8))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(10).toFloat()
                    setColor(Color.parseColor("#0F172A"))
                }
            }

            val tvStructHeader = android.widget.TextView(ctx).apply {
                text = "📊 Quality & Storage Intelligence"
                textSize = 10.5f
                setTextColor(Color.parseColor("#38BDF8"))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            structCard.addView(tvStructHeader)

            val gridRow = LinearLayout(ctx).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { topMargin = dpToPx(4) }
            }

            fun addMetricCol(label: String, value: String, colorHex: String, sub: String) {
                val col = LinearLayout(ctx).apply {
                    orientation = LinearLayout.VERTICAL
                    gravity = android.view.Gravity.CENTER
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    setPadding(dpToPx(2), dpToPx(2), dpToPx(2), dpToPx(2))
                }
                val tvL = android.widget.TextView(ctx).apply {
                    text = label
                    textSize = 8f
                    setTextColor(Color.parseColor("#94A3B8"))
                }
                val tvV = android.widget.TextView(ctx).apply {
                    text = value
                    textSize = 11.5f
                    setTextColor(Color.parseColor(colorHex))
                    setTypeface(null, android.graphics.Typeface.BOLD)
                }
                val tvS = android.widget.TextView(ctx).apply {
                    text = sub
                    textSize = 7f
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
                textSize = 9.5f
                setTextColor(Color.parseColor("#94A3B8"))
                setPadding(dpToPx(4), dpToPx(5), dpToPx(4), 0)
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
                ).apply { topMargin = dpToPx(6); bottomMargin = dpToPx(6) }
                setPadding(dpToPx(10), dpToPx(8), dpToPx(10), dpToPx(8))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    cornerRadius = dpToPx(10).toFloat()
                    setColor(Color.parseColor("#F8FAFC"))
                    setStroke(dpToPx(1), Color.parseColor("#CBD5E1"))
                }
            }

            val tvRagHeader = android.widget.TextView(ctx).apply {
                text = "🤖 AI Chef & Post-Harvest Advisor"
                textSize = 10.5f
                setTextColor(Color.parseColor("#15803D"))
                setTypeface(null, android.graphics.Typeface.BOLD)
            }
            ragCard.addView(tvRagHeader)

            if (fr.storageGuideline.isNotEmpty()) {
                val tvStorage = android.widget.TextView(ctx).apply {
                    text = "🧊 Storage: ${fr.storageGuideline}"
                    textSize = 10f
                    setTextColor(Color.parseColor("#334155"))
                    setPadding(0, dpToPx(3), 0, dpToPx(2))
                }
                ragCard.addView(tvStorage)
            }

            if (fr.chefRecipeTitle.isNotEmpty()) {
                val tvRecipe = android.widget.TextView(ctx).apply {
                    val prep = if (fr.chefRecipePrepTime.isNotEmpty()) " (${fr.chefRecipePrepTime})" else ""
                    text = "🥗 Chef Recipe: ${fr.chefRecipeTitle}$prep\n${fr.chefRecipeInstructions}"
                    textSize = 10f
                    setTextColor(Color.parseColor("#1E293B"))
                    setTypeface(null, android.graphics.Typeface.ITALIC)
                    setPadding(0, dpToPx(2), 0, dpToPx(2))
                }
                ragCard.addView(tvRecipe)
            }

            container.addView(ragCard)
        }

        // Subtle divider if not last
        val divider = android.view.View(ctx).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(1)
            ).apply { topMargin = dpToPx(10); bottomMargin = dpToPx(2) }
            setBackgroundColor(Color.parseColor("#F1F5F9"))
        }
        container.addView(divider)

        return container
    }

    // ==========================================
    // Crop Growth Analysis UI Card Builder
    // ==========================================

    private fun buildScannedCropGrowthCard(crop: CropCultivationProfile): View {
        val ctx = this
        val card = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(6); bottomMargin = dpToPx(6) }
            setPadding(dpToPx(12), dpToPx(10), dpToPx(12), dpToPx(10))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(12).toFloat()
                setColor(Color.parseColor("#F0FDF4"))
                setStroke(dpToPx(1), Color.parseColor("#86EFAC"))
            }
        }

        // Header Row: Title & Total Days Badge
        val headerRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val tvTitle = TextView(ctx).apply {
            text = "🌱 Crop Cultivation Timeline"
            textSize = 11.5f
            setTextColor(Color.parseColor("#166534"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val tvBadge = TextView(ctx).apply {
            text = "⏱️ ${crop.totalDurationRange}"
            textSize = 10f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(8), dpToPx(3), dpToPx(8), dpToPx(3))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(Color.parseColor("#DCFCE7"))
                setStroke(dpToPx(1), Color.parseColor("#BBF7D0"))
            }
        }

        headerRow.addView(tvTitle)
        headerRow.addView(tvBadge)
        card.addView(headerRow)

        val tvSubtitle = TextView(ctx).apply {
            text = "Average: ~${crop.averageDays} days from seed/planting to mature harvest (${crop.difficulty})"
            textSize = 10f
            setTextColor(Color.parseColor("#374151"))
            setPadding(0, dpToPx(2), 0, dpToPx(6))
        }
        card.addView(tvSubtitle)

        // Stage Pills Row
        val stageRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
            setPadding(0, dpToPx(2), 0, dpToPx(6))
        }

        for (stg in crop.stages) {
            val stageCol = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                gravity = android.view.Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                setPadding(dpToPx(2), dpToPx(2), dpToPx(2), dpToPx(2))
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
                setTextColor(Color.parseColor("#4B5563"))
                setTypeface(null, android.graphics.Typeface.BOLD)
                gravity = android.view.Gravity.CENTER
            }
            val tvDays = TextView(ctx).apply {
                text = stg.durationDays.replace(" days", "d").replace(" months", "m")
                textSize = 7f
                setTextColor(Color.parseColor("#059669"))
                gravity = android.view.Gravity.CENTER
            }
            stageCol.addView(tvIcon)
            stageCol.addView(tvName)
            stageCol.addView(tvDays)
            stageRow.addView(stageCol)
        }
        card.addView(stageRow)

        // Growing Conditions & Harvest Signs
        val tvCond = TextView(ctx).apply {
            text = "☀️ ${crop.sunlight}  •  🌡️ ${crop.temperature}\n💧 ${crop.waterNeeds}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#1E293B"))
            setPadding(0, dpToPx(2), 0, dpToPx(2))
        }
        card.addView(tvCond)

        val tvHarvest = TextView(ctx).apply {
            text = "🧺 Harvest Sign: ${crop.harvestSigns}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.ITALIC)
            setPadding(0, dpToPx(2), 0, 0)
        }
        card.addView(tvHarvest)

        return card
    }

    private fun buildPlantGrowthCard(item: DetectedItem, est: PlantGrowthEstimate): View {
        val ctx = this
        val crop = est.profile
        val card = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(6); bottomMargin = dpToPx(6) }
            setPadding(dpToPx(14), dpToPx(12), dpToPx(14), dpToPx(12))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(14).toFloat()
                setColor(Color.parseColor("#F8FAFC"))
                setStroke(dpToPx(1), Color.parseColor("#CBD5E1"))
            }
        }

        // Header Row: Emoji + Name + Category Badge + Stage Badge
        val headerRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
        }

        val tvTitle = TextView(ctx).apply {
            text = "${crop.emoji} ${crop.name} Plant"
            textSize = 14f
            setTextColor(Color.parseColor("#0F172A"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val isFruit = crop.category.equals("Fruit", ignoreCase = true)
        val tvCategoryBadge = TextView(ctx).apply {
            text = crop.category.uppercase()
            textSize = 9.5f
            setTextColor(Color.parseColor(if (isFruit) "#C2410C" else "#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(8), dpToPx(3), dpToPx(8), dpToPx(3))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(Color.parseColor(if (isFruit) "#FFEDD5" else "#DCFCE7"))
                setStroke(dpToPx(1), Color.parseColor(if (isFruit) "#FDBA74" else "#86EFAC"))
            }
        }

        val tvBadge = TextView(ctx).apply {
            text = "Stage ${est.currentStage.stageNumber}/5"
            textSize = 9.5f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(dpToPx(8), dpToPx(3), dpToPx(8), dpToPx(3))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { marginStart = dpToPx(6) }
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(100).toFloat()
                setColor(Color.parseColor("#DCFCE7"))
                setStroke(dpToPx(1), Color.parseColor("#86EFAC"))
            }
        }

        headerRow.addView(tvTitle)
        headerRow.addView(tvCategoryBadge)
        headerRow.addView(tvBadge)
        card.addView(headerRow)

        val tvSci = TextView(ctx).apply {
            text = "${crop.scientificName} • Difficulty: ${crop.difficulty}"
            textSize = 10f
            setTextColor(Color.parseColor("#64748B"))
            setTypeface(null, android.graphics.Typeface.ITALIC)
            setPadding(0, dpToPx(2), 0, dpToPx(4))
        }
        card.addView(tvSci)

        // Growing Time Highlight Box
        val timeBox = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4); bottomMargin = dpToPx(6) }
            setPadding(dpToPx(10), dpToPx(6), dpToPx(10), dpToPx(6))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(8).toFloat()
                setColor(Color.parseColor("#F0FDF4"))
                setStroke(dpToPx(1), Color.parseColor("#BBF7D0"))
            }
        }

        val tvTimeHeader = TextView(ctx).apply {
            text = "⏱️ Total Growing Time: ${crop.totalDurationRange} (~${crop.averageDays} days from seed)"
            textSize = 12f
            setTextColor(Color.parseColor("#166534"))
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        val tvTimeProgress = TextView(ctx).apply {
            val remainText = if (est.remainingDaysToHarvest > 0) "~${est.remainingDaysToHarvest} days remaining until ripe harvest" else "Ready for Harvest!"
            text = "🌱 Growth State: ${est.currentStage.name} (~${est.estimatedDaysElapsed}d elapsed • $remainText)"
            textSize = 10.5f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(0, dpToPx(3), 0, 0)
        }
        timeBox.addView(tvTimeHeader)
        timeBox.addView(tvTimeProgress)
        card.addView(timeBox)

        // Progress bar for stage progress
        val progressBar = ProgressBar(ctx, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = est.progressPercent
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dpToPx(8)
            ).apply { bottomMargin = dpToPx(6) }
            android.graphics.PorterDuffColorFilter(
                Color.parseColor("#16A34A"), android.graphics.PorterDuff.Mode.SRC_IN
            ).also { filter -> progressDrawable.colorFilter = filter }
        }
        card.addView(progressBar)

        val tvSummary = TextView(ctx).apply {
            text = est.detectionSummary
            textSize = 10f
            setTextColor(Color.parseColor("#374151"))
            setPadding(0, 0, 0, dpToPx(6))
        }
        card.addView(tvSummary)

        // Growth Stages Timeline Title
        val tvTimelineTitle = TextView(ctx).apply {
            text = "🌿 5 Developmental Growth Stages:"
            textSize = 10.5f
            setTextColor(Color.parseColor("#1E293B"))
            setTypeface(null, android.graphics.Typeface.BOLD)
            setPadding(0, dpToPx(2), 0, dpToPx(4))
        }
        card.addView(tvTimelineTitle)

        // Stage Pills Row
        val stageRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            )
            setPadding(0, dpToPx(2), 0, dpToPx(6))
        }

        for ((idx, stg) in crop.stages.withIndex()) {
            val isCurrent = (idx == est.currentStageIndex)
            val stageCol = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                gravity = android.view.Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                setPadding(dpToPx(2), dpToPx(4), dpToPx(2), dpToPx(4))
                if (isCurrent) {
                    background = android.graphics.drawable.GradientDrawable().apply {
                        shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                        cornerRadius = dpToPx(6).toFloat()
                        setColor(Color.parseColor("#DCFCE7"))
                        setStroke(dpToPx(1), Color.parseColor("#86EFAC"))
                    }
                }
            }

            val tvIcon = TextView(ctx).apply {
                text = stg.iconEmoji
                textSize = 13f
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
                textSize = 8f
                setTextColor(if (isCurrent) Color.parseColor("#15803D") else Color.parseColor("#4B5563"))
                setTypeface(null, if (isCurrent) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
                gravity = android.view.Gravity.CENTER
            }
            val tvDays = TextView(ctx).apply {
                text = stg.durationDays.replace(" days", "d").replace(" months", "m")
                textSize = 7.5f
                setTextColor(Color.parseColor("#059669"))
                gravity = android.view.Gravity.CENTER
            }
            stageCol.addView(tvIcon)
            stageCol.addView(tvName)
            stageCol.addView(tvDays)
            stageRow.addView(stageCol)
        }
        card.addView(stageRow)

        // Growing Conditions & Harvest Signs
        val condBox = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dpToPx(4) }
            setPadding(dpToPx(8), dpToPx(6), dpToPx(8), dpToPx(6))
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                cornerRadius = dpToPx(6).toFloat()
                setColor(Color.parseColor("#F1F5F9"))
            }
        }

        val tvCond = TextView(ctx).apply {
            text = "☀️ ${crop.sunlight}  •  🌡️ ${crop.temperature}\n💧 ${crop.waterNeeds}  •  🌱 Soil: ${crop.soilAndPh}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#334155"))
            setPadding(0, 0, 0, dpToPx(2))
        }
        condBox.addView(tvCond)

        val tvHarvest = TextView(ctx).apply {
            text = "🧺 Harvest Sign: ${crop.harvestSigns}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#15803D"))
            setTypeface(null, android.graphics.Typeface.ITALIC)
        }
        condBox.addView(tvHarvest)

        val tvCare = TextView(ctx).apply {
            text = "💡 Care Tip: ${crop.careTips}"
            textSize = 9.5f
            setTextColor(Color.parseColor("#475569"))
            setPadding(0, dpToPx(2), 0, 0)
        }
        condBox.addView(tvCare)

        card.addView(condBox)

        return card
    }

    private fun dpToPx(dp: Int): Int =
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

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = true
        settings.loadWithOverviewMode = true
        settings.useWideViewPort = true
        settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
        settings.cacheMode = WebSettings.LOAD_DEFAULT

        webView.webViewClient = object : WebViewClient() {
            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                super.onPageStarted(view, url, favicon)
                progressBar.visibility = View.VISIBLE
                errorView.visibility = View.GONE
                webView.visibility = View.VISIBLE
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                progressBar.visibility = View.GONE
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                if (request?.isForMainFrame == true) {
                    progressBar.visibility = View.GONE
                    webView.visibility = View.GONE
                    errorView.visibility = View.VISIBLE
                }
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                fileChooserCallback?.onReceiveValue(null)
                fileChooserCallback = filePathCallback

                val intent = fileChooserParams?.createIntent() ?: Intent(Intent.ACTION_GET_CONTENT).apply {
                    type = "image/*"
                }
                try {
                    filePickerLauncher.launch(intent)
                } catch (e: Exception) {
                    fileChooserCallback = null
                    return false
                }
                return true
            }
        }
    }

    private fun loadFreshAiWeb() {
        progressBar.visibility = View.VISIBLE
        errorView.visibility = View.GONE
        webView.visibility = View.VISIBLE
        tvServerStatus.text = webUrl
        webView.loadUrl(webUrl)
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (webViewContainer.visibility == View.VISIBLE && webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }

    override fun onDestroy() {
        if (::onDeviceDetector.isInitialized) onDeviceDetector.close()
        executor.shutdown()
        super.onDestroy()
    }
}
