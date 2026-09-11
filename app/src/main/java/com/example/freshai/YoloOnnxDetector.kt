package com.example.freshai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** Offline YOLOv8 detector bundled with the application. */
class YoloOnnxDetector(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val names: Array<String>

    init {
        val bytes = context.assets.open("freshai_yolo.onnx").use { it.readBytes() }
        session = environment.createSession(bytes, OrtSession.SessionOptions())

        val outInfo = session.outputInfo.values.firstOrNull()?.info as? ai.onnxruntime.TensorInfo
        val outShape = outInfo?.shape
        val numClasses = if (outShape != null && outShape.size >= 2) (outShape[1] - 4).toInt() else 12

        val labelFile = if (numClasses <= 3) "freshai_labels.txt" else "freshai_classes.txt"
        names = context.assets.open(labelFile).bufferedReader().useLines { lines ->
            // UTF-8 label assets can contain a byte-order mark on their first line.
            // Treat it as metadata, never as part of the class name.
            lines.map { it.trim().removePrefix("\uFEFF") }
                .filter { it.isNotEmpty() }
                .toList()
                .toTypedArray()
        }
    }

    fun detect(bitmap: Bitmap, confidence: Float = 0.15f, iou: Float = 0.35f): List<DetectedItem> {
        val inputSize = 640
        val srcW = bitmap.width.toFloat()
        val srcH = bitmap.height.toFloat()

        // Letterbox scale preserving aspect ratio
        val r = minOf(inputSize / srcW, inputSize / srcH)
        val newW = (srcW * r).toInt().coerceAtLeast(1)
        val newH = (srcH * r).toInt().coerceAtLeast(1)
        val padX = (inputSize - newW) / 2f
        val padY = (inputSize - newH) / 2f

        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val letterboxed = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(letterboxed)
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(scaled, padX, padY, null)

        val pixels = IntArray(inputSize * inputSize)
        letterboxed.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        val input = FloatArray(3 * inputSize * inputSize)
        for (i in pixels.indices) {
            val pixel = pixels[i]
            input[i] = ((pixel shr 16) and 0xff) / 255f
            input[inputSize * inputSize + i] = ((pixel shr 8) and 0xff) / 255f
            input[2 * inputSize * inputSize + i] = (pixel and 0xff) / 255f
        }

        val tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, 3, 640, 640))
        tensor.use { t ->
            session.run(mapOf(session.inputNames.first() to t)).use { output ->
                @Suppress("UNCHECKED_CAST")
                val values = output[0].value as Array<Array<FloatArray>> // [1, 16, 8400]
                val candidates = mutableListOf<DetectedItem>()
                val numBoxes = values[0][0].size

                for (boxIndex in 0 until numBoxes) {
                    var classId = -1
                    var best = 0f
                    val maxClasses = minOf(names.size, values[0].size - 4)
                    for (classIndex in 0 until maxClasses) {
                        val score = values[0][4 + classIndex][boxIndex]
                        if (score > best) { best = score; classId = classIndex }
                    }
                    if (best < confidence || classId < 0) continue

                    val cx = values[0][0][boxIndex]
                    val cy = values[0][1][boxIndex]
                    val w = values[0][2][boxIndex]
                    val h = values[0][3][boxIndex]

                    // Unpad coordinates back to original image space
                    val unpadX1 = ((cx - w / 2f) - padX) / r
                    val unpadY1 = ((cy - h / 2f) - padY) / r
                    val unpadX2 = ((cx + w / 2f) - padX) / r
                    val unpadY2 = ((cy + h / 2f) - padY) / r

                    val rawName = names[classId]
                    val x1 = unpadX1.coerceIn(0f, srcW)
                    val y1 = unpadY1.coerceIn(0f, srcH)
                    val x2 = unpadX2.coerceIn(0f, srcW)
                    val y2 = unpadY2.coerceIn(0f, srcH)

                    val (finalName, confOverride) = disambiguateCrop(bitmap, x1, y1, x2, y2, rawName)

                    candidates += DetectedItem(
                        finalName,
                        if (finalName != rawName) maxOf(best, confOverride) else best,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                }
                return nms(candidates, iou)
            }
        }
    }

    private fun disambiguateCrop(bitmap: Bitmap, x1: Float, y1: Float, x2: Float, y2: Float, rawName: String): Pair<String, Float> {
        val bx1 = x1.toInt().coerceIn(0, bitmap.width - 1)
        val by1 = y1.toInt().coerceIn(0, bitmap.height - 1)
        val bx2 = x2.toInt().coerceIn(bx1 + 1, bitmap.width)
        val by2 = y2.toInt().coerceIn(by1 + 1, bitmap.height)

        var count = 0
        var tomatoRedCount = 0
        var breakerOrangeCount = 0
        var greenCount = 0
        var onionVioletCount = 0   // anthocyanin violet/magenta OR dark maroon onion skin
        var onionGoldCount = 0

        val hsvTemp = FloatArray(3)

        // ─── TOP-ZONE NECK SCAN ─────────────────────────────────────────────────────
        // Onions always have a dry papery ochre/tan neck/stalk at the TOP of the object.
        // Tomato tops are either RED (h<20°) or GREEN calyx (h>65°) — never warm ochre.
        // So if top 22% of bbox is predominantly warm-ochre, the object is an ONION.
        val bboxH = by2 - by1
        val topEnd = (by1 + bboxH * 0.22f).toInt().coerceIn(by1 + 1, by2)
        var neckOchre = 0; var neckTotal = 0
        val topStep = maxOf(1, (bx2 - bx1) / 14)
        for (nx in bx1 until bx2 step topStep) {
            for (ny in by1 until topEnd step topStep) {
                val p = bitmap.getPixel(nx, ny)
                val nr = Color.red(p); val ng = Color.green(p); val nb = Color.blue(p)
                if ((nr > 248 && ng > 248 && nb > 248) || (nr < 10 && ng < 10 && nb < 10)) { neckTotal++; continue }
                Color.RGBToHSV(nr, ng, nb, hsvTemp)
                val nh = hsvTemp[0]; val ns = hsvTemp[1]; val nv = hsvTemp[2]
                // Warm ochre/golden/tan: hue 20–68°, r > g > b (warm), NOT red and NOT green
                if (nh in 20f..68f && ns in 0.10f..0.82f && nv in 0.14f..0.82f && nr > ng && ng > nb && (nr - ng) > 7 && (ng - nb) > 5) {
                    neckOchre++
                }
                neckTotal++
            }
        }
        val neckOchrePct = if (neckTotal > 2) neckOchre.toFloat() / neckTotal else 0f
        // ───────────────────────────────────────────────────────────────────────────


        // Keep roughly 1,600 samples regardless of the detection-box size.
        // The previous area / 1600 formula made the *linear* step enormous for
        // wide boxes (the onion in the reported scan was sampled only twice).
        val sampleArea = (bx2 - bx1).toDouble() * (by2 - by1).toDouble()
        val step = maxOf(1, kotlin.math.sqrt(sampleArea / 1600.0).toInt())

        for (px in bx1 until bx2 step step) {
            for (py in by1 until by2 step step) {
                val pixel = bitmap.getPixel(px, py)
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                Color.RGBToHSV(r, g, b, hsvTemp)
                val h = hsvTemp[0]
                val s = hsvTemp[1]
                val v = hsvTemp[2]

                // Skip floor / background (peach/tan tiles, concrete, marble, shadows)
                if ((s < 0.24f && v > 0.45f) ||
                    (Math.abs(r - g) < 22 && Math.abs(g - b) < 22 && s < 0.25f) ||
                    (r > 248 && g > 248 && b > 248) || (r < 15 && g < 15 && b < 15) ||
                    v < 0.08f || v > 0.96f) continue

                // ── CRITICAL ORDER: Dark maroon onion check FIRST ──────────────────────
                // A dark red onion has v ~ 0.35–0.62 (muted, NOT vivid)
                // A fresh ripe tomato has v > 0.62 (vivid bright red)
                // By checking this BEFORE isTomato, we ensure dark onion pixels go to onionVioletCount,
                // and only the VIVID BRIGHT pixels are counted as tomato.

                // 1. Dark maroon onion skin (red hue but muted brightness — v < 0.62, r < 192)
                //    Put FIRST so it steals pixels before isTomato fires
                if ((h <= 22f || h >= 340f) && s in 0.28f..0.84f && v in 0.14f..0.62f &&
                    r < 192 && r > g + 10) {
                    onionVioletCount++
                }
                // 2. Classic purple/violet onion — anthocyanin hue 260-350°
                else if (h in 260f..350f && s >= 0.18f && v >= 0.14f) {
                    onionVioletCount++
                }
                // 3. TRUE BRIGHT RIPE TOMATO — only vivid, high-value red reaches here
                //    (Anything with v < 0.62 or r < 192 was already caught by dark maroon above)
                else if ((h <= 18f || h >= 345f) && s >= 0.45f && v >= 0.55f &&
                         r > 155 && r > g + 45 && r > b + 45 && b < 85) {
                    tomatoRedCount++
                }
                // 4. Breaker/turning tomato — orange blush (r/b > 3.0 separates from onion golden skin)
                else if (h in 5f..30f && s >= 0.45f && v >= 0.35f &&
                         r > 152 && r > g + 30 && b < 75 && (r.toFloat() / maxOf(1, b).toFloat()) > 3.0f) {
                    breakerOrangeCount++
                }
                // 5. Green foliage / sepals / vine
                else if ((g > r && g > b) || (h in 65f..160f && s > 0.18f)) {
                    greenCount++
                }
                // 6. Yellow/brown onion DRY PAPERY HUSK
                else if (h in 18f..48f && s in 0.20f..0.75f && v in 0.22f..0.80f &&
                         r in 100..215 && g in 68..162 && b in 14..105 &&
                         (r - g) in 15..62 && (g - b) in 28..82) {
                    onionGoldCount++
                }
                count++
            }
        }

        val totalCount = maxOf(1, count)
        val redPct     = tomatoRedCount.toFloat()    / totalCount
        val breakerPct = breakerOrangeCount.toFloat() / totalCount
        val onionVPct  = onionVioletCount.toFloat()   / totalCount
        val onionGPct  = onionGoldCount.toFloat()     / totalCount
        val tomatoTotal = redPct + breakerPct
        val onionTotal  = onionVPct + onionGPct

        val cleanRaw = rawName.lowercase().trim()

        // ── Decision Engine ──

        // RULE 0 (highest priority): Onion neck/stalk check — lowered threshold to 0.25
        // Sandy background + stalk ochre in top 22% should easily exceed 0.25
        if (neckOchrePct > 0.25f) {
            return Pair("Onion", 0.92f)
        }

        // Rule 1: If the model said "onion" but pixels strongly show genuine tomato → correct it
        if ((cleanRaw == "onion" || cleanRaw == "pyaz") && tomatoTotal > 0.12f && tomatoTotal > onionTotal * 3f && neckOchrePct < 0.15f) {
            return Pair("Tomato", 0.93f)
        }

        // Rule 2: If the model said "onion" (any other case) → keep Onion (model knows shape)
        if (cleanRaw == "onion" || cleanRaw == "pyaz") {
            return Pair("Onion", 0.93f)
        }

        // Rule 3a: If model said "tomato" but dark maroon onion pixels are significant → correct to Onion
        if (cleanRaw == "tomato" && onionTotal > 0.06f && onionTotal >= tomatoTotal * 0.50f) {
            return Pair("Onion", 0.90f)
        }

        // Rule 3b: Also override tomato if neck check is meaningful
        if (cleanRaw == "tomato" && neckOchrePct > 0.15f) {
            return Pair("Onion", 0.90f)
        }

        // Rule 3c: Strong onion body evidence — even if tomato slightly wins pixels
        if (cleanRaw == "tomato" && onionVPct > 0.10f) {
            return Pair("Onion", 0.90f)
        }

        // Rule 4: Confirm Tomato from pixel evidence — tomato must clearly beat onion AND neck is clean
        if (tomatoTotal > 0.12f && tomatoTotal > onionTotal * 2.0f && neckOchrePct < 0.15f) {
            return if (cleanRaw == "apple") Pair("Apple", 0.94f) else Pair("Tomato", 0.94f)
        }

        // Rule 5: Confirm Onion from pixel evidence
        if (onionVPct > 0.08f || onionVioletCount >= 6) {
            return Pair("Onion", 0.94f)
        }
        if (onionGPct > 0.15f && onionGoldCount >= 8 && tomatoTotal < 0.05f) {
            return Pair("Onion", 0.92f)
        }

        // Rule 6: Pass through whatever the model said, formatted properly
        val formatted = rawName.replaceFirstChar { if (it.isLowerCase()) it.titlecase() else it.toString() }
        return Pair(formatted, 0.88f)
    }

    private fun nms(items: List<DetectedItem>, threshold: Float): List<DetectedItem> {
        val kept = mutableListOf<DetectedItem>()
        for (item in items.sortedByDescending { it.confidence }) {
            // Suppress duplicate boxes for the same item
            if (kept.none { overlap(it, item) > threshold }) kept += item
        }
        return kept
    }

    private fun overlap(a: DetectedItem, b: DetectedItem): Float {
        val left = maxOf(a.x1, b.x1); val top = maxOf(a.y1, b.y1)
        val right = minOf(a.x2, b.x2); val bottom = minOf(a.y2, b.y2)
        val intersection = maxOf(0f, right - left) * maxOf(0f, bottom - top)
        val union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - intersection
        return if (union > 0f) intersection / union else 0f
    }

    override fun close() = session.close()
}
