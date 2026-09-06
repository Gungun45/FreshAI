package com.example.freshai

import android.content.Context
import android.graphics.Bitmap
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

        // Match labels to output tensor shape:
        // freshai_labels.txt: 0=Onion, 1=Tomato, 2=Watermelon (Matches 7-output YOLO tensor [1, 4+3, 8400])
        val outInfo = session.outputInfo.values.firstOrNull()?.info as? ai.onnxruntime.TensorInfo
        val outShape = outInfo?.shape
        val numClasses = if (outShape != null && outShape.size >= 2) (outShape[1] - 4).toInt() else 3

        val labelFile = if (numClasses <= 3) "freshai_labels.txt" else "freshai_classes.txt"
        names = context.assets.open(labelFile).bufferedReader().useLines { lines ->
            lines.map(String::trim).filter { it.isNotEmpty() }.toList().toTypedArray()
        }
    }

    fun detect(bitmap: Bitmap, confidence: Float = 0.18f, iou: Float = 0.35f): List<DetectedItem> {
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
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114))
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
                val values = output[0].value as Array<Array<FloatArray>> // [1, 7, 8400]
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

                    candidates += DetectedItem(
                        names[classId],
                        best,
                        unpadX1.coerceIn(0f, srcW),
                        unpadY1.coerceIn(0f, srcH),
                        unpadX2.coerceIn(0f, srcW),
                        unpadY2.coerceIn(0f, srcH)
                    )
                }
                return nms(candidates, iou)
            }
        }
    }

    private fun nms(items: List<DetectedItem>, threshold: Float): List<DetectedItem> {
        val kept = mutableListOf<DetectedItem>()
        for (item in items.sortedByDescending { it.confidence }) {
            // Suppress boxes even when the model assigned different labels.
            // A low IoU threshold is intentional here: the training model
            // often places several broad, differently-labelled boxes over
            // one close-up fruit. Keep the strongest box for that item.
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
