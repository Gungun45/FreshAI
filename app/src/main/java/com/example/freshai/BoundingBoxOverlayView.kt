package com.example.freshai

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

data class DetectedItem(
    val className: String,
    val confidence: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

class BoundingBoxOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val detections = mutableListOf<DetectedItem>()
    private var imageWidth = 1
    private var imageHeight = 1

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
    }

    fun setDetections(items: List<DetectedItem>, imgW: Int, imgH: Int) {
        detections.clear()
        detections.addAll(items)
        imageWidth = if (imgW > 0) imgW else 1
        imageHeight = if (imgH > 0) imgH else 1
        invalidate()
    }
    fun clear() {
        detections.clear()
        invalidate()
    }

    private val hudPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.argb(120, 255, 255, 255)
    }

    private val cornerBracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        strokeCap = Paint.Cap.ROUND
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (detections.isEmpty() || width == 0 || height == 0) return

        // Compute scaling & centering aspect ratio fit
        val viewAspect = width.toFloat() / height.toFloat()
        val imgAspect = imageWidth.toFloat() / imageHeight.toFloat()

        val scale: Float
        val dx: Float
        val dy: Float

        if (viewAspect > imgAspect) {
            scale = height.toFloat() / imageHeight.toFloat()
            dx = (width - imageWidth * scale) / 2f
            dy = 0f
        } else {
            scale = width.toFloat() / imageWidth.toFloat()
            dx = 0f
            dy = (height - imageHeight * scale) / 2f
        }

        for (item in detections) {
            val color = getColorForClass(item.className)
            boxPaint.color = color
            fillPaint.color = Color.argb(35, Color.red(color), Color.green(color), Color.blue(color))
            textBgPaint.color = color
            cornerBracketPaint.color = color

            val left = item.x1 * scale + dx
            val top = item.y1 * scale + dy
            val right = item.x2 * scale + dx
            val bottom = item.y2 * scale + dy

            val rect = RectF(left, top, right, bottom)
            canvas.drawRoundRect(rect, 8f, 8f, fillPaint)
            canvas.drawRoundRect(rect, 8f, 8f, boxPaint)

            // Precision HUD Corner Brackets
            val cornerLen = minOf((right - left) * 0.20f, (bottom - top) * 0.20f, 28f)
            // Top-Left
            canvas.drawLine(left - 2f, top, left + cornerLen, top, cornerBracketPaint)
            canvas.drawLine(left, top - 2f, left, top + cornerLen, cornerBracketPaint)
            // Top-Right
            canvas.drawLine(right + 2f, top, right - cornerLen, top, cornerBracketPaint)
            canvas.drawLine(right, top - 2f, right, top + cornerLen, cornerBracketPaint)
            // Bottom-Left
            canvas.drawLine(left - 2f, bottom, left + cornerLen, bottom, cornerBracketPaint)
            canvas.drawLine(left, bottom + 2f, left, bottom - cornerLen, cornerBracketPaint)
            // Bottom-Right
            canvas.drawLine(right + 2f, bottom, right - cornerLen, bottom, cornerBracketPaint)
            canvas.drawLine(right, bottom + 2f, right, bottom - cornerLen, cornerBracketPaint)

            // Center Sensor Crosshair
            val cx = (left + right) / 2f
            val cy = (top + bottom) / 2f
            val crossSize = 10f
            canvas.drawLine(cx - crossSize, cy, cx + crossSize, cy, hudPaint)
            canvas.drawLine(cx, cy - crossSize, cx, cy + crossSize, hudPaint)
            canvas.drawCircle(cx, cy, 14f, hudPaint)

            // Draw label badge (draws inside box if near top edge to avoid colliding with sensor HUD)
            val label = "🎯 ${item.className} ${(item.confidence * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            val badgeHeight = 44f
            val badgeTop = if (top < 65f) (top + 6f) else (top - badgeHeight).coerceAtLeast(0f)
            val badgeBottom = badgeTop + badgeHeight
            val badgeRight = (left + textWidth + 24f).coerceAtMost(width.toFloat())

            val badgeRect = RectF(left, badgeTop, badgeRight, badgeBottom)
            canvas.drawRoundRect(badgeRect, 6f, 6f, textBgPaint)
            canvas.drawText(label, left + 12f, badgeTop + 32f, textPaint)
        }
    }

    private fun getColorForClass(name: String): Int {
        return when (name.lowercase()) {
            "onion" -> Color.parseColor("#A855F7")
            "tomato" -> Color.parseColor("#EF4444")
            "apple" -> Color.parseColor("#DC2626")
            "banana" -> Color.parseColor("#EAB308")
            "orange" -> Color.parseColor("#F97316")
            "potato" -> Color.parseColor("#D97706")
            "watermelon" -> Color.parseColor("#059669")
            "strawberry" -> Color.parseColor("#E11D48")
            "lemon" -> Color.parseColor("#FACC15")
            "mango" -> Color.parseColor("#FB923C")
            "eggplant" -> Color.parseColor("#7C3AED")
            "broccoli", "plant", "plant_leaf", "leaf" -> Color.parseColor("#22C55E")
            "carrot" -> Color.parseColor("#EA580C")
            "bell pepper", "bell_pepper", "pepper" -> Color.parseColor("#10B981")
            "growing fruit", "growing_fruit" -> Color.parseColor("#84CC16")
            else -> Color.parseColor("#3B82F6")
        }
    }
}