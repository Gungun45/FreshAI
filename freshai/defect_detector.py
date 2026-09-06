"""
FreshAI - Stage 3: Defect Detection & Segmentation
Models: YOLO Detection + YOLO Segmentation + Computer Vision Segmentation Fallback

Objective:
Detect visible physical defects that affect quality and shelf life:
• Bruises
• Black spots
• Cracks
• Mold
• Rot
• Fungal spots
• Wrinkles
• Discoloration

Why Two YOLO Approaches?
- YOLO Detection: What defect is present and where it is (bounding box).
- YOLO Segmentation: Exactly which pixels/area are affected (segmentation mask).
Output metric: Defect Area % = (Defect Mask Area / Produce Area) * 100
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


DEFECT_CLASSES = [
    "Bruise",
    "Black Spot",
    "Crack",
    "Mold",
    "Rot",
    "Fungal Spot",
    "Wrinkle",
    "Discoloration",
]

# Color palette for defect overlays (RGB tuples and Hex)
DEFECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Bruise": (234, 179, 8),          # Amber / yellow-brown
    "Black Spot": (55, 65, 81),        # Dark slate
    "Crack": (239, 68, 68),            # Bright Red
    "Mold": (168, 85, 247),           # Purple / Mycelium
    "Rot": (127, 29, 29),             # Dark brownish-red
    "Fungal Spot": (217, 119, 6),      # Ochre
    "Wrinkle": (59, 130, 246),         # Blue
    "Discoloration": (245, 158, 11),   # Orange
}

# Severity multiplier for shelf life impact (0.0 - 1.0)
DEFECT_SEVERITY_WEIGHTS: Dict[str, float] = {
    "Rot": 1.0,           # Critical spoilage, spreads quickly
    "Mold": 1.0,          # High health hazard, rapid decay
    "Crack": 0.7,         # Breaches skin barrier, invites bacteria
    "Black Spot": 0.5,    # Progressive necrosis
    "Fungal Spot": 0.6,   # Active pathogen
    "Bruise": 0.4,        # Internal enzymatic browning
    "Wrinkle": 0.3,       # Dehydration / water loss
    "Discoloration": 0.2, # Surface aesthetic / minor pigment degradation
}


@dataclass
class DefectItem:
    defect_type: str
    confidence: float
    bbox_xyxy: List[int]                     # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None        # 2D boolean or uint8 mask in crop coordinates
    area_pixels: int = 0
    area_percent: float = 0.0                # Percentage of produce surface
    severity_weight: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_type": self.defect_type,
            "confidence": round(self.confidence, 3),
            "bbox_xyxy": [int(v) for v in self.bbox_xyxy],
            "area_pixels": int(self.area_pixels),
            "area_percent": round(self.area_percent, 2),
            "severity_weight": self.severity_weight,
        }


@dataclass
class DefectResult:
    defects: List[DefectItem] = field(default_factory=list)
    total_defect_area_pct: float = 0.0       # Combined non-overlapping defect area %
    total_defect_pixels: int = 0
    produce_area_pixels: int = 0
    defect_count: int = 0
    defect_type_breakdown: Dict[str, float] = field(default_factory=dict) # type -> area %
    composite_mask: Optional[np.ndarray] = None  # (H, W) boolean mask of all defects
    annotated_crop: Optional[np.ndarray] = None  # RGB numpy image with colored masks & boxes
    defect_severity_score: float = 0.0       # 0.0 (pristine) to 1.0 (heavily degraded)
    model_mode: str = "cv_segmenter"         # "yolo_seg" | "yolo_det" | "cv_segmenter"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_count": self.defect_count,
            "total_defect_area_percent": round(self.total_defect_area_pct, 2),
            "total_defect_pixels": int(self.total_defect_pixels),
            "produce_area_pixels": int(self.produce_area_pixels),
            "defect_severity_score": round(self.defect_severity_score, 3),
            "model_mode": self.model_mode,
            "defect_type_breakdown": {k: round(v, 2) for k, v in self.defect_type_breakdown.items()},
            "defects": [d.to_dict() for d in self.defects],
        }


class FreshAIDefectDetector:
    """
    Produce Defect Detection & Segmentation Engine.
    Executes YOLO Detection (locating defects) + YOLO Segmentation (pixel-accurate boundaries),
    with an OpenCV computer-vision segmentation fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        mask_threshold: float = 0.5,
    ):
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold
        self.model = None
        self.model_mode = "cv_segmenter"
        self.model_path = model_path

        if model_path and os.path.exists(model_path) and ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                # Check if model has segmentation task
                is_seg = hasattr(self.model, "task") and self.model.task == "segment"
                self.model_mode = "yolo_seg" if is_seg else "yolo_det"
            except Exception as e:
                print(f"[DefectDetector] Could not load YOLO weights at {model_path}: {e}")
                self.model = None
                self.model_mode = "cv_segmenter"

    def predict(
        self,
        crop: Union[np.ndarray, Image.Image, str],
        produce_class: str = "",
        produce_mask: Optional[np.ndarray] = None,
    ) -> DefectResult:
        """
        Analyze a cropped produce image for physical defects and segment affected pixels.

        Args:
            crop: Cropped RGB produce image (numpy array, PIL Image, or file path)
            produce_class: Name of produce (e.g. 'Apple', 'Tomato') for tuning thresholds
            produce_mask: Optional binary mask of the produce foreground within the crop
        """
        # Standardize to RGB numpy array
        if isinstance(crop, str):
            img_pil = Image.open(crop).convert("RGB")
            crop_np = np.array(img_pil)
        elif isinstance(crop, Image.Image):
            crop_np = np.array(crop.convert("RGB"))
        elif isinstance(crop, np.ndarray):
            crop_np = crop.copy()
            if crop_np.ndim == 2:
                crop_np = cv2.cvtColor(crop_np, cv2.COLOR_GRAY2RGB)
            elif crop_np.shape[2] == 4:
                crop_np = cv2.cvtColor(crop_np, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported crop input type: {type(crop)}")

        h, w = crop_np.shape[:2]
        if h < 5 or w < 5:
            return DefectResult()

        # If a trained YOLO model is loaded, use it
        if self.model is not None:
            return self._predict_yolo(crop_np, produce_class, produce_mask)
        else:
            # High-precision OpenCV defect segmentation engine
            return self._predict_cv_segmenter(crop_np, produce_class, produce_mask)

    def _predict_yolo(
        self,
        crop_np: np.ndarray,
        produce_class: str,
        produce_mask: Optional[np.ndarray],
    ) -> DefectResult:
        h, w = crop_np.shape[:2]
        results = self.model.predict(
            source=crop_np,
            conf=self.conf_threshold,
            verbose=False,
        )

        defects: List[DefectItem] = []
        composite_mask = np.zeros((h, w), dtype=bool)

        # Estimate produce area
        if produce_mask is not None:
            produce_area = int(np.sum(produce_mask > 0))
        else:
            produce_area = self._estimate_produce_foreground_area(crop_np)
        produce_area = max(1, produce_area)

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            masks = getattr(res, "masks", None)
            names = self.model.names if hasattr(self.model, "names") else {}

            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(box.cls[0].item()) if hasattr(box, "cls") else 0
                conf = float(box.conf[0].item()) if hasattr(box, "conf") else 0.5
                defect_type = names.get(cls_id, DEFECT_CLASSES[cls_id % len(DEFECT_CLASSES)])

                mask = None
                if masks is not None and idx < len(masks.data):
                    raw_mask = masks.data[idx].cpu().numpy()
                    if raw_mask.shape != (h, w):
                        raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = (raw_mask > self.mask_threshold).astype(np.uint8)
                    defect_pixels = int(np.sum(mask > 0))
                    composite_mask |= (mask > 0)
                else:
                    # Bounding box fallback
                    x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w, xyxy[2]), min(h, xyxy[3])
                    mask = np.zeros((h, w), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 1
                    defect_pixels = (x2 - x1) * (y2 - y1)
                    composite_mask[y1:y2, x1:x2] = True

                area_pct = (defect_pixels / produce_area) * 100.0
                area_pct = min(100.0, area_pct)
                weight = DEFECT_SEVERITY_WEIGHTS.get(defect_type, 0.5)

                defects.append(DefectItem(
                    defect_type=defect_type,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                    mask=mask,
                    area_pixels=defect_pixels,
                    area_percent=area_pct,
                    severity_weight=weight,
                ))

        total_defect_pixels = int(np.sum(composite_mask))
        total_defect_pct = min(100.0, (total_defect_pixels / produce_area) * 100.0)

        breakdown: Dict[str, float] = {}
        for d in defects:
            breakdown[d.defect_type] = breakdown.get(d.defect_type, 0.0) + d.area_percent

        severity = self._compute_severity(defects, total_defect_pct)
        annotated = self._render_annotation(crop_np, defects, composite_mask, total_defect_pct)

        return DefectResult(
            defects=defects,
            total_defect_area_pct=total_defect_pct,
            total_defect_pixels=total_defect_pixels,
            produce_area_pixels=produce_area,
            defect_count=len(defects),
            defect_type_breakdown=breakdown,
            composite_mask=composite_mask,
            annotated_crop=annotated,
            defect_severity_score=severity,
            model_mode=self.model_mode,
        )

    def _predict_cv_segmenter(
        self,
        crop_np: np.ndarray,
        produce_class: str,
        produce_mask: Optional[np.ndarray],
    ) -> DefectResult:
        """
        OpenCV Computer Vision Defect Segmentation Engine.
        Identifies necrotic spots, mold fungal patches, cracks, rot lesions,
        and discoloration contours on the produce surface.
        """
        h, w = crop_np.shape[:2]

        # 1. Segment foreground produce from background
        if produce_mask is None:
            foreground_mask = self._segment_produce_foreground(crop_np)
        else:
            foreground_mask = (produce_mask > 0).astype(np.uint8)

        produce_area = max(1, int(np.sum(foreground_mask > 0)))

        # Convert color spaces
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(crop_np, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)

        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Compute healthy skin baseline statistics on foreground
        fg_indices = np.where(foreground_mask > 0)
        if len(fg_indices[0]) == 0:
            return DefectResult(produce_area_pixels=h * w, model_mode="cv_segmenter")

        med_L = float(np.median(L[fg_indices]))
        med_A = float(np.median(A[fg_indices]))
        med_B = float(np.median(B[fg_indices]))
        med_V = float(np.median(V[fg_indices]))
        med_S = float(np.median(S[fg_indices]))

        # Candidate defect masks
        defects: List[DefectItem] = []
        composite_mask = np.zeros((h, w), dtype=bool)

        # ── (A) Mold & Fungal Mycelium Detection ──
        # Whitish/grayish/pale cyan powdery surface patches with low saturation or high lightness relative to body
        mold_cand = (
            (foreground_mask > 0)
            & (S < 45)
            & (V > 120)
            & (L > med_L + 25)
        ).astype(np.uint8)
        # Morphological clean
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mold_clean = cv2.morphologyEx(mold_cand, cv2.MORPH_OPEN, kernel_s)
        self._extract_defect_contours(mold_clean, "Mold", 0.78, crop_np, produce_area, defects, composite_mask, min_area=15)

        # ── (B) Rot & Necrotic Lesions Detection ──
        # Dark sunken brown/black decomposed areas (significant decrease in Lightness and Value)
        rot_cand = (
            (foreground_mask > 0)
            & (V < np.clip(med_V * 0.45, 20, 90))
            & (L < np.clip(med_L * 0.50, 25, 95))
        ).astype(np.uint8)
        rot_clean = cv2.morphologyEx(rot_cand, cv2.MORPH_OPEN, kernel_s)
        self._extract_defect_contours(rot_clean, "Rot", 0.85, crop_np, produce_area, defects, composite_mask, min_area=25)

        # ── (C) Black Spots & Fungal Spots Detection ──
        # Concentrated small dark melanin points
        spot_cand = (
            (foreground_mask > 0)
            & (~composite_mask)
            & (V < 70)
            & (S > 20)
        ).astype(np.uint8)
        spot_clean = cv2.morphologyEx(spot_cand, cv2.MORPH_OPEN, kernel_s)
        self._extract_defect_contours(spot_clean, "Black Spot", 0.80, crop_np, produce_area, defects, composite_mask, min_area=8, max_area=500)

        # ── (D) Cracks & Fissures Detection ──
        # Thin elongated edges with high Laplacian gradient
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.uint8(np.absolute(lap))
        edges = cv2.Canny(gray, 60, 160)
        crack_cand = (
            (foreground_mask > 0)
            & (~composite_mask)
            & (edges > 0)
            & (lap_abs > 35)
        ).astype(np.uint8)
        # Dilate slightly to form crack mask
        crack_cand = cv2.dilate(crack_cand, kernel_s, iterations=1)
        self._extract_defect_contours(crack_cand, "Crack", 0.72, crop_np, produce_area, defects, composite_mask, min_area=20, aspect_ratio_min=2.2)

        # ── (E) Bruises & Discoloration ──
        # Color difference Delta E in LAB space
        delta_E = np.sqrt(
            (L.astype(np.float32) - med_L) ** 2 +
            (A.astype(np.float32) - med_A) ** 2 +
            (B.astype(np.float32) - med_B) ** 2
        )
        bruise_cand = (
            (foreground_mask > 0)
            & (~composite_mask)
            & (delta_E > 28.0)
            & (V < med_V)
        ).astype(np.uint8)
        bruise_clean = cv2.morphologyEx(bruise_cand, cv2.MORPH_OPEN, kernel_s)
        self._extract_defect_contours(bruise_clean, "Bruise", 0.65, crop_np, produce_area, defects, composite_mask, min_area=35)

        # Discoloration: remaining chromatic deviations
        discolor_cand = (
            (foreground_mask > 0)
            & (~composite_mask)
            & (delta_E > 38.0)
        ).astype(np.uint8)
        discolor_clean = cv2.morphologyEx(discolor_cand, cv2.MORPH_OPEN, kernel_s)
        self._extract_defect_contours(discolor_clean, "Discoloration", 0.60, crop_np, produce_area, defects, composite_mask, min_area=40)

        # Calculate totals
        total_defect_pixels = int(np.sum(composite_mask))
        total_defect_pct = min(100.0, (total_defect_pixels / produce_area) * 100.0)

        breakdown: Dict[str, float] = {}
        for d in defects:
            breakdown[d.defect_type] = breakdown.get(d.defect_type, 0.0) + d.area_percent

        severity = self._compute_severity(defects, total_defect_pct)
        annotated = self._render_annotation(crop_np, defects, composite_mask, total_defect_pct)

        return DefectResult(
            defects=defects,
            total_defect_area_pct=total_defect_pct,
            total_defect_pixels=total_defect_pixels,
            produce_area_pixels=produce_area,
            defect_count=len(defects),
            defect_type_breakdown=breakdown,
            composite_mask=composite_mask,
            annotated_crop=annotated,
            defect_severity_score=severity,
            model_mode="cv_segmenter",
        )

    def _extract_defect_contours(
        self,
        binary_mask: np.ndarray,
        defect_type: str,
        base_confidence: float,
        crop_np: np.ndarray,
        produce_area: int,
        defects_list: List[DefectItem],
        composite_mask: np.ndarray,
        min_area: int = 15,
        max_area: Optional[int] = None,
        aspect_ratio_min: float = 0.0,
    ):
        h, w = binary_mask.shape[:2]
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
            if aspect_ratio < aspect_ratio_min:
                continue

            # Create individual defect mask
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, thickness=-1)

            defect_pixels = int(np.sum(cnt_mask > 0))
            if defect_pixels == 0:
                continue

            area_pct = (defect_pixels / produce_area) * 100.0
            composite_mask[cnt_mask > 0] = True

            weight = DEFECT_SEVERITY_WEIGHTS.get(defect_type, 0.5)
            # Adjust confidence slightly by size
            conf = min(0.96, base_confidence + min(0.15, area / (produce_area * 0.1)))

            defects_list.append(DefectItem(
                defect_type=defect_type,
                confidence=conf,
                bbox_xyxy=[x, y, x + bw, y + bh],
                mask=cnt_mask,
                area_pixels=defect_pixels,
                area_percent=area_pct,
                severity_weight=weight,
            ))

    def _segment_produce_foreground(self, crop_np: np.ndarray) -> np.ndarray:
        """
        Segment the produce object from plain or table background using Otsu + Chroma.
        Returns a binary mask (uint8, 0 or 1).
        """
        h, w = crop_np.shape[:2]
        gray = cv2.cvtColor(crop_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(crop_np, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]

        # Produce typically has noticeable saturation or contrast compared to plain white/gray background
        # Filter near-white background
        white_bg = (gray > 238) & (sat < 35)
        # Filter near-black background
        black_bg = gray < 18

        fg = (~(white_bg | black_bg)).astype(np.uint8)

        # Morphological fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        # Keep largest connected component as produce
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            fg = (labels == largest_label).astype(np.uint8)

        # Fallback: if foreground covers < 10% of image, treat entire crop as produce
        if np.sum(fg) < (h * w * 0.10):
            fg = np.ones((h, w), dtype=np.uint8)

        return fg

    def _estimate_produce_foreground_area(self, crop_np: np.ndarray) -> int:
        fg = self._segment_produce_foreground(crop_np)
        return int(np.sum(fg > 0))

    def _compute_severity(self, defects: List[DefectItem], total_area_pct: float) -> float:
        """Compute aggregate defect severity score (0.0 to 1.0)."""
        if not defects or total_area_pct <= 0.01:
            return 0.0

        # Weighted severity by area
        weighted_sum = sum(d.area_percent * d.severity_weight for d in defects)
        base_severity = weighted_sum / 100.0  # normalized

        # Non-linear boost for critical rot / mold
        has_critical = any(d.defect_type in ["Rot", "Mold"] and d.area_percent > 1.5 for d in defects)
        if has_critical:
            base_severity = max(base_severity, 0.45 + (total_area_pct / 100.0) * 0.55)

        return float(np.clip(base_severity, 0.0, 1.0))

    def _render_annotation(
        self,
        crop_np: np.ndarray,
        defects: List[DefectItem],
        composite_mask: np.ndarray,
        total_defect_pct: float,
    ) -> np.ndarray:
        """
        Render semi-transparent segmentation masks, colored bounding boxes,
        and defect labels onto the produce crop.
        """
        annotated = crop_np.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent segmentation masks
        mask_overlay = np.zeros_like(annotated, dtype=np.uint8)
        for d in defects:
            if d.mask is not None:
                color = DEFECT_COLORS.get(d.defect_type, (255, 0, 0))
                mask_overlay[d.mask > 0] = color

        # Alpha blend masks with 45% opacity
        mask_indices = np.where(composite_mask > 0)
        if len(mask_indices[0]) > 0:
            alpha = 0.45
            annotated[mask_indices] = cv2.addWeighted(
                annotated[mask_indices], 1.0 - alpha,
                mask_overlay[mask_indices], alpha, 0.0
            )

        # Draw contours and bounding boxes
        for d in defects:
            color = DEFECT_COLORS.get(d.defect_type, (255, 0, 0))
            b_color = (int(color[0]), int(color[1]), int(color[2]))

            # Contour outline
            if d.mask is not None:
                contours, _ = cv2.findContours(d.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, b_color, thickness=2)

            # Bounding box
            x1, y1, x2, y2 = d.bbox_xyxy
            cv2.rectangle(annotated, (x1, y1), (x2, y2), b_color, thickness=2)

            # Text label
            label = f"{d.defect_type} {round(d.area_percent, 1)}%"
            font_scale = max(0.4, min(0.6, w / 400.0))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

            # Label badge
            by1 = max(0, y1 - th - 6)
            by2 = y1
            cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, by2), b_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Bottom banner with overall Defect Area %
        banner_text = f"Defect Area: {round(total_defect_pct, 1)}% ({len(defects)} defects)"
        cv2.rectangle(annotated, (0, h - 26), (w, h), (15, 23, 42), -1)
        status_color = (34, 197, 94) if total_defect_pct < 3.0 else ((234, 179, 8) if total_defect_pct < 10.0 else (239, 68, 68))
        cv2.putText(annotated, banner_text, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1, cv2.LINE_AA)

        return annotated
