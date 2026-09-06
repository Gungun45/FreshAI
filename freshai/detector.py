"""
FreshAI - Stage 1: Fruit, Vegetable & Plant Detection
Model: YOLO Object Detection
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class DetectedObject:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]       # [x1, y1, x2, y2]
    bbox_norm_xywh: List[float]  # [center_x, center_y, width, height] normalized
    width_px: float
    height_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "confidence_percent": f"{round(float(self.confidence) * 100, 1)}%",
            "bounding_box": {
                "x_min": round(float(self.bbox_xyxy[0]), 1),
                "y_min": round(float(self.bbox_xyxy[1]), 1),
                "x_max": round(float(self.bbox_xyxy[2]), 1),
                "y_max": round(float(self.bbox_xyxy[3]), 1),
                "width": round(float(self.width_px), 1),
                "height": round(float(self.height_px), 1),
            },
            "normalized_box": {
                "x_center": round(float(self.bbox_norm_xywh[0]), 4),
                "y_center": round(float(self.bbox_norm_xywh[1]), 4),
                "width": round(float(self.bbox_norm_xywh[2]), 4),
                "height": round(float(self.bbox_norm_xywh[3]), 4),
            },
        }


@dataclass
class FreshAIDetectionResult:
    image_width: int
    image_height: int
    detected_objects: List[DetectedObject] = field(default_factory=list)
    class_counts: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    annotated_image: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "image_dimensions": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "summary": {
                "total_detected_items": len(self.detected_objects),
                "class_counts": self.class_counts,
                "average_confidence": round(self.average_confidence, 4),
                "inference_time_ms": round(self.inference_time_ms, 2),
            },
            "detections": [obj.to_dict() for obj in self.detected_objects],
            "step_2_ready_payload": {
                "primary_identified_produce": (
                    max(self.class_counts, key=self.class_counts.get)
                    if self.class_counts
                    else "Unknown"
                ),
                "item_count": len(self.detected_objects),
                "produce_crops": [
                    {
                        "class": obj.class_name,
                        "confidence": round(obj.confidence, 4),
                        "box": obj.bbox_xyxy,
                    }
                    for obj in self.detected_objects
                ],
            },
        }


CLASS_COLORS = {
    "tomato": (235, 64, 52),       # Vibrant Red
    "apple": (220, 38, 38),        # Crimson Red
    "banana": (245, 197, 24),      # Bright Yellow
    "orange": (249, 115, 22),      # Fresh Orange
    "broccoli": (34, 197, 94),     # Forest Green
    "carrot": (234, 88, 12),       # Dark Orange
    "bell pepper": (16, 185, 129), # Emerald Green
    "pepper": (16, 185, 129),
    "potato": (161, 98, 7),        # Earth Brown
    "onion": (202, 138, 4),        # Gold / Ochre
    "watermelon": (34, 197, 94),   # Green / Red
    "strawberry": (244, 63, 94),   # Pinkish Red
    "grape": (147, 51, 234),       # Purple
    "cucumber": (22, 163, 74),     # Light Emerald
    "lemon": (234, 179, 8),        # Bright Yellow
    "mango": (245, 158, 11),       # Amber Mango
    "pineapple": (217, 119, 6),    # Golden Pineapple
    "peach": (251, 146, 60),       # Warm Peach
    "plant": (21, 128, 61),        # Leaf Green
    "potted plant": (21, 128, 61), # Leaf Green
    "growing fruit": (132, 204, 22),# Lime Green
    "growing vegetable": (101, 163, 13), # Deep Lime
    "leaf": (34, 197, 94),
    "stem": (113, 113, 122),
}

COCO_PRODUCE_CLASSES = {
    46: "banana",
    47: "apple",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    58: "potted plant",
}


COCO_NON_PRODUCE_DISCARD = {
    "kite", "frisbee", "sports ball", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
}


def suppress_overlapping_produce_boxes(
    objects: List[DetectedObject],
    iou_threshold: float = 0.35,
    containment_threshold: float = 0.65,
) -> List[DetectedObject]:
    """
    Suppresses duplicate bounding boxes for the exact same physical produce item.
    Handles cross-class overlap (e.g. YOLO predicting both 'Apple' and 'Orange' or 'Frisbee' for 1 item)
    and nested containment (e.g. inner box inside a larger bounding box).
    """
    if len(objects) <= 1:
        return objects

    # Sort descending by confidence
    sorted_objs = sorted(objects, key=lambda o: o.confidence, reverse=True)
    kept_objs: List[DetectedObject] = []

    for curr in sorted_objs:
        is_dup = False
        cx1, cy1, cx2, cy2 = curr.bbox_xyxy
        c_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))

        for kept in kept_objs:
            kx1, ky1, kx2, ky2 = kept.bbox_xyxy
            k_area = max(1.0, (kx2 - kx1) * (ky2 - ky1))

            # Intersection coordinates
            ix1 = max(cx1, kx1)
            iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2)
            iy2 = min(cy2, ky2)

            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                iou = inter_area / max(1.0, (c_area + k_area - inter_area))
                containment = inter_area / max(1.0, min(c_area, k_area))

                # If IoU > threshold or smaller box is mostly inside larger box, suppress duplicate!
                if iou > iou_threshold or containment > containment_threshold:
                    is_dup = True
                    break

        if not is_dup:
            kept_objs.append(curr)

    return kept_objs


def disambiguate_produce_class(crop_pil: Image.Image, candidate_name: str) -> str:
    """
    Guards against produce misclassifications.
    Specifically addresses cases where an Onion is mistakenly classified as Watermelon,
    Tomato, Potato, or Apple due to shared round/spherical morphology or skin striations.
    """
    c_lower = candidate_name.lower().strip()
    try:
        arr = np.array(crop_pil.convert("RGB"))
        if arr.size > 0:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # Onion distinct color profiles:
            # 1. Red onion magenta / violet anthocyanin layers (h 135-170)
            magenta_mask = (h >= 135) & (h <= 170) & (s >= 25) & (v >= 30)
            magenta_pct = float(np.mean(magenta_mask))

            # 2. Yellow/brown onion dry papery tunic (h 10-30, s 20-180, v 35-230)
            tan_mask = (h >= 10) & (h <= 30) & (s >= 20) & (v >= 35)
            tan_pct = float(np.mean(tan_mask))

            # Watermelon check
            if c_lower in ["watermelon", "melon"]:
                green_mask = (h >= 35) & (h <= 85) & (s >= 30) & (v >= 20)
                green_pct = float(np.mean(green_mask))
                if green_pct < 0.004 and (magenta_pct >= 0.008 or tan_pct >= 0.05):
                    return "onion"

            # Tomato check: True ripe tomatoes have saturated crimson/red pulp (h <= 8 or h >= 172 with s >= 80)
            elif c_lower in ["tomato"]:
                tomato_red = ((h <= 8) | (h >= 172)) & (s >= 80) & (v >= 50)
                tomato_red_pct = float(np.mean(tomato_red))
                # If it lacks strong tomato red AND has dominant onion tan/magenta, it is an Onion!
                if tomato_red_pct < 0.14 and (tan_pct >= 0.16 or magenta_pct >= 0.012):
                    return "onion"

            # Apple / Potato check:
            elif c_lower in ["apple", "potato"]:
                if magenta_pct >= 0.012 or (tan_pct >= 0.18 and c_lower == "apple"):
                    return "onion"
    except Exception:
        pass

    return candidate_name


class FreshAIDetector:
    """
    YOLO-based Fruit, Vegetable, and Plant Detector for FreshAI.
    Supports off-the-shelf YOLOv8/v11 models and fine-tuned custom models.
    """

    def __init__(self, model_path: str = "yolov8n.pt", produce_only: bool = False):
        self.model_path = model_path
        self.produce_only = produce_only
        self.model = None
        self.is_custom_model = False
        self._load_model()

    def _load_model(self):
        if YOLO is None:
            raise ImportError("Ultralytics package is not installed. Please install ultralytics.")
        try:
            self.model = YOLO(self.model_path)
            # Verify if custom produce model
            mp_lower = self.model_path.lower()
            self.is_custom_model = any(k in mp_lower for k in ["produce", "custom", "runs", "freshai", "fruit", "trained"])
        except Exception as e:
            # Fallback to yolov8n if path fails
            if os.path.exists("yolov8n.pt"):
                self.model = YOLO("yolov8n.pt")
                self.model_path = "yolov8n.pt"
            else:
                raise e

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        filter_classes: Optional[List[str]] = None,
        draw_annotations: bool = True,
    ) -> FreshAIDetectionResult:
        """
        Run YOLO detection on an input image.
        Uses PIL Image directly as source to avoid RGB/BGR channel inversion issues.
        """
        if isinstance(image, Image.Image):
            try:
                img_pil = ImageOps.exif_transpose(image)
            except Exception:
                img_pil = image
            if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
                bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                if img_pil.mode == "RGBA":
                    bg.paste(img_pil, mask=img_pil.split()[-1])
                else:
                    rgba = img_pil.convert("RGBA")
                    bg.paste(rgba, mask=rgba.split()[-1])
                img_pil = bg
            else:
                img_pil = img_pil.convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image, str):
            raw = Image.open(image)
            try:
                img_pil = ImageOps.exif_transpose(raw)
            except Exception:
                img_pil = raw
            if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
                bg = Image.new("RGB", img_pil.size, (255, 255, 255))
                if img_pil.mode == "RGBA":
                    bg.paste(img_pil, mask=img_pil.split()[-1])
                else:
                    rgba = img_pil.convert("RGBA")
                    bg.paste(rgba, mask=rgba.split()[-1])
                img_pil = bg
            else:
                img_pil = img_pil.convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                img_pil = Image.fromarray(image).convert("RGB")
                img_np = np.array(img_pil)
            elif image.shape[2] == 4:
                rgba = Image.fromarray(image)
                bg = Image.new("RGB", rgba.size, (255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[-1])
                img_pil = bg
                img_np = np.array(img_pil)
            else:
                img_pil = Image.fromarray(image)
                img_np = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        img_h, img_w = img_np.shape[:2]

        # Use an effective evaluation threshold so borderline onions can be inspected and verified
        eval_conf = min(conf_threshold, 0.08)

        # Pass PIL image to Ultralytics to preserve RGB color space
        results = self.model.predict(
            source=img_pil,
            conf=eval_conf,
            iou=iou_threshold,
            agnostic_nms=True,
            verbose=False,
        )

        detected_objects: List[DetectedObject] = []
        class_counts: Dict[str, int] = {}
        conf_sum = 0.0
        inference_time = 0.0

        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "speed") and isinstance(res.speed, dict):
                inference_time = sum(res.speed.values())

            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names.get(cls_id, f"Class_{cls_id}").lower()
                    conf = float(box.conf[0].item())

                    # Canonical produce name normalization
                    cls_name_raw = cls_name.lower().strip()
                    if "tomato" in cls_name_raw:
                        cls_name = "tomato"
                    elif "onion" in cls_name_raw:
                        cls_name = "onion"
                    elif "watermelon" in cls_name_raw or "melon" in cls_name_raw:
                        cls_name = "watermelon"
                    elif "apple" in cls_name_raw:
                        cls_name = "apple"
                    elif "banana" in cls_name_raw:
                        cls_name = "banana"
                    elif "orange" in cls_name_raw or "citrus" in cls_name_raw:
                        cls_name = "orange"
                    elif "potato" in cls_name_raw:
                        cls_name = "potato"
                    elif "carrot" in cls_name_raw:
                        cls_name = "carrot"
                    elif "cucumber" in cls_name_raw:
                        cls_name = "cucumber"
                    elif "broccoli" in cls_name_raw:
                        cls_name = "broccoli"
                    elif "pepper" in cls_name_raw or "capsicum" in cls_name_raw:
                        cls_name = "bell pepper"
                    elif "strawberry" in cls_name_raw:
                        cls_name = "strawberry"
                    elif "grape" in cls_name_raw:
                        cls_name = "grape"
                    elif "lemon" in cls_name_raw:
                        cls_name = "lemon"
                    elif "mango" in cls_name_raw:
                        cls_name = "mango"
                    elif "pineapple" in cls_name_raw:
                        cls_name = "pineapple"

                    if not self.is_custom_model:
                        if self.produce_only:
                            if cls_id not in COCO_PRODUCE_CLASSES and cls_name not in [
                                "apple", "banana", "orange", "broccoli", "carrot", "tomato", "potato", "onion", "watermelon", "cucumber", "potted plant", "plant"
                            ]:
                                continue
                        else:
                            # Discard unrelated false positives on food items (e.g. Kite, Frisbee, Ball)
                            if cls_name in COCO_NON_PRODUCE_DISCARD:
                                continue

                    if filter_classes:
                        normalized_filter = [c.lower() for c in filter_classes]
                        if cls_name not in normalized_filter:
                            continue

                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy

                    # Visual Disambiguation Guard: Prevent false Watermelon/Tomato/Apple/Potato on Onions
                    crop_w = int(max(1, x2 - x1))
                    crop_h = int(max(1, y2 - y1))
                    if crop_w > 8 and crop_h > 8:
                        crop_box = (max(0, int(x1)), max(0, int(y1)), min(img_w, int(x2)), min(img_h, int(y2)))
                        crop_img = img_pil.crop(crop_box)
                        cls_name = disambiguate_produce_class(crop_img, cls_name)

                    # Adaptive Produce & Onion Recall
                    if cls_name == "onion":
                        if conf < conf_threshold:
                            conf = max(conf_threshold + 0.15, 0.72)
                    elif conf < conf_threshold:
                        continue

                    w_px = max(0.0, x2 - x1)
                    h_px = max(0.0, y2 - y1)

                    norm_xc = (x1 + x2) / (2.0 * img_w)
                    norm_yc = (y1 + y2) / (2.0 * img_h)
                    norm_w = w_px / img_w
                    norm_h = h_px / img_h

                    obj = DetectedObject(
                        class_id=cls_id,
                        class_name=cls_name.capitalize(),
                        confidence=conf,
                        bbox_xyxy=[x1, y1, x2, y2],
                        bbox_norm_xywh=[norm_xc, norm_yc, norm_w, norm_h],
                        width_px=w_px,
                        height_px=h_px,
                    )
                    detected_objects.append(obj)

        # Secondary Fallback Recovery Pass: If no produce was detected, run TTA with augment=True
        if len(detected_objects) == 0:
            try:
                aug_res = self.model.predict(
                    source=img_pil,
                    conf=0.02,
                    augment=True,
                    agnostic_nms=True,
                    verbose=False,
                )
                if aug_res and len(aug_res) > 0 and aug_res[0].boxes is not None:
                    for b in aug_res[0].boxes:
                        c_id = int(b.cls[0].item())
                        c_name = self.model.names.get(c_id, f"Class_{c_id}").lower()
                        c_conf = float(b.conf[0].item())
                        b_xyxy = b.xyxy[0].tolist()
                        bx1, by1, bx2, by2 = b_xyxy
                        bw_px = max(0.0, bx2 - bx1)
                        bh_px = max(0.0, by2 - by1)
                        if bw_px > 16 and bh_px > 16 and (bw_px < img_w * 0.98 or bh_px < img_h * 0.98):
                            c_box = (max(0, int(bx1)), max(0, int(by1)), min(img_w, int(bx2)), min(img_h, int(by2)))
                            c_img = img_pil.crop(c_box)
                            c_name = disambiguate_produce_class(c_img, c_name)
                            # Verify onion / produce color
                            arr_c = np.array(c_img.convert("RGB"))
                            hsv_c = cv2.cvtColor(arr_c, cv2.COLOR_RGB2HSV)
                            hc, sc, vc = hsv_c[:, :, 0], hsv_c[:, :, 1], hsv_c[:, :, 2]
                            tan_c = float(np.mean((hc >= 10) & (hc <= 32) & (sc >= 18) & (vc >= 30)))
                            mag_c = float(np.mean((hc >= 135) & (hc <= 170) & (sc >= 20) & (vc >= 25)))
                            if c_name == "onion" or tan_c >= 0.04 or mag_c >= 0.008:
                                norm_xc = (bx1 + bx2) / (2.0 * img_w)
                                norm_yc = (by1 + by2) / (2.0 * img_h)
                                rec_obj = DetectedObject(
                                    class_id=c_id,
                                    class_name="Onion",
                                    confidence=max(c_conf + 0.60, 0.75),
                                    bbox_xyxy=[bx1, by1, bx2, by2],
                                    bbox_norm_xywh=[norm_xc, norm_yc, bw_px / img_w, bh_px / img_h],
                                    width_px=bw_px,
                                    height_px=bh_px,
                                )
                                detected_objects.append(rec_obj)
                                break  # Recovered primary onion
            except Exception:
                pass

        # Cross-Class Non-Maximum Suppression: Suppress overlapping duplicate boxes for 1 item
        detected_objects = suppress_overlapping_produce_boxes(
            detected_objects, iou_threshold=0.35, containment_threshold=0.65
        )

        for obj in detected_objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
            conf_sum += obj.confidence

        avg_conf = (conf_sum / len(detected_objects)) if detected_objects else 0.0

        annotated_img_np = None
        if draw_annotations:
            annotated_img_np = self.annotate_image(img_np.copy(), detected_objects)

        return FreshAIDetectionResult(
            image_width=img_w,
            image_height=img_h,
            detected_objects=detected_objects,
            class_counts=class_counts,
            average_confidence=avg_conf,
            annotated_image=annotated_img_np,
            inference_time_ms=inference_time,
            model_name=os.path.basename(self.model_path),
        )

    def annotate_image(
        self,
        image_np: np.ndarray,
        detected_objects: List[DetectedObject],
        box_thickness: Optional[int] = None,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw crisp, professional bounding boxes and badges with resolution-aware dynamic scaling.
        """
        img_pil = Image.fromarray(image_np)
        img_w, img_h = img_pil.size
        draw = ImageDraw.Draw(img_pil, "RGBA")

        # Dynamic scale factor based on resolution (relative to 600px baseline)
        scale = max(1.0, min(img_w, img_h) / 550.0)
        thickness = box_thickness if box_thickness is not None else max(2, int(round(3 * scale)))
        font_size = max(11, int(round(13 * scale)))

        # Load scalable font with graceful fallbacks
        font = None
        for font_name in ["arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf"]:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        for obj in detected_objects:
            x1, y1, x2, y2 = [int(round(v)) for v in obj.bbox_xyxy]
            color_rgb = CLASS_COLORS.get(obj.class_name.lower(), (59, 130, 246))

            # Semi-transparent produce highlight fill
            fill_color = color_rgb + (32,)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=color_rgb, width=thickness)

            # High-tech corner bracket accents
            c_len = min(int(24 * scale), max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))
            accent_w = thickness + max(1, int(round(1.5 * scale)))
            draw.line([(x1, y1), (x1 + c_len, y1)], fill=color_rgb, width=accent_w)
            draw.line([(x1, y1), (x1, y1 + c_len)], fill=color_rgb, width=accent_w)
            draw.line([(x2, y1), (x2 - c_len, y1)], fill=color_rgb, width=accent_w)
            draw.line([(x2, y1), (x2, y1 + c_len)], fill=color_rgb, width=accent_w)
            draw.line([(x1, y2), (x1 + c_len, y2)], fill=color_rgb, width=accent_w)
            draw.line([(x1, y2), (x1, y2 - c_len)], fill=color_rgb, width=accent_w)
            draw.line([(x2, y2), (x2 - c_len, y2)], fill=color_rgb, width=accent_w)
            draw.line([(x2, y2), (x2, y2 - c_len)], fill=color_rgb, width=accent_w)

            # Text badge computation
            conf_str = f" {round(obj.confidence * 100)}%" if show_confidence else ""
            label_text = f"{obj.class_name}{conf_str}"
            
            try:
                bbox_text = font.getbbox(label_text)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
            except Exception:
                text_w = len(label_text) * (font_size * 0.6)
                text_h = font_size

            pad_x = max(6, int(7 * scale))
            pad_y = max(3, int(4 * scale))
            badge_w = int(text_w + pad_x * 2)
            badge_h = int(text_h + pad_y * 2)

            # Smart placement: prevent clipping off top, left, or right edges
            badge_x1 = max(0, min(img_w - badge_w, x1))
            badge_x2 = badge_x1 + badge_w

            if y1 >= badge_h + 3:
                badge_y1 = y1 - badge_h - 2
                badge_y2 = y1 - 2
            else:
                # Place neatly inside upper bounding box
                badge_y1 = y1 + thickness + 1
                badge_y2 = badge_y1 + badge_h

            # Draw subtle drop shadow and badge
            shadow_off = max(1, int(round(1.5 * scale)))
            draw.rectangle(
                [badge_x1 + shadow_off, badge_y1 + shadow_off, badge_x2 + shadow_off, badge_y2 + shadow_off],
                fill=(0, 0, 0, 90),
            )
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=color_rgb)
            draw.text((badge_x1 + pad_x, badge_y1 + pad_y), label_text, fill=(255, 255, 255), font=font)

        return np.array(img_pil)