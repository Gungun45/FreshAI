"""
FreshAI - Stage 1: Produce Classification (Fruits-360 & Produce Datasets)
Model: YOLO-cls (Classification)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass
class ClassificationItem:
    class_id: int
    class_name: str
    confidence: float
    confidence_percent: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "confidence_percent": self.confidence_percent,
        }


@dataclass
class FreshAIClassificationResult:
    predicted_class: str
    confidence: float
    confidence_percent: str
    top5_predictions: List[ClassificationItem] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "predicted_produce": self.predicted_class,
            "top_confidence": round(self.confidence, 4),
            "confidence_percent": self.confidence_percent,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "top5": [item.to_dict() for item in self.top5_predictions],
            "step_2_ready_payload": {
                "primary_identified_produce": self.predicted_class,
                "confidence": round(self.confidence, 4),
            }
        }


class FreshAIClassifier:
    """
    Produce Classifier for Fruits-360 and custom produce classification datasets.
    """
    def __init__(self, model_path: str = "yolov8n-cls.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if YOLO is None:
            raise ImportError("Ultralytics package is not installed.")
        self.model = YOLO(self.model_path)

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> FreshAIClassificationResult:
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        elif isinstance(image, str):
            img_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        results = self.model.predict(source=img_np, verbose=False)
        top5_items: List[ClassificationItem] = []
        pred_class = "Unknown"
        top_conf = 0.0
        inference_time = 0.0

        if results and len(results) > 0:
            res = results[0]
            if hasattr(res, "speed") and isinstance(res.speed, dict):
                inference_time = sum(res.speed.values())

            probs = res.probs
            if probs is not None:
                top1_id = int(probs.top1)
                top_conf = float(probs.top1conf.item())
                pred_class = res.names.get(top1_id, f"Class_{top1_id}")

                # Get top 5 predictions
                top5_indices = probs.top5
                for idx in top5_indices:
                    c_id = int(idx)
                    c_name = res.names.get(c_id, f"Class_{c_id}")
                    c_conf = float(probs.data[c_id].item())
                    top5_items.append(
                        ClassificationItem(
                            class_id=c_id,
                            class_name=c_name,
                            confidence=c_conf,
                            confidence_percent=f"{round(c_conf * 100, 2)}%",
                        )
                    )

        return FreshAIClassificationResult(
            predicted_class=pred_class,
            confidence=top_conf,
            confidence_percent=f"{round(top_conf * 100, 2)}%",
            top5_predictions=top5_items,
            inference_time_ms=inference_time,
            model_name=os.path.basename(self.model_path),
        )
