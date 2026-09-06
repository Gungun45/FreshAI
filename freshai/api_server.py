"""
FreshAI - Multimodal & Post-Harvest Advisory API Server
- Stage 1: Produce Detection (YOLO)
- Stage 2: Freshness & Ripeness (ConvNeXt-Tiny)
- Stage 3: Defect Detection & Segmentation (YOLO)
- Stage 4: Color & Texture Feature Extraction (OpenCV)
- Stage 5: Structured Prediction Layer (XGBoost & LightGBM)
- Stage 6: LLM + RAG Recommendations Engine
Port: 8088
"""

import io
import json
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
from PIL import Image

from freshai.detector import FreshAIDetector, DetectedObject
from freshai.freshness_detector import ConvNeXtFreshnessDetector
from freshai.defect_detector import FreshAIDefectDetector
from freshai.feature_extractor import ProduceFeatureExtractor
from freshai.fusion import assemble_feature_vector
from freshai.fusion_predictor import FreshAIFusionPredictor
from freshai.structured_features import EnvironmentalData, TimelineStorageData, StructuredProduceInput
from freshai.structured_predictor import FreshAIStructuredPredictor
from freshai.rag_engine import FreshAIRAGEngine


def find_latest_model():
    candidates = list(Path("runs/detect").rglob("best.pt"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])
    return None


BASE_MODEL = "yolov8n.pt"
CUSTOM_MODEL = find_latest_model()
MODEL_PATH = CUSTOM_MODEL or BASE_MODEL

print("=== FreshAI Multimodal & RAG API Server ===")
print("Detection Model:", MODEL_PATH)

detector = FreshAIDetector(model_path=MODEL_PATH, produce_only=False)
freshness_detector = ConvNeXtFreshnessDetector()
defect_detector = FreshAIDefectDetector()
feature_extractor = ProduceFeatureExtractor()
fusion_predictor = FreshAIFusionPredictor()
structured_predictor = FreshAIStructuredPredictor()
rag_engine = FreshAIRAGEngine()

print("All 6 AI stages initialized successfully. Server listening on port 8088.")


class FreshAIRequestHandler(BaseHTTPRequestHandler):

    def _send_json(self, status_code: int, data: dict):
        resp_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status" or self.path == "/":
            self._send_json(200, {
                "status": "online",
                "service": "FreshAI Multimodal Post-Harvest Server",
                "stages": {
                    "stage_1_detection": "YOLO Active",
                    "stage_2_freshness": f"ConvNeXt Active ({freshness_detector.mode})",
                    "stage_3_defect": f"YOLO Defect Segmentation Active ({defect_detector.model_mode})",
                    "stage_4_features": "OpenCV Color & GLCM Texture Active",
                    "stage_5_structured": f"XGBoost & LightGBM Active (models_loaded={structured_predictor.models_loaded})",
                    "stage_6_rag": f"LLM RAG Active (gemini_enabled={rag_engine.api_key is not None})",
                },
                "model_path": MODEL_PATH,
            })
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"status": "error", "message": "Empty request body"})
            return

        body = self.rfile.read(content_length)

        if self.path.startswith("/api/recommend"):
            # Direct text RAG query
            try:
                req_json = json.loads(body.decode("utf-8"))
                p_class = req_json.get("produce_class", "Tomato")
                q_score = float(req_json.get("quality_score", 84.0))
                p_age = float(req_json.get("physiological_age_days", 6.0))
                s_life = float(req_json.get("remaining_shelf_life_days", 3.0))
                d_area = float(req_json.get("defect_area_pct", 5.0))
                risk = req_json.get("spoilage_risk", "Low")
                temp = float(req_json.get("temperature_c", 22.0))

                rec = rag_engine.generate_recommendations(
                    produce_class=p_class,
                    quality_score=q_score,
                    physiological_age_days=p_age,
                    remaining_shelf_life_days=s_life,
                    defect_area_pct=d_area,
                    spoilage_risk=risk,
                    storage_temp_c=temp,
                )
                self._send_json(200, {"status": "success", "recommendations": rec.to_dict()})
            except Exception as e:
                self._send_json(500, {"status": "error", "message": str(e)})
            return

        # Main Vision & Multimodal Pipeline: /api/detect
        try:
            pil_image = Image.open(io.BytesIO(body)).convert("RGB")
            img_w, img_h = pil_image.size
            img_np = np.array(pil_image)

            # Query params or headers for environment
            temp_c = float(self.headers.get("X-Temperature-C", 22.0))
            humid_pct = float(self.headers.get("X-Humidity-Pct", 60.0))
            days_elapsed = float(self.headers.get("X-Days-Elapsed", 2.0))
            storage_cond = self.headers.get("X-Storage-Condition", "Pantry / Room Temperature")

            # Stage 1: YOLO Detection
            det_res = detector.predict(
                image=pil_image,
                conf_threshold=0.35,
                iou_threshold=0.45,
                draw_annotations=False,
            )

            target_objects = det_res.detected_objects

            # Whole-produce fallback if no isolated bounding box was found
            if not target_objects:
                target_objects = [
                    DetectedObject(
                        class_id=0,
                        class_name="Produce",
                        confidence=0.85,
                        bbox_xyxy=[0.0, 0.0, float(img_w), float(img_h)],
                        bbox_norm_xywh=[0.5, 0.5, 1.0, 1.0],
                        width_px=float(img_w),
                        height_px=float(img_h),
                    )
                ]

            detected_items = []

            for obj in target_objects:
                x1 = max(0, int(obj.bbox_xyxy[0]))
                y1 = max(0, int(obj.bbox_xyxy[1]))
                x2 = min(img_w, int(obj.bbox_xyxy[2]))
                y2 = min(img_h, int(obj.bbox_xyxy[3]))

                if x2 > x1 and y2 > y1:
                    crop_np = img_np[y1:y2, x1:x2]
                    crop_pil = Image.fromarray(crop_np)
                else:
                    crop_np = img_np
                    crop_pil = pil_image

                # Stage 2: Freshness & Ripeness (ConvNeXt)
                fr = freshness_detector.predict(crop_pil, produce_class=obj.class_name)

                # Stage 3: Defect Detection & Segmentation (YOLO)
                dr = defect_detector.predict(crop_np, produce_class=obj.class_name)

                # Stage 4: Color & Texture + Fusion
                fe = feature_extractor.extract(crop_np)
                fv = assemble_feature_vector(fr, dr, fe, produce_class=obj.class_name)
                fa = fusion_predictor.predict(fv)

                # Stage 5: Structured Prediction Layer (XGBoost & LightGBM)
                struct_input = StructuredProduceInput(
                    feature_vector=fv,
                    environmental=EnvironmentalData(temperature_c=temp_c, humidity_pct=humid_pct),
                    timeline=TimelineStorageData(days_since_purchase=days_elapsed, storage_condition=storage_cond),
                    produce_class=obj.class_name,
                )
                s_res = structured_predictor.predict(struct_input)

                # Stage 6: LLM + RAG Recommendations Engine
                rag_rec = rag_engine.generate_recommendations(
                    produce_class=obj.class_name,
                    quality_score=s_res.quality_score,
                    physiological_age_days=s_res.physiological_age_days,
                    remaining_shelf_life_days=s_res.remaining_shelf_life_days,
                    defect_area_pct=dr.total_defect_area_pct,
                    spoilage_risk=s_res.spoilage_risk,
                    storage_temp_c=temp_c,
                    days_since_purchase=days_elapsed,
                )

                # Biological Arrhenius Shelf Life Kinetics
                arr_res = fa.arrhenius_shelf_life(
                    produce_class=obj.class_name,
                    temp_c=temp_c,
                    defect_area_pct=dr.total_defect_area_pct,
                )

                item_data = {
                    # Stage 1: Detection
                    "class_name": obj.class_name,
                    "confidence": round(float(obj.confidence), 4),
                    "bbox_xyxy": [round(float(c), 1) for c in obj.bbox_xyxy],
                    # Stage 2: ConvNeXt
                    "freshness_stage": fr.freshness_stage,
                    "freshness_prob": round(float(fr.freshness_prob), 4),
                    "ripeness_stage": fr.ripeness_stage,
                    "ripeness_prob": round(float(fr.ripeness_prob), 4),
                    "freshness_emoji": fr.freshness_emoji,
                    "ripeness_emoji": fr.ripeness_emoji,
                    # Stage 3: YOLO Defect Segmentation
                    "defect_count": dr.defect_count,
                    "total_defect_area_pct": round(float(dr.total_defect_area_pct), 2),
                    "defect_severity_score": round(float(dr.defect_severity_score), 2),
                    # Stage 4: Color/Texture Fusion
                    "multimodal_freshness_score": fa.freshness_score,
                    "quality_grade": fa.quality_grade,
                    # Biological Arrhenius Kinetics
                    "arrhenius_shelf_life": arr_res.to_dict(),
                    # Stage 5: Structured Predictions (XGBoost & LightGBM)
                    "structured_predictions": {
                        "quality_score": round(float(s_res.quality_score), 1),
                        "quality_score_str": s_res.quality_score_str,
                        "physiological_age": s_res.physiological_age_range,
                        "physiological_age_days": round(float(s_res.physiological_age_days), 1),
                        "post_harvest_age": s_res.post_harvest_age_range,
                        "remaining_shelf_life": s_res.remaining_shelf_life_range,
                        "remaining_shelf_life_days": round(float(s_res.remaining_shelf_life_days), 1),
                        "spoilage_risk": s_res.spoilage_risk,
                        "spoilage_risk_color": s_res.spoilage_risk_color,
                        "time_to_harvest": s_res.time_to_harvest_range,
                        "what_if_shelf_life": s_res.what_if_shelf_life_days,
                    },
                    # Stage 6: LLM + RAG Recommendations
                    "rag_recommendations": rag_rec.to_dict(),
                }
                detected_items.append(item_data)

            response_data = {
                "status": "success",
                "model": MODEL_PATH,
                "image_dimensions": {"width": img_w, "height": img_h},
                "total_detected": len(detected_items),
                "primary_produce": detected_items[0]["class_name"] if detected_items else "None",
                "environmental": {
                    "temperature_c": temp_c,
                    "humidity_pct": humid_pct,
                    "days_elapsed": days_elapsed,
                    "storage_condition": storage_cond,
                },
                "detections": detected_items,
            }

            self._send_json(200, response_data)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._send_json(500, {"status": "error", "message": str(e)})


def run_server(port: int = 8088):
    server_address = ("", port)
    httpd = HTTPServer(server_address, FreshAIRequestHandler)
    print(f"FreshAI Server running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
