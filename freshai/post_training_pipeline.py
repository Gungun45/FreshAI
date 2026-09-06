"""
FreshAI - Post-Training Pipeline
Waits for training to complete, then:
  1. Validates the best model (mAP, precision, recall)
  2. Exports to TFLite (float32 + INT8 quantized) for Android
  3. Copies model into the Android app assets folder
  4. Generates a training summary report

Usage:
  python -m freshai.post_training_pipeline            # waits for training to finish
  python -m freshai.post_training_pipeline --skip-wait  # use existing best.pt now
"""

import os
import sys
import time
import shutil
import json
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent.resolve()
def find_latest_best_weights():
    candidates = list((PROJECT_ROOT / "runs" / "detect").rglob("best.pt"))
    if not candidates:
        return PROJECT_ROOT / "runs" / "detect" / "runs" / "freshai_detect" / "produce_detector" / "weights" / "best.pt"
    # Sort by modification time, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

BEST_PT        = find_latest_best_weights()
ANDROID_ASSETS = PROJECT_ROOT / "app" / "src" / "main" / "assets"
REPORT_PATH    = PROJECT_ROOT / "runs" / "freshai_detect" / "training_report.json"
DATA_YAML      = PROJECT_ROOT / "datasets" / "freshai_produce_focus" / "data_fixed.yaml"
LOG_PATH       = PROJECT_ROOT / "runs" / "freshai_detect" / "gpu_training.log"

TFLITE_FLOAT   = "freshai_produce_detector.tflite"
TFLITE_INT8    = "freshai_produce_detector_int8.tflite"
ONNX_NAME      = "freshai_produce_detector.onnx"

CLASSES = ["Onion", "Tomato", "Watermelon"]


# ─────────────────────────────────────────────────────────────────────────────
def wait_for_training(check_interval=60, timeout_hours=12):
    """Poll until best.pt exists and the training log signals completion."""
    print(f"\n{'='*60}")
    print("  FreshAI Post-Training Pipeline — Waiting for Training...")
    print(f"  Watching: {BEST_PT}")
    print(f"{'='*60}\n")

    start = time.time()
    timeout_sec = timeout_hours * 3600

    while True:
        elapsed = time.time() - start
        if elapsed > timeout_sec:
            print("Timeout reached — training may have stalled.")
            return False

        if BEST_PT.exists():
            if LOG_PATH.exists():
                log_text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
                if any(k in log_text for k in ["Results saved", "Training complete", "Speed:"]):
                    print(f"Training complete!  best.pt -> {BEST_PT}\n")
                    return True
            else:
                print(f"best.pt detected — proceeding.\n")
                return True

        elapsed_min = int(elapsed / 60)
        print(f"  [{elapsed_min} min] Training still running... next check in {check_interval}s")
        time.sleep(check_interval)


# ─────────────────────────────────────────────────────────────────────────────
def validate_model(model_path):
    print(f"\n{'─'*60}")
    print("  Step 1: Validating Model")
    print(f"{'─'*60}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    metrics = model.val(data=str(DATA_YAML), verbose=True)

    box = getattr(metrics, "box", None)
    result = {
        "map50":     round(float(getattr(box, "map50", 0.0)), 4) if box else 0.0,
        "map50_95":  round(float(getattr(box, "map",   0.0)), 4) if box else 0.0,
        "precision": round(float(getattr(box, "mp",    0.0)), 4) if box else 0.0,
        "recall":    round(float(getattr(box, "mr",    0.0)), 4) if box else 0.0,
    }

    print(f"\n  Validation Results:")
    print(f"     mAP@50:     {result['map50']:.4f}  ({result['map50']*100:.1f}%)")
    print(f"     mAP@50-95:  {result['map50_95']:.4f}  ({result['map50_95']*100:.1f}%)")
    print(f"     Precision:  {result['precision']:.4f}")
    print(f"     Recall:     {result['recall']:.4f}")

    rating = (
        "EXCELLENT" if result["map50"] >= 0.80 else
        "GOOD"       if result["map50"] >= 0.65 else
        "ACCEPTABLE" if result["map50"] >= 0.50 else
        "NEEDS_IMPROVEMENT"
    )
    print(f"  Quality Rating: {rating}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
def export_model(model_path, export_dir):
    print(f"\n{'─'*60}")
    print("  Step 2: Exporting Model for Android")
    print(f"{'─'*60}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    exported = {}

    # ONNX
    print("\n  Exporting -> ONNX ...")
    try:
        out = model.export(format="onnx", imgsz=640, simplify=True)
        dest = Path(export_dir) / ONNX_NAME
        shutil.copy2(str(out), str(dest))
        exported["onnx"] = str(dest)
        print(f"     ONNX saved: {dest}")
    except Exception as e:
        print(f"     ONNX export failed: {e}")

    # TFLite float32
    print("\n  Exporting -> TFLite float32 ...")
    try:
        out = model.export(format="tflite", imgsz=640)
        dest = Path(export_dir) / TFLITE_FLOAT
        shutil.copy2(str(out), str(dest))
        exported["tflite_float32"] = str(dest)
        print(f"     TFLite float32: {dest}  ({dest.stat().st_size//1024} KB)")
    except Exception as e:
        print(f"     TFLite float32 failed: {e}")

    # TFLite INT8
    print("\n  Exporting -> TFLite INT8 quantized ...")
    try:
        out = model.export(format="tflite", imgsz=640, int8=True)
        dest = Path(export_dir) / TFLITE_INT8
        shutil.copy2(str(out), str(dest))
        exported["tflite_int8"] = str(dest)
        print(f"     TFLite INT8:    {dest}  ({dest.stat().st_size//1024} KB)")
    except Exception as e:
        print(f"     TFLite INT8 failed: {e}")

    return exported


# ─────────────────────────────────────────────────────────────────────────────
def deploy_to_android(exported):
    print(f"\n{'─'*60}")
    print("  Step 3: Deploying to Android App Assets")
    print(f"{'─'*60}")

    ANDROID_ASSETS.mkdir(parents=True, exist_ok=True)
    deployed = {}

    # Prefer INT8 (smaller/faster on mobile), fallback to float32
    src = exported.get("tflite_int8") or exported.get("tflite_float32")
    if src and Path(src).exists():
        dest = ANDROID_ASSETS / TFLITE_FLOAT   # app always loads this filename
        if dest.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = ANDROID_ASSETS / f"freshai_produce_detector_backup_{stamp}.tflite"
            shutil.copy2(str(dest), str(backup))
            print(f"  Backed up old model -> {backup.name}")
        shutil.copy2(src, str(dest))
        deployed["android_tflite"] = str(dest)
        print(f"  Deployed: {dest.name}  ({dest.stat().st_size//1024} KB)")
    else:
        print("  No TFLite model available to deploy.")

    # Write labels file
    labels_dest = ANDROID_ASSETS / "freshai_labels.txt"
    labels_dest.write_text("\n".join(CLASSES), encoding="utf-8")
    deployed["labels"] = str(labels_dest)
    print(f"  Labels:   {labels_dest.name}")

    return deployed


# ─────────────────────────────────────────────────────────────────────────────
def save_report(metrics, exported, deployed):
    report = {
        "training_completed_at": datetime.now().isoformat(),
        "model": "YOLOv8n — FreshAI Produce Detector",
        "classes": CLASSES,
        "dataset": {"train": 1476, "val": 339, "epochs": 30, "imgsz": 640},
        "metrics": metrics,
        "exported": exported,
        "deployed": deployed,
        "quality": (
            "EXCELLENT"        if metrics.get("map50", 0) >= 0.80 else
            "GOOD"             if metrics.get("map50", 0) >= 0.65 else
            "ACCEPTABLE"       if metrics.get("map50", 0) >= 0.50 else
            "NEEDS_IMPROVEMENT"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {REPORT_PATH}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(skip_wait=False):
    print(f"\n{'='*60}")
    print(f"  FreshAI Post-Training Pipeline  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if not skip_wait:
        if not wait_for_training():
            print("Training did not complete successfully. Exiting.")
            sys.exit(1)
    else:
        if not BEST_PT.exists():
            print(f"ERROR: best.pt not found at {BEST_PT}")
            sys.exit(1)
        print(f"Using existing best.pt: {BEST_PT}")

    export_dir = PROJECT_ROOT / "runs" / "freshai_detect" / "exports"
    metrics  = validate_model(BEST_PT)
    exported = export_model(BEST_PT, export_dir)
    deployed = deploy_to_android(exported)
    report   = save_report(metrics, exported, deployed)

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"  Classes   : {', '.join(CLASSES)}")
    print(f"  mAP@50    : {metrics.get('map50',0)*100:.1f}%")
    print(f"  mAP@50-95 : {metrics.get('map50_95',0)*100:.1f}%")
    print(f"  Precision : {metrics.get('precision',0)*100:.1f}%")
    print(f"  Recall    : {metrics.get('recall',0)*100:.1f}%")
    print(f"  Quality   : {report['quality']}")
    print(f"  Android   : {deployed.get('android_tflite','N/A')}")
    print(f"  Report    : {REPORT_PATH}")
    print(f"{'='*60}\n")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="FreshAI Post-Training Pipeline")
    ap.add_argument("--skip-wait", action="store_true",
                    help="Use existing best.pt immediately without waiting")
    args = ap.parse_args()
    run_pipeline(skip_wait=args.skip_wait)
