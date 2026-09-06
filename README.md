# 🌿 FreshAI — AI-Powered Produce Lifecycle & Freshness Intelligence Platform

FreshAI is an intelligent platform designed to analyze fruits, vegetables, and crops across their entire journey — from plant growth and harvest estimation to post-harvest freshness, quality scoring, remaining shelf life, and AI-assisted storage recommendations.

---

## 🎯 Step 1: Fruit, Vegetable & Plant Detection (YOLO)

The first stage identifies objects present in an image, determines their exact coordinates (bounding boxes), identifies their class names (Tomato, Apple, Banana, Growing Fruit/Vegetable, etc.), and evaluates detection confidence scores.

### Detection Pipeline Flow
```
Input Image (Camera / Upload / Crop)
  │
  ▼
YOLO Detection Engine (Ultralytics YOLOv8 / YOLOv11 / Custom Model)
  │
  ▼
Object Identification & Localization
  ├── Class: Tomato / Apple / Plant / ...
  ├── Bounding Box: [x_min, y_min, x_max, y_max] (px & normalized)
  └── Confidence: 95.8%
  │
  ▼
FreshAI Step 2 Integration Bridge (JSON Payload)
```

---

## 🚀 Quick Start (Running the Prototype Interface)

### 1. Launch Interactive Web App
Double-click `run_freshai.bat` or run:
```bash
.venv\Scripts\streamlit run app.py
```
This opens the FreshAI interactive testing dashboard in your browser (`http://localhost:8501`), where you can:
- 📸 **Upload your own produce/plant images** (or use the built-in Sample Showcase / Webcam).
- 🎛️ **Adjust confidence and IoU thresholds** in real-time.
- 📊 **Inspect bounding boxes, count metrics, and item breakdown tables**.
- 🔗 **View the structured Step 2 JSON payload**.

---

## 🛠️ Custom Training Pipeline (Fine-Tuning YOLO)

### Train on Synthetic Demo Dataset (Instant Verification):
```bash
.venv\Scripts\python freshai/train_yolo.py --demo --epochs 10 --batch 8
```

### Train on Custom Kaggle / Roboflow Dataset:
1. Organize your dataset in standard YOLO format:
   ```
   datasets/my_produce_data/
     images/train/, images/val/
     labels/train/, labels/val/
     data.yaml
   ```
2. Run training:
   ```bash
   .venv\Scripts\python freshai/train_yolo.py --data datasets/my_produce_data/data.yaml --epochs 30 --model yolov8n.pt
   ```
3. The trained weights (`best.pt`) will automatically appear in the Streamlit UI dropdown for immediate testing!

---

## 📁 Project Structure

```
FreshAI/
├── freshai/
│   ├── __init__.py           # Package exports
│   ├── detector.py           # FreshAIDetector class & YOLO inference wrapper
│   ├── dataset_utils.py      # Dataset generators, YOLO data.yaml builders
│   └── train_yolo.py         # Modular fine-tuning pipeline & metric evaluation
├── samples/                  # Preloaded sample produce test images
├── tests/
│   └── test_detector.py      # Automated unit tests for detection & data prep
├── app.py                    # Streamlit interactive testing web application
├── run_freshai.bat           # One-click Windows startup script
├── requirements.txt          # Dependencies (ultralytics, streamlit, torch, etc.)
└── pyproject.toml            # Project packaging configuration
```

---

## 🗺️ FreshAI Master Roadmap
- [x] **Step 1:** Fruit, Vegetable & Plant Detection (YOLO) + Interactive Testing Interface
- [ ] **Step 2:** Growth Stage, Maturity & Freshness / Quality Scoring Model
- [ ] **Step 3:** Environmental Sync (Weather API: Temperature & Humidity + Purchase Date)
- [ ] **Step 4:** Generative AI & RAG for Nutrition & Shelf-Life Extension Recommendations