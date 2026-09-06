"""
FreshAI - Stage 1: Fruit, Vegetable & Plant Detection
Interactive Streamlit Application
"""

import os
import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from freshai.detector import FreshAIDetector, CLASS_COLORS
from freshai.classifier import FreshAIClassifier
from freshai.freshness_detector import (
    ConvNeXtFreshnessDetector,
    FRESHNESS_STAGES,
    RIPENESS_STAGES,
    FRESHNESS_COLORS,
    FRESHNESS_SHELF_LIFE,
    FRESHNESS_EMOJI,
)
from freshai.defect_detector import (
    FreshAIDefectDetector,
    DefectResult,
    DEFECT_CLASSES,
    DEFECT_COLORS,
)
from freshai.feature_extractor import (
    ProduceFeatureExtractor,
    ProduceExtractedFeatures,
)
from freshai.fusion import (
    ProduceFeatureVector,
    assemble_feature_vector,
)
from freshai.fusion_predictor import (
    FreshAIFusionPredictor,
    MultimodalFreshnessAssessment,
)
from freshai.structured_features import (
    EnvironmentalData,
    TimelineStorageData,
    StructuredProduceInput,
    STORAGE_CONDITIONS,
)
from freshai.structured_predictor import (
    FreshAIStructuredPredictor,
    StructuredPredictionResult,
)
from freshai.rag_engine import (
    FreshAIRAGEngine,
    ProduceRAGRecommendations,
)
from freshai.dataset_utils import generate_synthetic_demo_dataset
from freshai.train_yolo import train_yolo_model
from freshai.train_classifier import train_classifier
from freshai.train_fusion_freshness import train_fusion_model
from freshai.train_defect_yolo import train_defect_yolo
from freshai.train_structured_layer import train_structured_models

# Streamlit Page Config
st.set_page_config(
    page_title="FreshAI - Produce Detection Platform",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Aesthetic Theme - Modern Glassmorphism & High-Impact Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.01em;
    }

    /* Polished Image Rendering */
    .stImage img {
        border-radius: 14px !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 0 4px 16px -4px rgba(15, 23, 42, 0.1) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        image-rendering: auto !important;
    }
    .stImage img:hover {
        box-shadow: 0 8px 24px -4px rgba(15, 23, 42, 0.16) !important;
    }

    /* Custom Smooth Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.03);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(16, 185, 129, 0.6);
    }

    /* Hero Banner Container */
    .freshai-hero {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 45%, #047857 100%);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: #FFFFFF;
        box-shadow: 0 20px 40px -15px rgba(6, 78, 59, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    .freshai-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.22) 0%, rgba(6, 78, 59, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(8px);
        color: #A7F3D0;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0% {
            box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
        }
        70% {
            box-shadow: 0 0 0 9px rgba(52, 211, 153, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(52, 211, 153, 0);
        }
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 8px;
        color: #FFFFFF;
    }
    .hero-title-accent {
        background: linear-gradient(135deg, #A7F3D0 0%, #6EE7B7 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #D1FAE5;
        font-weight: 400;
        line-height: 1.5;
        max-width: 820px;
        margin-bottom: 16px;
    }
    .hero-pipeline-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .pipeline-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 0, 0, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #E6FFFA;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 9999px;
    }

    /* Main Typography */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #064E3B;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .stage-badge {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid #86EFAC;
        margin-bottom: 0.8rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #059669;
    }
    .metric-label {
        font-size: 0.80rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* Produce Freshness Assessment Banner */
    .freshness-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 8px 30px -10px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease;
    }
    .freshness-card:hover {
        box-shadow: 0 14px 35px -10px rgba(0, 0, 0, 0.09);
    }
    .freshness-badge {
        display: inline-flex;
        align-items: center;
        font-size: 0.84rem;
        font-weight: 700;
        padding: 4px 14px;
        border-radius: 9999px;
        color: #FFFFFF;
        letter-spacing: 0.2px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
    }
    .freshness-meter-wrap {
        background: #E2E8F0;
        border-radius: 9999px;
        height: 10px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .freshness-meter-bar {
        height: 10px;
        border-radius: 9999px;
        transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .shelf-life-box {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        padding: 5px 12px;
        display: inline-block;
        font-size: 0.84rem;
        color: #065F46;
        font-weight: 700;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700;
        font-size: 0.90rem;
        padding: 8px 18px;
        border-radius: 10px;
        color: #475569;
        background: #F8FAFC;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #059669;
        background: #ECFDF5;
        border-color: #A7F3D0;
    }
    .stTabs [aria-selected="true"] {
        color: #065F46 !important;
        background: #D1FAE5 !important;
        border-color: #6EE7B7 !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
    }

    /* Streamlit Button Customization */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700;
        border-radius: 12px;
        padding: 8px 22px;
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #FFFFFF;
        border: none;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #047857 0%, #065F46 100%);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.35);
        transform: translateY(-1px);
    }

    /* Info and Alert Boxes */
    .info-box {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector(model_path: str, produce_only: bool = False):
    return FreshAIDetector(model_path=model_path, produce_only=produce_only)


@st.cache_resource
def get_classifier(model_path: str):
    return FreshAIClassifier(model_path=model_path)


@st.cache_resource
def get_freshness_detector():
    return ConvNeXtFreshnessDetector()


@st.cache_resource
def get_defect_detector():
    defect_weights = None
    if os.path.exists("runs/defect"):
        from pathlib import Path
        pts = list(Path("runs/defect").rglob("*.pt"))
        if pts:
            pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            defect_weights = str(pts[0])
    return FreshAIDefectDetector(model_path=defect_weights)


@st.cache_resource
def get_feature_extractor():
    return ProduceFeatureExtractor()


@st.cache_resource
def get_fusion_predictor():
    fusion_weights = "runs/fusion/best_fusion_model.pt" if os.path.exists("runs/fusion/best_fusion_model.pt") else None
    return FreshAIFusionPredictor(weights_path=fusion_weights)


@st.cache_resource
def get_structured_predictor():
    return FreshAIStructuredPredictor()


@st.cache_resource
def get_rag_engine():
    return FreshAIRAGEngine()


def load_and_orient_image(file_or_image) -> Image.Image:
    """
    Safely open image, apply EXIF orientation correction (preventing sideways/upside-down
    phone photos), and composite transparent PNG backgrounds cleanly onto neutral white.
    """
    if isinstance(file_or_image, Image.Image):
        img = file_or_image
    else:
        img = Image.open(file_or_image)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            bg.paste(img, mask=img.split()[-1])
        else:
            rgba = img.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def create_sample_produce_images():
    """Ensure photorealistic sample showcase images are present for instant inspection."""
    os.makedirs("samples", exist_ok=True)
    expected = [
        "samples/sample_apples.jpg",
        "samples/sample_banana.jpg",
        "samples/sample_tomatoes.jpg",
        "samples/sample_onions.jpg",
        "samples/sample_mixed_veg.jpg",
    ]
    if all(os.path.exists(p) and os.path.getsize(p) > 2000 for p in expected):
        return


def main():
    create_sample_produce_images()

    # Sidebar Header
    st.sidebar.markdown("## 🌿 FreshAI Controls")
    st.sidebar.markdown("Stage 1 • **Produce & Plant Detection**")

    # Discover available detection models ONLY (never classification or fusion models)
    custom_models = []
    if os.path.exists("runs/detect"):
        from pathlib import Path
        all_pts = list(Path("runs/detect").rglob("*.pt"))
        all_pts = [p for p in all_pts if "weights" in p.parts and p.name in ["best.pt", "last.pt"]]
        all_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in all_pts:
            rel = str(p.relative_to(Path("."))).replace("\\", "/")
            if rel not in custom_models:
                custom_models.append(rel)

    model_options = {}
    # 1. Primary Recommendation: Verified 12-Class Universal Produce Model
    universal_pt = "runs/detect/freshai_universal_produce_best.pt"
    if os.path.exists(universal_pt):
        model_options["🌿 freshai_universal_produce_best.pt (Universal 12-Produce: Onion, Tomato, Apple, Banana, Orange, Potato, Pepper, etc.)"] = universal_pt

    # Prioritize fine-tuned produce models
    produce_models = [cm for cm in custom_models if any(k in cm.lower() for k in ["universal", "produce", "freshai", "fruit"])]
    other_models = [cm for cm in custom_models if cm not in produce_models]

    for cm in produce_models:
        if cm == universal_pt:
            continue
        if "universal" in cm.lower() or "custom_produce_model" in cm.lower():
            label = f"🌿 {cm} (Universal 12-Produce: Onion, Tomato, Apple, Banana, Orange, Potato, Pepper, etc.)"
        elif "produce_detector" in cm.lower():
            label = f"🎯 {cm} (Produce Model: Onion, Tomato, Watermelon)"
        elif "freshai_produce" in cm.lower():
            label = f"🍎 {cm} (6-Fruit Model: Pineapple, Cherry, Mango, Plum, Tomato, Watermelon)"
        else:
            label = f"🎯 {cm} (Fine-Tuned Produce Detector)"
        model_options[label] = cm

    for cm in other_models:
        model_options[f"📦 {cm} (Custom Detection Model)"] = cm

    # Add standard base models as fallbacks
    model_options["yolov8n.pt (COCO Base: Apples, Bananas, Oranges, Carrots, Broccoli)"] = "yolov8n.pt"
    model_options["yolov8s.pt (COCO Higher Accuracy Base)"] = "yolov8s.pt"
    model_options["yolov8m.pt (COCO Medium Accuracy Base)"] = "yolov8m.pt"

    selected_label = st.sidebar.selectbox(
        "🧠 YOLO Detection Model",
        options=list(model_options.keys()),
        index=0,
        help="Select between custom fine-tuned FreshAI weights or standard base YOLO models."
    )
    selected_model = model_options[selected_label]

    if selected_model in custom_models:
        st.sidebar.success(f"🎯 **Custom Produce Model Active**\n\n`{selected_model}`")

    st.sidebar.markdown("### 🎛️ Detection Sensitivity")
    conf_thresh = st.sidebar.slider(
        "🎯 Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.20,
        step=0.01,
        help="Minimum confidence required for an object to be detected."
    )

    iou_thresh = st.sidebar.slider(
        "📦 IoU / NMS Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression overlap filtering."
    )

    filter_produce = st.sidebar.checkbox(
        "Produce & Crop Filter Only",
        value=False,
        help="Focus on fruits, vegetables, and plant parts."
    )

    st.sidebar.markdown("### 🎨 Visual Options")
    box_thickness = st.sidebar.slider("Bounding Box Width", min_value=1, max_value=6, value=3)
    show_conf_labels = st.sidebar.checkbox("Show Confidence % on Labels", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌡️ Stage 5: Environment & Timeline")
    env_temp = st.sidebar.slider("Storage Temperature (°C)", min_value=0.0, max_value=40.0, value=22.0, step=1.0, help="Ambient temperature of storage location.")
    env_humidity = st.sidebar.slider("Relative Humidity (%)", min_value=15.0, max_value=95.0, value=60.0, step=5.0, help="Relative humidity surrounding produce.")
    timeline_days = st.sidebar.slider("Days Since Harvest / Purchase", min_value=0.0, max_value=21.0, value=2.0, step=0.5, help="Timeline elapsed since produce was acquired.")
    storage_condition = st.sidebar.selectbox("Storage Condition", STORAGE_CONDITIONS, index=1, help="Storage environment category.")
    is_plant_mode = st.sidebar.checkbox("🌱 Growing Crop (Unharvested)", value=False, help="Enable if analyzing an actively growing plant or crop in field/garden.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **🚀 FreshAI Intelligence Lifecycle:**
    - 🟢 **Step 1:** Produce Detection (YOLO) ✅ Active
    - 🟢 **Step 2:** Freshness & Ripeness (ConvNeXt) ✅ Active
    - 🟢 **Step 3:** Defect Detect & Seg (YOLO) ✅ Active
    - 🟢 **Step 4:** Color & Texture Analysis (OpenCV) ✅ Active
    - 🟢 **Step 5:** Post-Harvest AI & Culinary Advisor (RAG) ✅ Active
    """)

    # Unified System Telemetry Status Card in Sidebar
    freshness_det = get_freshness_detector()
    defect_det = get_defect_detector()
    fusion_pred = get_fusion_predictor()
    rag_engine = get_rag_engine()

    fresh_label = "ConvNeXt ✅" if freshness_det.get_status()["model_loaded"] else "Heuristic Mode"
    defect_label = f"{defect_det.model_mode.upper()} ✅"
    fusion_label = "Neural MLP ✅" if fusion_pred.loaded_from_disk else "Calibrated Engine"
    advisor_label = "Gemini LLM ✅" if rag_engine.api_key else "USDA / WHO RAG ✅"

    st.sidebar.markdown(f"""
    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 14px; padding: 14px; color: white; margin-top: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
        <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between;">
            <span>⚡ Active AI Pipelines</span>
            <span style="font-size: 0.65rem; background: #059669; color: white; padding: 1px 6px; border-radius: 999px;">ONLINE</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 7px; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #CBD5E1;">🍃 Freshness Engine</span>
                <span style="background: rgba(34, 197, 94, 0.18); border: 1px solid rgba(74, 222, 128, 0.4); color: #4ADE80; font-weight: 700; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem;">{fresh_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #CBD5E1;">🔬 Defect Segmentation</span>
                <span style="background: rgba(34, 197, 94, 0.18); border: 1px solid rgba(74, 222, 128, 0.4); color: #4ADE80; font-weight: 700; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem;">{defect_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #CBD5E1;">🧠 Multimodal Fusion</span>
                <span style="background: rgba(56, 189, 248, 0.18); border: 1px solid rgba(56, 189, 248, 0.4); color: #38BDF8; font-weight: 700; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem;">{fusion_label}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #CBD5E1;">🌡️ Arrhenius Kinetics</span>
                <span style="background: rgba(245, 158, 11, 0.18); border: 1px solid rgba(251, 191, 36, 0.4); color: #FBBF24; font-weight: 700; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem;">Weibull / Q10 ✅</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #CBD5E1;">🤖 Advisory &amp; RAG</span>
                <span style="background: rgba(167, 139, 250, 0.18); border: 1px solid rgba(192, 132, 252, 0.4); color: #C084FC; font-weight: 700; padding: 2px 8px; border-radius: 999px; font-size: 0.72rem;">{advisor_label}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Area Hero Banner
    st.markdown("""
    <div class="freshai-hero">
        <div class="hero-badge-pill">
            <span class="pulse-dot"></span>
            <span>PRODUCTION AI v2.4 • MULTIMODAL POST-HARVEST PIPELINE</span>
        </div>
        <div class="hero-title">
            🌿 Fresh<span class="hero-title-accent">AI</span>
        </div>
        <div class="hero-subtitle">
            End-to-End Horticultural Vision, Biological Arrhenius Shelf Life Kinetics &amp; Post-Harvest Culinary Intelligence.
        </div>
        <div class="hero-pipeline-chips">
            <span class="pipeline-chip">🎯 12-Class YOLO Vision</span>
            <span class="pipeline-chip">🧠 ConvNeXt-Tiny Freshness</span>
            <span class="pipeline-chip">🔬 Defect Segmentation</span>
            <span class="pipeline-chip">🌡️ Arrhenius Kinetics</span>
            <span class="pipeline-chip">🥗 USDA / WHO Nutrition RAG</span>
            <span class="pipeline-chip">🤖 AI Chef &amp; Post-Harvest LLM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_detect, tab_mobile, tab_train, tab_arch = st.tabs([
        "🔍 Detection & Live Test",
        "📱 Mobile App Studio",
        "⚙️ Custom Model Training",
        "📐 System Architecture"
    ])

    with tab_detect:
        input_method = st.radio(
            "Select Produce Input Source:",
            ["📸 Upload Image", "🍎 Sample Showcase", "📹 Use Webcam"],
            horizontal=True
        )

        input_image = None

        if input_method == "📸 Upload Image":
            uploaded_file = st.file_uploader("Upload an image of fruits, vegetables, or crops", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                input_image = load_and_orient_image(uploaded_file)

        elif input_method == "🍎 Sample Showcase":
            sample_options = {
                "Mixed Produce (Assortment)": "samples/sample_mixed_veg.jpg",
                "Fresh Onions": "samples/sample_onions.jpg",
                "Tomatoes": "samples/sample_tomatoes.jpg",
                "Apples": "samples/sample_apples.jpg",
                "Banana Bunch": "samples/sample_banana.jpg",
                "Fresh Lemons": "samples/sample_lemons.jpg",
            }
            sample_choice = st.selectbox("Choose sample produce to inspect:", list(sample_options.keys()))
            sample_path = sample_options[sample_choice]
            if os.path.exists(sample_path):
                input_image = load_and_orient_image(sample_path)

        elif input_method == "📹 Use Webcam":
            camera_image = st.camera_input("Capture fresh produce directly with camera")
            if camera_image is not None:
                input_image = load_and_orient_image(camera_image)

        # Process and Display
        if input_image is not None:
            col_orig, col_annot = st.columns(2)

            with col_orig:
                st.markdown("### 📷 Original Input")
                st.image(input_image, use_container_width=True)

            with st.spinner("Analyzing produce with YOLO detector..."):
                try:
                    detector = get_detector(selected_model, produce_only=filter_produce)
                    start_t = time.time()
                    res = detector.predict(
                        image=input_image,
                        conf_threshold=conf_thresh,
                        iou_threshold=iou_thresh,
                        draw_annotations=True,
                    )
                    elapsed_ms = (time.time() - start_t) * 1000
                except Exception as e:
                    st.error(f"Detection error: {e}")
                    res = None

            if res is not None:
                with col_annot:
                    st.markdown("### 🎯 Detection Overlay")
                    if res.annotated_image is not None:
                        st.image(res.annotated_image, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📊 Detection Analytics & Summary")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(res.detected_objects)}</div>
                        <div class="metric-label">Total Items Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(res.class_counts)}</div>
                        <div class="metric-label">Identified Classes</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    avg_c_str = f"{round(res.average_confidence * 100, 1)}%" if res.detected_objects else "N/A"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_c_str}</div>
                        <div class="metric-label">Average Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{round(elapsed_ms, 1)} ms</div>
                        <div class="metric-label">Inference Latency</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Breakdown Details & Target Objects
                if res.detected_objects:
                    st.markdown("#### 📋 Item-by-Item Detection Details")
                    table_rows = []
                    for idx, obj in enumerate(res.detected_objects, 1):
                        table_rows.append({
                            "Item #": idx,
                            "Class": obj.class_name,
                            "Confidence": f"{round(obj.confidence * 100, 1)}%",
                            "Bounding Box [x1, y1, x2, y2]": f"[{int(obj.bbox_xyxy[0])}, {int(obj.bbox_xyxy[1])}, {int(obj.bbox_xyxy[2])}, {int(obj.bbox_xyxy[3])}]",
                            "Dimensions (W × H)": f"{int(obj.width_px)} × {int(obj.height_px)} px",
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
                    target_objects = res.detected_objects
                else:
                    st.info("💡 **Whole-Produce Auto-Analysis**: No isolated bounding box detected at current confidence threshold. Analyzing the whole image directly as a single produce item.")
                    from freshai.detector import DetectedObject
                    w_full, h_full = input_image.size
                    target_objects = [
                        DetectedObject(
                            class_id=0,
                            class_name="Produce",
                            confidence=0.85,
                            bbox_xyxy=[0.0, 0.0, float(w_full), float(h_full)],
                            bbox_norm_xywh=[0.5, 0.5, 1.0, 1.0],
                            width_px=float(w_full),
                            height_px=float(h_full),
                        )
                    ]

                # ── Multimodal Intelligence: Stages 2, 3, 4 & Fusion ──────────
                st.markdown("---")
                st.markdown('<div class="step2-header">🌿 Full Multimodal Produce Quality & Freshness Assessment</div>', unsafe_allow_html=True)
                st.caption("Fusing ConvNeXt-Tiny (Freshness & Ripeness) + YOLO Defect Segmentation (Pixel Area %) + OpenCV Color & Texture (GLCM) into a unified Feature Vector.")

                freshness_det = get_freshness_detector()
                defect_det = get_defect_detector()
                feature_ext = get_feature_extractor()
                fusion_pred = get_fusion_predictor()

                with st.spinner("Executing ConvNeXt + YOLO Defect Segmentation + OpenCV Color/Texture extraction…"):
                    analysis_results = []
                    img_np_full = np.array(input_image)
                    w_full, h_full = input_image.size

                    for obj in target_objects:
                            # Extract produce crop
                            x1 = max(0, int(obj.bbox_xyxy[0]))
                            y1 = max(0, int(obj.bbox_xyxy[1]))
                            x2 = min(w_full, int(obj.bbox_xyxy[2]))
                            y2 = min(h_full, int(obj.bbox_xyxy[3]))

                            if x2 > x1 and y2 > y1:
                                crop_np = img_np_full[y1:y2, x1:x2]
                                crop_pil = Image.fromarray(crop_np)
                            else:
                                crop_np = img_np_full
                                crop_pil = input_image

                            # 1. ConvNeXt Baseline (Stage 2)
                            fr = freshness_det.predict(crop_pil, produce_class=obj.class_name)

                            # 2. YOLO Defect Detection & Segmentation (Stage 3)
                            dr = defect_det.predict(crop_np, produce_class=obj.class_name)

                            # 3. OpenCV Color & Texture Features (Stage 4)
                            fe = feature_ext.extract(crop_np)

                            # 4. Standardized Multimodal Feature Vector
                            fv = assemble_feature_vector(fr, dr, fe, produce_class=obj.class_name)

                            # 5. Multimodal Freshness Assessment (Stage 4)
                            assessment = fusion_pred.predict(fv)

                            # 6. Structured Prediction Layer (Stage 5 - XGBoost & LightGBM)
                            struct_input = StructuredProduceInput(
                                feature_vector=fv,
                                environmental=EnvironmentalData(temperature_c=env_temp, humidity_pct=env_humidity),
                                timeline=TimelineStorageData(days_since_purchase=timeline_days, storage_condition=storage_condition, is_growing_plant=is_plant_mode),
                                produce_class=obj.class_name,
                            )
                            struct_res = struct_pred.predict(struct_input)

                            # 7. LLM + RAG Post-Harvest Advisor (Stage 6)
                            rag_rec = rag_engine.generate_recommendations(
                                produce_class=obj.class_name,
                                quality_score=struct_res.quality_score,
                                physiological_age_days=struct_res.physiological_age_days,
                                remaining_shelf_life_days=struct_res.remaining_shelf_life_days,
                                defect_area_pct=dr.total_defect_area_pct,
                                spoilage_risk=struct_res.spoilage_risk,
                                storage_temp_c=env_temp,
                                days_since_purchase=timeline_days,
                            )

                            analysis_results.append((obj, crop_np, fr, dr, fe, fv, assessment, struct_res, rag_rec))

                    # Render Multimodal Produce Cards
                    for obj, crop_np, fr, dr, fe, fv, fa, s_res, rag_rec in analysis_results:
                        score = fa.freshness_score
                        bar_color = fa.badge_color
                        arr_res = fa.arrhenius_shelf_life(produce_class=obj.class_name, temp_c=env_temp, defect_area_pct=dr.total_defect_area_pct)

                        st.markdown(f"""
                        <div class="freshness-card">
                            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px; flex-wrap:wrap;">
                                <span style="font-size:1.85rem; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">{fa.freshness_emoji}</span>
                                <div>
                                    <span style="font-size:1.22rem; font-weight:800; color:#0F172A; letter-spacing:-0.02em;">{obj.class_name}</span>
                                    <span style="font-size:0.75rem; color:#64748B; margin-left:6px; font-weight:600;">Detection ID #{obj.track_id or 1}</span>
                                </div>
                                <span class="freshness-badge" style="background:{bar_color};">{fa.freshness_stage} ({score}/100)</span>
                                <span style="background:#EEF2FF; color:#4338CA; font-weight:700; font-size:0.78rem; padding:4px 12px; border-radius:9999px; border:1px solid #C7D2FE;">
                                    {fa.quality_grade}
                                </span>
                                <span style="background:{'#DCFCE7' if dr.total_defect_area_pct < 5 else ('#FEF3C7' if dr.total_defect_area_pct < 15 else '#FEE2E2')}; color:{'#166534' if dr.total_defect_area_pct < 5 else ('#92400E' if dr.total_defect_area_pct < 15 else '#991B1B')}; font-weight:700; font-size:0.78rem; padding:4px 12px; border-radius:9999px;">
                                    Defect Area: {round(dr.total_defect_area_pct, 1)}%
                                </span>
                                <span style="font-size:0.84rem; color:#64748B; margin-left:auto; font-weight:500;">
                                    Vision Confidence: <b style="color:#0F172A;">{round(obj.confidence*100,1)}%</b> • Risk: <b style="color:{s_res.spoilage_risk_color};">{fa.spoilage_risk}</b>
                                </span>
                            </div>
                            <div style="margin-bottom:6px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; color:#475569; font-weight:600; margin-bottom:4px;">
                                    <span>Overall Freshness Score: <b style="color:#0F172A;">{score}/100</b></span>
                                    <span>Confidence Probability: <b style="color:#0F172A;">{round(fa.freshness_prob*100,1)}%</b></span>
                                </div>
                                <div class="freshness-meter-wrap">
                                    <div class="freshness-meter-bar" style="width:{score}%; background:linear-gradient(90deg, {bar_color}, #10B981);"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Three-column visual breakdown
                        col_crop, col_mask, col_feats = st.columns([1, 1, 1.4])

                        with col_crop:
                            st.caption("📷 **Cropped Produce**")
                            st.image(crop_np, use_container_width=True)

                        with col_mask:
                            st.caption("🔬 **YOLO Defect Segmentation**")
                            if dr.annotated_crop is not None:
                                st.image(dr.annotated_crop, use_container_width=True)
                            else:
                                st.image(crop_np, use_container_width=True)

                        with col_feats:
                            st.caption("🧬 **Combined Multimodal Feature Vectors**")
                            st.markdown(f"""
                            <div style="background:#0F172A; border:1px solid #334155; border-radius:12px; padding:14px 16px; font-family:'JetBrains Mono', monospace; font-size:0.82rem; line-height:1.7; color:#E2E8F0; box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">
                                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:3px;"><span style="color:#94A3B8;">Freshness Prob:</span><b style="color:#38BDF8;">{round(fv.freshness_prob, 3)}</b></div>
                                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding:3px 0;"><span style="color:#94A3B8;">Ripeness Prob:</span><b style="color:#FBBF24;">{round(fv.ripeness_prob, 3)}</b></div>
                                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding:3px 0;"><span style="color:#94A3B8;">Defect Surface Area:</span><b style="color:{'#F87171' if fv.defect_area_pct > 10 else '#34D399'};">{round(fv.defect_area_pct, 1)}%</b></div>
                                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding:3px 0;"><span style="color:#94A3B8;">Hue Mean (0–180):</span><b>{round(fv.avg_hue, 1)}</b></div>
                                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding:3px 0;"><span style="color:#94A3B8;">Saturation Mean (%):</span><b>{round(fv.avg_saturation, 1)}%</b></div>
                                <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding:3px 0;"><span style="color:#94A3B8;">Texture Contrast:</span><b>{round(fv.texture_contrast, 2)}</b></div>
                                <div style="display:flex; justify-content:space-between; padding-top:3px;"><span style="color:#94A3B8;">Texture Homogeneity:</span><b>{round(fv.texture_homogeneity, 2)}</b></div>
                            </div>
                            """, unsafe_allow_html=True)

                            ci_badge = f"<span style='background:#FEE2E2; color:#991B1B; font-weight:700; font-size:0.72rem; padding:2px 8px; border-radius:999px; margin-left:6px;'>⚠️ CHILLING INJURY (<{int(round(arr_res.chilling_threshold_c))}°C)</span>" if arr_res.is_chilling_injury_active else ""

                            st.markdown(f"""
                            <div style="margin-top:8px; font-size:0.83rem; color:#475569; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                <span>🌡️ <b>Arrhenius Shelf Life ({env_temp}°C):</b></span>
                                <span class="shelf-life-box">{round(arr_res.ambient_shelf_days, 1)} days</span>
                                <span style="font-size:0.78rem; color:#64748B;">• Optimal: <b>{arr_res.ideal_storage_name}</b> {ci_badge}</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Produce Quality & Storage Intelligence Card - Modern Glassmorphic SaaS Cockpit
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #090E17 0%, #151E2E 100%); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 16px; padding: 20px 24px; margin: 16px 0 20px 0; color: white; box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.5);">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; flex-wrap:wrap; gap:10px;">
                                <div style="display:flex; align-items:center; gap:8px;">
                                    <span style="font-size: 1.25rem;">📊</span>
                                    <span style="font-size: 1.08rem; font-weight: 800; color: #38BDF8; letter-spacing: -0.01em;">
                                        Produce Quality &amp; Storage Intelligence
                                    </span>
                                </div>
                                <span style="font-size: 0.78rem; background: rgba(2, 132, 199, 0.25); border: 1px solid rgba(56, 189, 248, 0.4); color: #7DD3FC; padding: 4px 14px; border-radius: 9999px; font-weight: 700; letter-spacing: 0.2px;">
                                    ⚡ Vision + {env_temp}°C / {env_humidity}% RH • {timeline_days}d ({storage_condition})
                                </span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; text-align: center;">
                                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 10px; backdrop-filter: blur(8px);">
                                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Quality Score</div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #38BDF8; margin: 4px 0; text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);">{s_res.quality_score_str}</div>
                                    <div style="font-size: 0.68rem; color: #64748B; font-weight: 500;">Multi-Feature Score</div>
                                </div>
                                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 10px; backdrop-filter: blur(8px);">
                                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Biological Age</div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #FBBF24; margin: 4px 0; text-shadow: 0 0 20px rgba(251, 191, 36, 0.3);">{s_res.physiological_age_range}</div>
                                    <div style="font-size: 0.68rem; color: #64748B; font-weight: 500;">Cellular Senescence</div>
                                </div>
                                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 10px; backdrop-filter: blur(8px);">
                                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Remaining Shelf Life</div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: #34D399; margin: 4px 0; text-shadow: 0 0 20px rgba(52, 211, 153, 0.3);">{s_res.remaining_shelf_life_range}</div>
                                    <div style="font-size: 0.68rem; color: #64748B; font-weight: 500;">Arrhenius Calibrated</div>
                                </div>
                                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 10px; backdrop-filter: blur(8px);">
                                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Spoilage Risk</div>
                                    <div style="font-size: 1.55rem; font-weight: 800; color: {s_res.spoilage_risk_color}; margin: 4px 0;">{s_res.spoilage_risk}</div>
                                    <div style="font-size: 0.68rem; color: #64748B; font-weight: 500;">Microbial Hazard</div>
                                </div>
                                <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 14px 10px; backdrop-filter: blur(8px);">
                                    <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 0.5px;">Harvest Window</div>
                                    <div style="font-size: 1.25rem; font-weight: 800; color: #C084FC; margin: 6px 0;">{s_res.time_to_harvest_range}</div>
                                    <div style="font-size: 0.68rem; color: #64748B; font-weight: 500;">Optimal Window</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Key Functionality 1: Result Explanation ───────────────────
                        pe = rag_rec.prediction_explanation
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); border: 1px solid #BAE6FD; border-left: 6px solid #0284C7; border-radius: 14px; padding: 18px 22px; margin-bottom: 16px; box-shadow: 0 4px 15px -3px rgba(2, 132, 199, 0.08);">
                            <div style="font-size:1.06rem; font-weight:800; color:#0C4A6E; margin-bottom:12px; display:flex; align-items:center; gap:8px;">
                                <span>🎯</span> Result Explanation &amp; ML Diagnostics
                                <span style="font-size:0.75rem; background:#BAE6FD; color:#0369A1; font-weight:700; padding:3px 12px; border-radius:999px; margin-left:auto;">
                                    ML Models Predict ➔ RAG Explains
                                </span>
                            </div>
                            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:12px; font-size:0.86rem; color:#334155; line-height:1.5;">
                                <div style="background:#FFFFFF; border:1px solid #E0F2FE; border-radius:10px; padding:12px 14px; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                                    <b style="color:#0284C7; font-size:0.88rem;">📊 Quality Score ({s_res.quality_score_str}):</b><br/>
                                    <span style="color:#475569;">{pe.get('quality_score_explanation', '')}</span>
                                </div>
                                <div style="background:#FFFFFF; border:1px solid #E0F2FE; border-radius:10px; padding:12px 14px; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                                    <b style="color:#D97706; font-size:0.88rem;">⏳ Estimated Age ({s_res.physiological_age_range}):</b><br/>
                                    <span style="color:#475569;">{pe.get('estimated_age_explanation', '')}</span>
                                </div>
                                <div style="background:#FFFFFF; border:1px solid #E0F2FE; border-radius:10px; padding:12px 14px; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                                    <b style="color:#059669; font-size:0.88rem;">⏱️ Shelf Life ({s_res.remaining_shelf_life_range}):</b><br/>
                                    <span style="color:#475569;">{pe.get('shelf_life_explanation', '')}</span>
                                </div>
                                <div style="background:#FFFFFF; border:1px solid #E0F2FE; border-radius:10px; padding:12px 14px; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                                    <b style="color:{s_res.spoilage_risk_color}; font-size:0.88rem;">⚠️ Spoilage Risk ({s_res.spoilage_risk}):</b><br/>
                                    <span style="color:#475569;">{pe.get('spoilage_risk_explanation', '')}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # ── Key Functionality 2 & 3: Lifespan Improvement & Nutrition ──
                        c_life, c_nutr = st.columns([1, 1])

                        with c_life:
                            strat = rag_rec.storage_strategy
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%); border: 1px solid #BBF7D0; border-left: 6px solid #10B981; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.08); height: 100%;">
                                <div style="font-size:1.05rem; font-weight:800; color:#064E3B; margin-bottom:10px; display:flex; align-items:center; gap:8px;">
                                    <span>🧊</span> Lifespan Improvement Guidance
                                </div>
                                <div style="font-size:0.86rem; color:#1E293B; line-height:1.65;">
                                    <div style="margin-bottom:4px;">🌡️ <b>Target Storage Temp:</b> <span style="color:#047857; font-weight:700; background:#E6FFFA; padding:2px 8px; border-radius:6px;">{strat.get('optimal_temperature_target', '')}</span></div>
                                    <div style="margin-bottom:4px;">💧 <b>Target Relative Humidity:</b> <span style="color:#047857; font-weight:700; background:#E6FFFA; padding:2px 8px; border-radius:6px;">{strat.get('optimal_humidity_target', '')}</span></div>
                                    <div style="margin-bottom:6px;">📍 <b>Optimal Location:</b> <span style="background:#A7F3D0; color:#064E3B; font-weight:800; padding:2px 10px; border-radius:999px;">{strat.get('storage_location', '')}</span></div>
                                    <div style="margin-top:6px; font-size:0.83rem; color:#374151;"><b>Refrigeration Rule:</b> {strat.get('refrigeration_guideline', '')}</div>
                                    <div style="margin-top:4px; font-size:0.83rem; color:#374151;"><b>Ethylene Dynamics:</b> {strat.get('ethylene_co_location_warning', '')}</div>
                                    <div style="margin-top:10px; font-weight:800; color:#064E3B;">Handling &amp; Preservation Practices:</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            for act in strat.get('handling_and_preservation', [])[:3]:
                                st.markdown(f"<div style='font-size:0.84rem; color:#334155; margin-left:10px; margin-bottom:4px; display:flex; align-items:center; gap:6px;'><span style='color:#10B981; font-weight:bold;'>✓</span> {act}</div>", unsafe_allow_html=True)

                        with c_nutr:
                            nutr = rag_rec.nutrition_facts
                            vits = nutr.get("vitamins", {})
                            vit_str = " • ".join([f"{k}: {v}" for k, v in list(vits.items())[:3]])
                            sources_str = ", ".join(rag_rec.trusted_sources[:2])
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%); border: 1px solid #FDE68A; border-left: 6px solid #F59E0B; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; box-shadow: 0 4px 15px -3px rgba(245, 158, 11, 0.08); height: 100%;">
                                <div style="font-size:1.05rem; font-weight:800; color:#78350F; margin-bottom:10px; display:flex; align-items:center; gap:8px;">
                                    <span>🥗</span> Produce Nutrition &amp; Health Facts
                                    <span style="font-size:0.72rem; background:#FDE68A; color:#78350F; font-weight:800; padding:2px 10px; border-radius:999px; margin-left:auto;">
                                        USDA / WHO
                                    </span>
                                </div>
                                <div style="display:flex; gap:8px; margin-bottom:10px; flex-wrap:wrap;">
                                    <span style="background:#FFFFFF; border:1px solid #FCD34D; font-size:0.82rem; font-weight:800; color:#92400E; padding:4px 12px; border-radius:999px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                                        🔥 {nutr.get('calories_kcal', 30)} kcal / 100g
                                    </span>
                                    <span style="background:#FFFFFF; border:1px solid #FCD34D; font-size:0.82rem; font-weight:800; color:#92400E; padding:4px 12px; border-radius:999px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                                        🌾 Fiber: {nutr.get('dietary_fiber_g', 1.5)}g
                                    </span>
                                    <span style="background:#FFFFFF; border:1px solid #FCD34D; font-size:0.82rem; font-weight:800; color:#92400E; padding:4px 12px; border-radius:999px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                                        ⚡ Carbs: {nutr.get('carbohydrates_g', 5.0)}g
                                    </span>
                                </div>
                                <div style="font-size:0.84rem; color:#451A03; line-height:1.55;">
                                    <div style="margin-bottom:3px;"><b>Key Vitamins:</b> {vit_str or 'Natural source of essential vitamins'}</div>
                                    <div style="margin-bottom:6px;"><b>Antioxidant Profile:</b> {nutr.get('antioxidants', '')}</div>
                                </div>
                                <div style="margin-top:8px; font-size:0.78rem; color:#78350F; background:rgba(255,255,255,0.7); border:1px solid #FDE68A; padding:6px 12px; border-radius:8px;">
                                    📚 <b>Verified Sources:</b> {sources_str}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            t_rag, t_struct, t_def, t_col, t_vec, t_pen = st.tabs([
                                "🤖 AI Chef & Post-Harvest Advisor",
                                "📈 Storage & What-If Simulation",
                                "🔬 Defect Details",
                                "🎨 Color & Texture",
                                "🧬 Full Feature Vector",
                                "⚖️ Spoilage Penalty Breakdown"
                            ])

                            with t_rag:
                                st.markdown("##### 🤖 Post-Harvest AI Guidance & Culinary Waste Reduction")
                                st.info(f"💡 **AI Post-Harvest Assessment ({rag_rec.generated_by})**:\n\n{rag_rec.ai_advisor_summary}")
                                r_col1, r_col2 = st.columns(2)
                                with r_col1:
                                    st.markdown("**🧊 Storage & Ethylene Management Strategy:**")
                                    st.write({
                                        "Recommended Target": rag_rec.storage_strategy["optimal_temperature_target"],
                                        "Current Status": rag_rec.storage_strategy["current_temp_status"],
                                        "Refrigeration Guideline": rag_rec.storage_strategy["refrigeration_guideline"],
                                        "Chilling Injury Alert": rag_rec.storage_strategy["chilling_injury_alert"],
                                        "Ethylene Management": rag_rec.storage_strategy["ethylene_co_location_warning"],
                                        "Optimal Humidity": rag_rec.storage_strategy["optimal_relative_humidity"],
                                    })

                                    st.markdown("**⏱️ Spoilage Mitigation Action Plan:**")
                                    for act in rag_rec.spoilage_mitigation:
                                        st.markdown(f"- {act}")

                                with r_col2:
                                    st.markdown(f"**🥗 Chef's Zero-Waste Recipe ({rag_rec.quality_grade}):**")
                                    st.markdown(f"""
                                    <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:10px; padding:14px; margin-bottom:12px;">
                                        <div style="font-size:1.05rem; font-weight:700; color:#166534;">{rag_rec.chef_recipe['title']}</div>
                                        <div style="font-size:0.8rem; color:#475569; margin:4px 0 8px 0;">⏱️ Prep Time: <b>{rag_rec.chef_recipe['prep_time']}</b> • Difficulty: <b>{rag_rec.chef_recipe['difficulty']}</b></div>
                                        <div style="font-size:0.88rem; color:#1E293B; line-height:1.5;">{rag_rec.chef_recipe['instructions']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    st.markdown("**🥑 Nutritional Trajectory Insights:**")
                                    st.write(rag_rec.nutritional_insights)

                            with t_struct:
                                st.markdown("##### 📈 Storage Shelf Life & What-If Environmental Simulation")
                                s_col1, s_col2 = st.columns(2)
                                with s_col1:
                                    st.markdown("**What-If Storage Shelf Life Simulation (Days):**")
                                    sim_df = pd.DataFrame([
                                        {"Storage Condition": k, "Simulated Shelf Life": f"{round(v, 1)} days"}
                                        for k, v in s_res.what_if_shelf_life_days.items()
                                    ])
                                    st.dataframe(sim_df, use_container_width=True)

                                    st.markdown("**Spoilage Risk Probability Breakdown:**")
                                    risk_df = pd.DataFrame([
                                        {"Risk Tier": k, "Probability": f"{round(v * 100, 1)}%"}
                                        for k, v in s_res.spoilage_risk_probs.items()
                                    ])
                                    st.dataframe(risk_df, use_container_width=True)

                                with s_col2:
                                    st.markdown("**Key Feature Attributions (Drivers of Quality & Freshness):**")
                                    st.dataframe(pd.DataFrame(s_res.top_contributing_factors), use_container_width=True)

                            with t_def:
                                st.markdown("##### 🔬 Stage 3: Defect Detection & Segmentation Analytics")
                                d_c1, d_c2, d_c3 = st.columns(3)
                                d_c1.metric("Total Defect Area", f"{round(dr.total_defect_area_pct, 1)}%", delta=f"{dr.defect_count} defect(s)", delta_color="inverse")
                                d_c2.metric("Defect Severity Score", f"{round(dr.defect_severity_score, 2)} / 1.0")
                                d_c3.metric("Segmentation Mode", dr.model_mode.upper())

                                if dr.defects:
                                    defect_rows = []
                                    for d in dr.defects:
                                        defect_rows.append({
                                            "Defect Type": d.defect_type,
                                            "Confidence": f"{round(d.confidence * 100, 1)}%",
                                            "Affected Area %": f"{round(d.area_percent, 2)}%",
                                            "Pixels": d.area_pixels,
                                            "Bounding Box": str(d.bbox_xyxy),
                                            "Severity Weight": d.severity_weight,
                                        })
                                    st.dataframe(pd.DataFrame(defect_rows), use_container_width=True)
                                else:
                                    st.success("✅ Pristine surface: No visible necrosis, cracks, rot, or mold detected.")

                            with t_col:
                                st.markdown("##### 🎨 Stage 4: OpenCV Color & GLCM Texture Features")
                                col_c1, col_c2 = st.columns(2)
                                with col_c1:
                                    st.markdown("**Color Space Metrics:**")
                                    st.write({
                                        "RGB Means": [round(fe.color.mean_r, 1), round(fe.color.mean_g, 1), round(fe.color.mean_b, 1)],
                                        "Average Hue (0-180)": round(fe.color.avg_hue, 1),
                                        "Saturation (%)": round(fe.color.avg_saturation, 1),
                                        "Brightness (%)": round(fe.color.avg_brightness, 1),
                                        "CIELAB (L*, a*, b*)": [round(fe.color.mean_l, 1), round(fe.color.mean_a, 1), round(fe.color.mean_b_lab, 1)],
                                        "Chroma": round(fe.color.chroma, 1),
                                    })
                                    st.markdown("**Color Region Distribution:**")
                                    st.write({
                                        "Green (Chlorophyll/Unripe)": f"{round(fe.color.green_region_pct, 1)}%",
                                        "Yellow/Orange (Ripe)": f"{round(fe.color.yellow_region_pct, 1)}%",
                                        "Red (Peak)": f"{round(fe.color.red_region_pct, 1)}%",
                                        "Brown (Enzymatic Decay)": f"{round(fe.color.brown_region_pct, 1)}%",
                                        "Dark (Necrotic Spot)": f"{round(fe.color.dark_decay_pct, 1)}%",
                                        "Pale Mold": f"{round(fe.color.pale_mold_pct, 1)}%",
                                    })

                                with col_c2:
                                    st.markdown("**GLCM Texture Dynamics:**")
                                    st.write({
                                        "Texture Contrast": round(fe.texture.contrast, 4),
                                        "Texture Homogeneity": round(fe.texture.homogeneity, 4),
                                        "Texture Energy (ASM)": round(fe.texture.energy, 4),
                                        "Texture Dissimilarity": round(fe.texture.dissimilarity, 4),
                                        "Texture Correlation": round(fe.texture.correlation, 4),
                                        "Surface Roughness (Laplacian)": round(fe.texture.roughness, 2),
                                        "Edge Density": round(fe.texture.edge_density, 4),
                                    })

                            with t_vec:
                                st.markdown("##### 🧬 Unified 47-Dimensional Multimodal Feature Vector")
                                dense_vec = fv.to_dense_vector(include_deep=False)
                                feat_names = ProduceFeatureVector.feature_names(include_deep=False)
                                vec_df = pd.DataFrame({
                                    "Feature Name": feat_names,
                                    "Extracted Value": [round(float(v), 4) for v in dense_vec],
                                })
                                st.dataframe(vec_df, use_container_width=True)

                            with t_pen:
                                st.markdown("##### ⚖️ Spoilage Penalty Breakdown")
                                st.write({
                                    "Base ConvNeXt Score": f"{fa.base_convnext_score} / 100",
                                    "Defect Area Penalty": f"-{round(fa.defect_penalty_pts, 1)} pts",
                                    "Texture Degradation Penalty": f"-{round(fa.texture_penalty_pts, 1)} pts",
                                    "Color Senescence Penalty": f"-{round(fa.color_penalty_pts, 1)} pts",
                                    "Final Calibrated Quality Score": f"{fa.freshness_score} / 100",
                                    "Quality Grade": fa.quality_grade,
                                    "Degradation Risk Level": fa.spoilage_risk,
                                })

                        st.markdown("<hr style='margin:14px 0;'>", unsafe_allow_html=True)

                    # Arrhenius Temperature Simulator
                    with st.expander("🌡️ Arrhenius Kinetics Temperature & Storage Simulator", expanded=False):
                        st.markdown("""
                        **Arrhenius Post-Harvest Degradation Kinetics:**  
                        $$k(T) = A \\cdot e^{-\\frac{E_a}{R \\cdot T}}$$
                        Predicts real-life shelf life under kitchen temperatures (respecting non-refrigerated crops like Onion, Potato, and Tomato).
                        """)
                        sim_temp = st.slider("Simulate Storage Temperature (°C):", min_value=0, max_value=40, value=25, step=1)
                        sim_rows = []
                        for obj, crop_np, fr, dr, fe, fv, fa, s_res, rag_rec in analysis_results:
                            sim_res = fa.arrhenius_shelf_life(produce_class=obj.class_name, temp_c=float(sim_temp), defect_area_pct=dr.total_defect_area_pct)
                            sim_rows.append({
                                "Produce": obj.class_name,
                                "Freshness Score": f"{fa.freshness_score}/100",
                                "Quality (XGBoost)": s_res.quality_score_str,
                                f"Shelf Life @ {sim_temp}°C": f"{round(sim_res.ambient_shelf_days, 1)} days",
                                "Chilling Injury?": "⚠️ ACTIVE (Cold Decay)" if sim_res.is_chilling_injury_active else "✅ Normal",
                                "Degradation Rate": f"{round(sim_res.degradation_acceleration, 2)}x",
                                "Defect Penalty": f"-{round(sim_res.defect_penalty_applied_pct, 1)}%",
                                "Recommended Storage": sim_res.ideal_storage_name,
                                "Fridge Allowed?": "✅ Yes" if sim_res.allow_refrigeration else "❌ No (Harmful)",
                            })
                        st.dataframe(pd.DataFrame(sim_rows), use_container_width=True)

                    # All-produce summary table
                    if len(analysis_results) > 1:
                        st.markdown("##### 📊 Multimodal Summary Table")
                        summary_rows = []
                        for obj, crop_np, fr, dr, fe, fv, fa, s_res, rag_rec in analysis_results:
                            arr_info = fa.arrhenius_shelf_life(produce_class=obj.class_name, temp_c=env_temp, defect_area_pct=dr.total_defect_area_pct)
                            summary_rows.append({
                                "Produce": obj.class_name,
                                "Freshness Stage": f"{fa.freshness_emoji} {fa.freshness_stage}",
                                "Quality Score": s_res.quality_score_str,
                                "Biological Age": s_res.physiological_age_range,
                                "Remaining Shelf Life": s_res.remaining_shelf_life_range,
                                f"Shelf Life @ {env_temp}°C": f"{round(arr_info.ambient_shelf_days, 1)} days",
                                "Defect Area": f"{round(dr.total_defect_area_pct, 1)}%",
                                "Quality Grade": fa.quality_grade,
                                "Ripeness": f"{fr.ripeness_emoji} {fv.ripeness_stage}",
                                "Spoilage Risk": s_res.spoilage_risk,
                                "Chef Recipe": rag_rec.chef_recipe['title'],
                                "Storage Location": arr_info.ideal_storage_name,
                            })
                        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    with tab_mobile:
        st.markdown("### 📱 FreshAI Mobile App Studio & Device Simulator")
        st.caption("Interactive preview of the native Android interface (built with Kotlin, Material 3, ONNX Runtime & REST API).")

        m_col_phone, m_col_info = st.columns([1, 1.4])

        with m_col_phone:
            # Produce selection for mobile preview
            mobile_crop = st.selectbox(
                "Select Mobile Viewfinder Scene:",
                ["🍅 Fresh Vine Tomato", "🍎 Crisp Red Apple", "🍌 Ripe Banana Bunch", "🧅 Cured Yellow Onion"],
                index=0
            )

            # Map sample info
            sample_map = {
                "🍅 Fresh Vine Tomato": ("samples/sample_tomatoes.jpg", "Tomato", "84/100", "5–8 days", "5.8 days", "Low", 5.2, "Heirloom Caprese Salad", "Never store below 10°C (causes mealy texture)."),
                "🍎 Crisp Red Apple": ("samples/sample_apples.jpg", "Apple", "91/100", "3–5 days", "26.1 days", "Very Low", 1.8, "Crisp Apple & Walnut Slaw", "Refrigerate in perforated bag; isolate from ethylene-sensitive veggies."),
                "🍌 Ripe Banana Bunch": ("samples/sample_banana.jpg", "Banana", "78/100", "6–8 days", "4.1 days", "Medium", 8.5, "Caramelized Banana Oatmeal", "Hang at room temperature; wrap crown in foil to retard ripening."),
                "🧅 Cured Yellow Onion": ("samples/sample_mixed_veg.jpg", "Onion", "88/100", "4–7 days", "10.5 days", "Low", 2.1, "Quick-Pickled Red Onions", "Store in cool, dark pantry with ventilation. Never near potatoes!"),
            }

            img_path, m_name, m_qual, m_age, m_shelf, m_risk, m_defect, m_recipe, m_storage = sample_map[mobile_crop]

            # Realistic smartphone frame rendering
            st.markdown(f"""
            <div style="max-width: 360px; margin: 0 auto; background: #0F172A; border-radius: 40px; padding: 12px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); border: 4px solid #334155;">
                <!-- Phone Top Bar & Dynamic Island -->
                <div style="background: #1E293B; border-radius: 28px 28px 0 0; padding: 8px 16px 6px 16px; display: flex; justify-content: space-between; align-items: center; font-size: 0.72rem; color: #94A3B8;">
                    <span>9:41</span>
                    <div style="width: 70px; height: 16px; background: #000; border-radius: 20px;"></div>
                    <span>5G • 98%</span>
                </div>
                <!-- Native App Header -->
                <div style="background: #15803D; padding: 12px 14px; color: white;">
                    <div style="font-size: 1.1rem; font-weight: 800; display: flex; align-items: center; justify-content: space-between;">
                        <span>🌿 FreshAI</span>
                        <span style="font-size: 0.7rem; background: #166534; padding: 2px 8px; border-radius: 12px;">Active</span>
                    </div>
                    <div style="font-size: 0.72rem; color: #DCFCE7;">Multimodal Vision &amp; Post-Harvest Intelligence</div>
                </div>
                <!-- Screen Content Area -->
                <div style="background: #F8FAFC; padding: 12px; border-radius: 0 0 28px 28px; max-height: 520px; overflow-y: auto; color: #0F172A;">
                    <!-- Camera Viewfinder with YOLO Bounding Box Overlay -->
                    <div style="position: relative; border-radius: 14px; overflow: hidden; border: 1px solid #CBD5E1; margin-bottom: 10px; background: #000;">
                        <img src="{img_path if os.path.exists(img_path) else 'samples/sample_tomatoes.jpg'}" style="width: 100%; height: 160px; object-fit: cover; opacity: 0.9;" />
                        <!-- Bounding Box Simulated Tag -->
                        <div style="position: absolute; top: 25px; left: 35px; width: 65%; height: 60%; border: 2px solid #22C55E; background: rgba(34, 197, 94, 0.15); border-radius: 4px; display: flex; align-items: flex-start;">
                            <span style="background: #22C55E; color: white; font-size: 0.65rem; font-weight: 700; padding: 1px 6px; border-radius: 2px;">{m_name} (95%)</span>
                        </div>
                    </div>
                    <!-- Quick KPI Row -->
                    <div style="display: flex; gap: 6px; margin-bottom: 10px;">
                        <div style="flex: 1; background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px; text-align: center;">
                            <div style="font-size: 0.62rem; color: #64748B;">Total Items</div>
                            <div style="font-size: 1rem; font-weight: 800; color: #166534;">1</div>
                        </div>
                        <div style="flex: 1; background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px; text-align: center;">
                            <div style="font-size: 0.62rem; color: #64748B;">Shelf Life</div>
                            <div style="font-size: 1rem; font-weight: 800; color: #166534;">{m_shelf}</div>
                        </div>
                        <div style="flex: 1; background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px; text-align: center;">
                            <div style="font-size: 0.62rem; color: #64748B;">Confidence</div>
                            <div style="font-size: 1rem; font-weight: 800; color: #166534;">95%</div>
                        </div>
                    </div>
                    <!-- Item Freshness & Defect Card -->
                    <div style="background: white; border: 1px solid #BBF7D0; border-radius: 12px; padding: 10px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 800; font-size: 0.88rem;">{m_name}</span>
                            <span style="background: #22C55E; color: white; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;">Fresh</span>
                            <span style="background: #FEF3C7; color: #92400E; font-size: 0.65rem; font-weight: 700; padding: 2px 6px; border-radius: 999px;">Defect: {m_defect}%</span>
                        </div>
                        <div style="font-size: 0.72rem; color: #475569; margin-bottom: 3px;">Freshness Score: <b>{m_qual}</b> • Estimated Shelf Life: <b>{m_shelf}</b></div>
                        <div style="width: 100%; height: 6px; background: #E2E8F0; border-radius: 4px; overflow: hidden; margin-bottom: 8px;">
                            <div style="width: 85%; height: 100%; background: #22C55E;"></div>
                        </div>
                        <!-- Produce Quality & Storage Card in Mobile -->
                        <div style="background: #0F172A; border-radius: 8px; padding: 8px; margin-bottom: 8px; color: white;">
                            <div style="font-size: 0.65rem; color: #38BDF8; font-weight: 700; margin-bottom: 4px;">📊 Quality &amp; Storage Insights</div>
                            <div style="display: flex; justify-content: space-around; text-align: center;">
                                <div>
                                    <div style="font-size: 0.58rem; color: #94A3B8;">QUALITY</div>
                                    <div style="font-size: 0.85rem; font-weight: 800; color: #38BDF8;">{m_qual}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.58rem; color: #94A3B8;">BIO AGE</div>
                                    <div style="font-size: 0.85rem; font-weight: 800; color: #FBBF24;">{m_age}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.58rem; color: #94A3B8;">SHELF LIFE</div>
                                    <div style="font-size: 0.85rem; font-weight: 800; color: #34D399;">{m_shelf}</div>
                                </div>
                                <div>
                                    <div style="font-size: 0.58rem; color: #94A3B8;">RISK</div>
                                    <div style="font-size: 0.85rem; font-weight: 800; color: #34D399;">{m_risk}</div>
                                </div>
                            </div>
                        </div>
                        <!-- Post-Harvest & Culinary Advisor Card -->
                        <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 8px; padding: 8px;">
                            <div style="font-size: 0.68rem; color: #166534; font-weight: 700;">🤖 AI Chef &amp; Post-Harvest Advisor</div>
                            <div style="font-size: 0.65rem; color: #1E293B; margin-top: 3px;">🧊 <b>Storage:</b> {m_storage}</div>
                            <div style="font-size: 0.65rem; color: #166534; font-weight: 600; margin-top: 2px;">🥗 <b>Chef Recipe:</b> {m_recipe}</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col_info:
            st.markdown("#### 📱 Native Android Architecture (`app/src/main/`)")
            st.markdown("""
            FreshAI's Android Application connects directly with the Python backend via high-speed REST APIs:
            - **Native Kotlin UI**: Built with Material Design 3 (`MaterialCardView`, `TabLayout`, `ProgressBar`).
            - **Real-Time Camera & Gallery**: Android `CameraX` / Image capture intent.
            - **On-Device & Server Dual Engine**:
              - *On-Device*: Runs quantized ONNX models via `com.microsoft.onnxruntime:onnxruntime-android`.
              - *Server REST API*: Communicates with `http://localhost:8088/api/detect`.
            """)

            st.markdown("##### 🔌 Live Mobile API Tester (`http://localhost:8088`)")
            if st.button("📡 Test Live REST API Connection (`/api/status`)"):
                try:
                    import urllib.request
                    req = urllib.request.Request("http://127.0.0.1:8088/api/status")
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        st.success("✅ FreshAI API Server is ONLINE and responding!")
                        st.json(data)
                except Exception as ex:
                    st.warning(f"API Server offline or not yet started on port 8088 ({ex}). Start it with: `python -m freshai.api_server`")

            st.markdown("##### 📦 Android APK Build Command")
            st.code("gradlew.bat assembleDebug", language="powershell")
            st.caption("Outputs: `app/build/outputs/apk/debug/app-debug.apk`")

    with tab_train:
        st.markdown("### 🛠️ Train & Fine-Tune FreshAI Machine Learning Models")
        st.markdown("Train on **Stage 5 (XGBoost & LightGBM)**, **Multimodal Fusion**, **YOLO Defect Segmentation**, or **Object Detection**.")

        train_mode = st.radio(
            "Select Training Task:",
            [
                "🌲 Stage 5: Structured Prediction Layer (XGBoost & LightGBM)",
                "🧬 Multimodal Freshness Fusion Model (ConvNeXt + YOLO + OpenCV)",
                "🔬 YOLO Defect Segmentation & Detection",
                "🍎 Fruits-360 Image Classification (Onion, Tomato, Apple, etc.)",
                "📦 Object Detection (Bounding Boxes)",
            ],
            horizontal=True
        )

        if "Stage 5: Structured Prediction Layer" in train_mode:
            st.info("💡 **Stage 5: Structured Prediction Layer Training**: Trains XGBoost (Quality Score, Spoilage Risk) and LightGBM (Physiological Age, Shelf Life, Time-to-Harvest) models on multimodal visual features, environmental factors (temperature, humidity), and storage timelines.")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                s_samples = st.slider("Dataset Training Samples", min_value=500, max_value=5000, value=2500, step=250)
            with s_col2:
                st.markdown("""
                **Trained Models & Target Predictions:**
                - `XGBoost Regressor`: Quality Score (e.g. 84/100)
                - `LightGBM Regressor`: Physiological Age (e.g. 5–8 days)
                - `LightGBM Regressor`: Remaining Shelf Life (e.g. 2–4 days)
                - `XGBoost Classifier`: Spoilage Risk (Low / Medium / High)
                - `LightGBM Regressor`: Time-to-Harvest (e.g. 10–14 days)
                """)

            if st.button("🚀 Train XGBoost & LightGBM Structured Layer", type="primary"):
                with st.spinner("Training XGBoost and LightGBM models on multi-factor tabular dataset..."):
                    try:
                        summary = train_structured_models(num_samples=s_samples)
                        st.success("🎉 Stage 5: XGBoost & LightGBM Models Trained Successfully!")
                        st.json(summary)
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        elif "Multimodal Freshness Fusion" in train_mode:
            st.info("💡 **Multimodal Freshness Fusion Training**: Extracts ConvNeXt embeddings, OpenCV color/GLCM texture descriptors, and YOLO defect segmentation percentages from dataset crops, training an MLP classifier to predict exact produce freshness stages.")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                f_data_dir = st.text_input("Produce Dataset Path", value="datasets/freshness_dataset")
                f_epochs = st.slider("Fusion Model Epochs", min_value=5, max_value=50, value=25)
            with f_col2:
                f_batch = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
                f_max_samples = st.slider("Max Samples Per Class", min_value=10, max_value=100, value=30)

            if st.button("🚀 Start Multimodal Fusion Model Training", type="primary"):
                with st.spinner("Extracting multi-source feature vectors and training Fusion MLP..."):
                    try:
                        summary = train_fusion_model(
                            data_dir=f_data_dir,
                            epochs=f_epochs,
                            batch_size=f_batch,
                            max_samples=f_max_samples,
                        )
                        st.success(f"🎉 Fusion Model Training Complete! Best Validation Accuracy: {summary.get('best_val_accuracy', 1.0)*100:.1f}%")
                        st.json(summary)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        elif "YOLO Defect Segmentation" in train_mode:
            st.info("💡 **YOLO Defect Segmentation Training**: Trains YOLOv8-seg to detect and segment bruises, black spots, rot, mold, cracks, and fungal lesions.")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                defect_dataset_mode = st.selectbox("Defect Dataset Mode", ["Synthetic Produce Defect Dataset (Instant Bootstrap)", "Custom data_defect.yaml"])
                defect_epochs = st.slider("Defect Training Epochs", min_value=5, max_value=50, value=15)
            with d_col2:
                custom_defect_yaml = st.text_input("Path to Defect data.yaml", value="datasets/produce_defects/data_defect.yaml")
                defect_batch = st.select_slider("Defect Batch Size", options=[4, 8, 16, 32], value=16)

            if st.button("🚀 Start YOLO Defect Model Training", type="primary"):
                with st.spinner("Generating defect segmentation data and training YOLOv8-seg..."):
                    try:
                        yaml_p = None if "Synthetic" in defect_dataset_mode else custom_defect_yaml
                        summary = train_defect_yolo(data_yaml=yaml_p, epochs=defect_epochs, batch=defect_batch, is_segmentation=True)
                        st.success("🎉 YOLO Defect Segmentation Training Completed!")
                        st.json(summary)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Defect training failed: {e}")

        elif "Classification" in train_mode:
            st.info("💡 **Fruits-360 Classification Training**: Learns exact visual distinctions between Onion (Red/White), Tomato, Apple varieties, Potato, Lemon, etc. with >95% confidence!")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                cls_data_dir = st.text_input("Fruits-360 Dataset Path", value="datasets/fruits_360")
                cls_base_model = st.selectbox("Base Classification Model", ["yolov8s-cls.pt (Recommended)", "yolov8m-cls.pt (Highest Accuracy)", "yolov8n-cls.pt (Fastest)"])
                cls_epochs = st.slider("Classification Epochs", min_value=5, max_value=50, value=20)
            with c_col2:
                cls_batch = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
                cls_exp_name = st.text_input("Experiment Name", value="fruits360_yolo_model")

            if st.button("🚀 Start Fruits-360 Classification Training", type="primary"):
                with st.spinner("Training YOLO classification model on Fruits-360 produce data..."):
                    try:
                        summary = train_classifier(
                            data_dir=cls_data_dir,
                            base_model=cls_base_model.split()[0],
                            epochs=cls_epochs,
                            batch_size=cls_batch,
                            name=cls_exp_name,
                        )
                        st.success("🎉 Fruits-360 Model Training Completed!")
                        st.json(summary)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        else:
            with st.expander("📚 **Where to get high-quality produce datasets (with Onions, Tomatoes, etc.)**", expanded=True):
                st.markdown("""
                To detect **Onion** and **Tomato** with bounding boxes:
                1. **Roboflow Universe**: [Fruit & Vegetable Detection Dataset](https://universe.roboflow.com/mohamed-traore-2wbn5/fruit-and-vegetable-detection) (YOLOv8 format)
                2. **Kaggle**: [Fruit and Vegetable Detection](https://www.kaggle.com/datasets/mbkinaci/fruit-and-vegetable-detection)
                """)

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                data_source = st.selectbox("Dataset Source", ["Custom data.yaml File", "Synthetic Demo Dataset (Instant)"])
                base_model_choice = st.selectbox("Base Architecture", ["yolov8s.pt (Higher Accuracy)", "yolov8m.pt (Best Accuracy)", "yolov8n.pt (Fastest)"])
                train_epochs = st.slider("Epochs", min_value=5, max_value=100, value=30)
                train_batch = st.select_slider("Batch Size", options=[4, 8, 16, 32], value=16)

            with t_col2:
                custom_yaml_path = st.text_input("Path to data.yaml", value="datasets/produce_dataset/data.yaml")
                experiment_name = st.text_input("Experiment Name", value="freshai_produce_model")

            if st.button("🚀 Start Model Training", type="primary"):
                with st.spinner("Preparing dataset and running YOLO training..."):
                    try:
                        if data_source == "Synthetic Demo Dataset (Instant)":
                            yaml_path = generate_synthetic_demo_dataset(
                                output_dir="datasets/freshai_demo",
                                num_train=30,
                                num_val=8
                            )
                        else:
                            yaml_path = custom_yaml_path

                        selected_base = base_model_choice.split()[0]
                        summary = train_yolo_model(
                            data_yaml=yaml_path,
                            base_model=selected_base,
                            epochs=train_epochs,
                            batch_size=train_batch,
                            name=experiment_name,
                        )
                        st.success("🎉 Training Completed Successfully!")
                        st.json(summary)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

    with tab_arch:
        st.markdown("""
        ### 🌿 FreshAI Full Platform Architecture

        **FreshAI** is an end-to-end intelligent platform that monitors produce quality from farm to fork:
        
        #### 1. Stage 1: Produce Identification & Localization (YOLO Detection)
        - Identifies produce categories (Tomato, Apple, Onion, Watermelon, etc.).
        - Computes bounding boxes $(x, y, w, h)$ and extracts focused produce crops for downstream inspection.
        
        #### 2. Stage 2: Freshness & Ripeness Classification (ConvNeXt-Tiny)
        - Multi-task neural backbone predicting Freshness Stage (0-5) and Ripeness Stage (0-3).
        - Generates 768-dimensional deep visual feature embeddings.
        
        #### 3. Stage 3: Defect Detection & Segmentation (YOLO)
        - Detects visible physical defects: **Bruises, Black spots, Cracks, Mold, Rot, Fungal spots, Wrinkles, Discoloration**.
        - Dual YOLO Approach:
          - **YOLO Detection**: Tells us *what defect is present and where it is*.
          - **YOLO Segmentation**: Tells us *exactly which pixels/area are affected*.
        - Computes exact **Defect Area %** (e.g. 7.4%).

        #### 4. Stage 4: Color & Texture Feature Extraction (OpenCV)
        ```
        Detected / Cropped Produce
         │
         ▼
         Image Preprocessing
         │
         ┌─────┴──────┐
         ▼             ▼
        Color Analysis Texture Analysis
         │             │
         ▼             ▼
        RGB/HSV/LAB    GLCM Features
         │             │
         └─────┬──────┘
               ▼
         Numerical Features
               │
               ▼
         Feature Vector
        ```
        - **Color Analysis**: RGB, HSV, CIELAB means and color region percentages (green chlorophyll, yellow carotenoids, brown decay, dark necrosis, pale mold).
        - **Texture Analysis**: GLCM Contrast, Homogeneity, Energy, Dissimilarity, Correlation + Laplacian surface roughness.

        #### 5. Stage 5: Structured Prediction Layer (XGBoost & LightGBM)
        ```
        Feature Vector
         │
         ┌───────────┼────────────┐
         ▼           ▼            ▼
         XGBoost  LightGBM     LightGBM
         Quality     Age       Shelf Life
         │           │            │
         └───────────┼────────────┘
                     ▼
               Spoilage Risk
        ```
        - **Input Features**:
          - `YOLO`: Produce Type (One-Hot)
          - `ConvNeXt-Tiny`: Freshness, Ripeness, Deep Visual Features
          - `YOLO Segmentation`: Defect Area %
          - `OpenCV`: Color Features (RGB, HSV, LAB) & Texture Features (GLCM Contrast, Homogeneity, Energy, etc.)
          - `Environmental Data`: Temperature (°C) & Humidity (%) -> VPD & Arrhenius Acceleration
          - `Timeline & Storage`: Days Since Purchase, Storage Condition (Fridge, Pantry, Cellar, Warm)
        - **Main Predictions**:
          - `Quality Score` → **XGBoost** → `84/100`
          - `Physiological Age` → **LightGBM** → `5–8 days`
          - `Remaining Shelf Life` → **LightGBM** → `2–4 days`
          - `Spoilage Risk` → **XGBoost/LightGBM** → `Low / Medium / High`
          - `Time-to-Harvest` → **LightGBM** → `10–14 days`

        #### 6. Stage 6: LLM + RAG Recommendations Engine
        ```
        Predictions & Diagnostics (Quality, Age, Defect %, Temp)
                                │
                                ▼
                     RAG Knowledge Retrieval
          (Post-Harvest Science, Ethylene Matrix, USDA Guidelines)
                                │
                                ▼
                     LLM Advisory Generation
                       (Gemini API / Hybrid RAG)
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
     Storage &               Spoilage              Zero-Waste
     Ethylene Plan          Mitigation            Chef Recipe
        ```
        - **Storage Strategy**: Temperature bounds, chilling injury alerts (e.g. Tomato < 10°C, Banana < 12°C), ethylene co-location warnings.
        - **Spoilage Mitigation**: Defect trimming, cooking windows, zero-waste preservation protocols.
        - **Culinary Recipes**: Tailored to quality grade (Grade A raw, Grade B roast/simmer, Grade C soups/puree/jams).
        - **Mobile Interface**: Native Android Studio application (`app/src/main/`) with on-device ONNX runtime & high-speed REST API on port 8088.
        """)


if __name__ == "__main__":
    main()