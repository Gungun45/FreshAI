"""
FreshAI - Dataset Download & Preparation Helper for Real Produce
Classes: Onion, Tomato, Apple, Potato, Banana, Bell Pepper, Carrot, Cucumber, etc.
"""

import os

ROBOFLOW_GUIDE = """
========================================================================
🥗 HOW TO GET HIGH-ACCURACY PRODUCE DATASETS WITH ONIONS & TOMATOES:
========================================================================

Option 1: Roboflow Universe (Recommended — Instant YOLOv8 format)
-----------------------------------------------------------------
1. Visit Roboflow Universe:
   👉 https://universe.roboflow.com/search?q=fruits+and+vegetables
   or specific produce datasets:
   👉 https://universe.roboflow.com/mohamed-traore-2wbn5/fruit-and-vegetable-detection
   👉 https://universe.roboflow.com/roboflow-100/supermarket-produce

2. Click 'Download Dataset' -> Select Format 'YOLOv8' -> Click 'Continue'.

3. You will get a Python snippet or zip download link:
   --------------------------------------------------------------
   pip install roboflow
   --------------------------------------------------------------
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_FREE_API_KEY")
   project = rf.workspace("workspace-name").project("project-name")
   version = project.version(1)
   dataset = version.download("yolov8", location="datasets/produce_dataset")
   --------------------------------------------------------------

4. After downloading, the folder will have:
   datasets/produce_dataset/
     ├── data.yaml
     ├── train/ (images & labels)
     ├── valid/ (images & labels)
     └── test/

Option 2: Kaggle Fruits & Vegetables Object Detection
-----------------------------------------------------------------
1. Visit Kaggle:
   👉 https://www.kaggle.com/datasets/mbkinaci/fruit-and-vegetable-detection
   👉 https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
2. Download and place the extracted images & labels inside `datasets/produce_data/`.

Option 3: Train with FreshAI Training Script:
-----------------------------------------------------------------
.venv\\Scripts\\python freshai/train_yolo.py --data datasets/produce_dataset/data.yaml --model yolov8s.pt --epochs 40 --batch 16
========================================================================
"""

def print_dataset_instructions():
    print(ROBOFLOW_GUIDE)


if __name__ == "__main__":
    print_dataset_instructions()
