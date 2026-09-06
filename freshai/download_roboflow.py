"""
FreshAI - Automated Dataset Downloader for Roboflow Universe
Allows downloading large fruit and vegetable datasets directly with a single command.
"""

import os
import sys
import argparse

# Curated High-Quality Large Roboflow Produce Datasets (with Onions, Tomatoes, Apples, etc.)
CURATED_DATASETS = {
    "fruits-and-vegetables-large": {
        "workspace": "mohamed-traore-2wbn5",
        "project": "fruit-and-vegetable-detection",
        "version": 1,
        "description": "Comprehensive fruit & vegetable dataset with Onions, Tomatoes, Apples, Potatoes, Bananas, etc.",
        "url": "https://universe.roboflow.com/mohamed-traore-2wbn5/fruit-and-vegetable-detection",
    },
    "supermarket-produce-100": {
        "workspace": "roboflow-100",
        "project": "supermarket-produce",
        "version": 2,
        "description": "Supermarket produce dataset with real packaging, lighting, and diverse produce categories.",
        "url": "https://universe.roboflow.com/roboflow-100/supermarket-produce",
    },
    "fresh-produce-detection": {
        "workspace": "fresh-produce",
        "project": "fruits-vegetables",
        "version": 1,
        "description": "Multi-class fruits and vegetables dataset for grocery detection.",
        "url": "https://universe.roboflow.com/search?q=fruits+and+vegetables",
    }
}


def download_dataset(
    api_key: str,
    dataset_key: str = "fruits-and-vegetables-large",
    workspace: str = None,
    project_name: str = None,
    version_num: int = 1,
    output_dir: str = "datasets/roboflow_produce",
) -> str:
    """
    Download a YOLOv8 formatted dataset from Roboflow Universe using the Roboflow SDK.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Roboflow package is not installed. Please run: pip install roboflow")

    if dataset_key in CURATED_DATASETS and not (workspace and project_name):
        meta = CURATED_DATASETS[dataset_key]
        workspace = meta["workspace"]
        project_name = meta["project"]
        version_num = meta["version"]

    print("============================================================")
    print(f"  FreshAI — Roboflow Dataset Downloader")
    print(f"  Workspace: {workspace}")
    print(f"  Project:   {project_name} (v{version_num})")
    print(f"  Target:    {output_dir}")
    print("============================================================")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    
    os.makedirs(output_dir, exist_ok=True)
    dataset = version.download("yolov8", location=output_dir)

    yaml_path = os.path.join(output_dir, "data.yaml")
    print("\n✅ Dataset downloaded successfully!")
    print(f"📁 data.yaml location: {yaml_path}")
    print("\nYou can now start training with:")
    print(f".venv\\Scripts\\python freshai/train_yolo.py --data {yaml_path} --model yolov8s.pt --epochs 40 --batch 16")
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Download large produce datasets from Roboflow Universe")
    parser.add_argument("--api-key", type=str, required=True, help="Your Roboflow API Key (free from app.roboflow.com)")
    parser.add_argument("--preset", type=str, default="fruits-and-vegetables-large", choices=list(CURATED_DATASETS.keys()), help="Curated dataset preset")
    parser.add_argument("--workspace", type=str, default=None, help="Custom Roboflow workspace name")
    parser.add_argument("--project", type=str, default=None, help="Custom Roboflow project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    parser.add_argument("--dest", type=str, default="datasets/roboflow_produce", help="Destination folder")

    args = parser.parse_args()

    download_dataset(
        api_key=args.api_key,
        dataset_key=args.preset,
        workspace=args.workspace,
        project_name=args.project,
        version_num=args.version,
        output_dir=args.dest,
    )


if __name__ == "__main__":
    main()
