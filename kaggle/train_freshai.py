"""Run this file as a Kaggle Notebook script with a YOLO-format dataset attached.

Outputs (in /kaggle/working): freshai_best.pt, freshai_yolo.onnx and
freshai_classes.txt. Download all three after training; the Android app uses
the ONNX model and class-label file offline.
"""
from pathlib import Path
import shutil
import yaml
from ultralytics import YOLO

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
EPOCHS = 50
IMGSZ = 640
BATCH = -1  # Automatically fit the available Kaggle GPU.


def locate_yaml() -> Path:
    yamls = list(INPUT_ROOT.rglob("data.yaml"))
    if not yamls:
        raise FileNotFoundError("Attach a YOLO dataset containing data.yaml to this Kaggle Notebook.")
    return yamls[0]


def prepare_yaml(source: Path) -> Path:
    """Replace author-machine paths in a Kaggle data.yaml with mounted paths."""
    config = yaml.safe_load(source.read_text())
    root = source.parent
    config["path"] = str(root)
    for split in ("train", "val", "test"):
        if split in config:
            value = Path(str(config[split]))
            if value.is_absolute() or not (root / value).exists():
                candidate = root / "images" / split
                if candidate.exists():
                    config[split] = str(candidate)
            else:
                config[split] = str(root / value)
    destination = WORK_ROOT / "freshai_data.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


data_yaml = prepare_yaml(locate_yaml())
data = yaml.safe_load(data_yaml.read_text())
names = data["names"]
if isinstance(names, dict):
    names = [names[index] for index in sorted(names)]

print("Training classes:", names)
model = YOLO("yolov8n.pt")
model.train(
    data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
    device=0, workers=2, patience=12, project=str(WORK_ROOT),
    name="freshai_produce", exist_ok=True,
)

best = WORK_ROOT / "freshai_produce" / "weights" / "best.pt"
shutil.copy2(best, WORK_ROOT / "freshai_best.pt")
exported = Path(YOLO(str(best)).export(format="onnx", imgsz=IMGSZ, simplify=True))
shutil.copy2(exported, WORK_ROOT / "freshai_yolo.onnx")
(WORK_ROOT / "freshai_classes.txt").write_text("\n".join(names) + "\n")
print("Done. Download freshai_yolo.onnx and freshai_classes.txt from the Output tab.")
