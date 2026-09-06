"""Create a clean Onion/Tomato/Watermelon YOLO dataset from the LVIS download."""
from pathlib import Path
import shutil
import yaml

SOURCE = Path("datasets/LVIS_Fruits_And_Vegetables")
TARGET = Path("datasets/freshai_produce_focus")
# The source has two tomato aliases. Both become the one Tomato class.
CLASS_MAP = {43: 0, 35: 1, 59: 1, 61: 2}
NAMES = ["Onion", "Tomato", "Watermelon"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def label_for(image: Path) -> Path:
    relative = image.relative_to(SOURCE / "images")
    return SOURCE / "labels" / relative.with_suffix(".txt")


def prepare_split(split: str) -> int:
    image_out = TARGET / "images" / split
    label_out = TARGET / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in (SOURCE / "images" / split).rglob("*"):
        if not image.is_file() or image.suffix.lower() not in EXTENSIONS:
            continue
        label = label_for(image)
        if not label.exists():
            continue
        remapped = []
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            class_id = int(values[0])
            if class_id in CLASS_MAP:
                remapped.append(" ".join([str(CLASS_MAP[class_id]), *values[1:]]))
        if not remapped:
            continue
        # The original COCO image IDs are unique, so flattening nested ZIP
        # folders is safe and yields the standard YOLO directory layout.
        destination_image = image_out / image.name
        shutil.copy2(image, destination_image)
        (label_out / f"{image.stem}.txt").write_text("\n".join(remapped) + "\n")
        count += 1
    return count


if __name__ == "__main__":
    train_count = prepare_split("train")
    val_count = prepare_split("val")
    (TARGET / "data.yaml").write_text(yaml.safe_dump({
        "path": str(TARGET.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": NAMES,
    }, sort_keys=False))
    print(f"Prepared {train_count} training and {val_count} validation images in {TARGET}")
