import os
import sys
from pathlib import Path
from collections import Counter
from PIL import Image

from behavior_model.config import DATASET_DIR, BEHAVIOR_CLASSES, TEST_SAMPLES_DIR


def inspect_dataset(dataset_dir: Path = DATASET_DIR):
    print("=" * 65)
    print("MOOTRACK - CBVD-5 CATTLE BEHAVIOR DATASET INSPECTOR")
    print("=" * 65)

    if not dataset_dir.exists():
        print(f"[!] Dataset directory not found at: {dataset_dir.resolve()}")
        print("    Checking test samples directory instead...")
        dataset_dir = TEST_SAMPLES_DIR

    print(f"Inspecting directory: {dataset_dir.resolve()}\n")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_counts = Counter()
    image_shapes = []

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            cls_name = subdir.name.lower()
            files = [f for f in subdir.iterdir() if f.suffix.lower() in valid_extensions]
            class_counts[cls_name] = len(files)
            for f in files[:5]:
                try:
                    with Image.open(f) as img:
                        image_shapes.append(img.size)
                except Exception:
                    pass
    else:
        files = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        for f in files:
            for cls in BEHAVIOR_CLASSES:
                if cls in f.name.lower():
                    class_counts[cls] += 1
                    try:
                        with Image.open(f) as img:
                            image_shapes.append(img.size)
                    except Exception:
                        pass
                    break

    print("Class Distribution:")
    total = sum(class_counts.values())
    for cls in BEHAVIOR_CLASSES:
        count = class_counts.get(cls, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  - {cls.capitalize():<12}: {count:>4} samples ({pct:>5.1f}%)")
    print(f"  -----------------------------------")
    print(f"  Total Samples : {total:>4} samples\n")

    if image_shapes:
        avg_w = sum(s[0] for s in image_shapes) // len(image_shapes)
        avg_h = sum(s[1] for s in image_shapes) // len(image_shapes)
        print(f"Sample Dimensions: ~{avg_w} x {avg_h} px (Resized to 224x224 for ResNet18)")
    
    print("=" * 65)


if __name__ == "__main__":
    inspect_dataset()
