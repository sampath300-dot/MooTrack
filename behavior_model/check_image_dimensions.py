import os
from PIL import Image

from behavior_model.config import DATASET_DIR

FRAME_DIR = os.path.join(
    DATASET_DIR,
    "labelframes",
    "labelframes"
)

print("=" * 70)
print("CBVD-5 IMAGE DIMENSION CHECK")
print("=" * 70)

files = sorted([
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith(".jpg")
])

print(f"\nTotal JPG files: {len(files)}")

print("\nChecking first 10 images:")
print("-" * 70)

dimensions = {}

for filename in files[:10]:

    path = os.path.join(FRAME_DIR, filename)

    with Image.open(path) as img:
        width, height = img.size

    dimensions.setdefault((width, height), 0)
    dimensions[(width, height)] += 1

    print(
        f"{filename:20} "
        f"Width: {width:5} "
        f"Height: {height:5}"
    )

print("\n" + "=" * 70)
print("CHECKING ALL IMAGE DIMENSIONS")
print("=" * 70)

for filename in files:

    path = os.path.join(FRAME_DIR, filename)

    try:
        with Image.open(path) as img:
            size = img.size

        dimensions.setdefault(size, 0)
        dimensions[size] += 1

    except Exception as e:
        print(f"Error reading {filename}: {e}")

print("\nUnique image dimensions:")

for size, count in sorted(dimensions.items()):
    print(
        f"{size[0]} x {size[1]} : {count:,} images"
    )

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)