import os
import pandas as pd
from PIL import Image, ImageDraw

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"

FRAME_DIR = os.path.join(
    DATASET_DIR,
    "labelframes",
    "labelframes"
)

ANNOTATION_FILE = os.path.join(
    DATASET_DIR,
    "processed_behavior",
    "train_annotations.csv"
)

OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "crop_test"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


LABELS = {
    1: "stand",
    2: "lying_down",
    3: "foraging",
    4: "drinking_water",
    5: "rumination"
}


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

df = pd.read_csv(ANNOTATION_FILE)

print("=" * 70)
print("CATTLE CROP TEST")
print("=" * 70)

print(f"\nAnnotations available: {len(df):,}")


# ============================================================
# PROCESS FIRST 10 ANNOTATIONS
# ============================================================

for index, row in df.head(10).iterrows():

    video_id = int(row["video_id"])
    timestamp = int(row["timestamp"])
    action_id = int(row["action_id"])

    frame_name = f"{video_id}_{timestamp:05d}.jpg"

    frame_path = os.path.join(
        FRAME_DIR,
        frame_name
    )

    if not os.path.exists(frame_path):
        print(f"Missing frame: {frame_name}")
        continue

    image = Image.open(frame_path).convert("RGB")

    width, height = image.size

    # Normalized coordinates → pixels

    left = int(row["x1"] * width)
    top = int(row["y1"] * height)

    right = int(row["x2"] * width)
    bottom = int(row["y2"] * height)

    # Clamp coordinates

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))

    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    # --------------------------------------------------------
    # Create crop
    # --------------------------------------------------------

    crop = image.crop(
        (left, top, right, bottom)
    )

    label = LABELS.get(
        action_id,
        "unknown"
    )

    crop_filename = (
        f"crop_{index:03d}_"
        f"{video_id}_"
        f"{timestamp:05d}_"
        f"{label}.jpg"
    )

    crop_path = os.path.join(
        OUTPUT_DIR,
        crop_filename
    )

    crop.save(
        crop_path,
        "JPEG",
        quality=95
    )

    # --------------------------------------------------------
    # Create image with bounding box
    # --------------------------------------------------------

    marked = image.copy()

    draw = ImageDraw.Draw(marked)

    draw.rectangle(
        (left, top, right, bottom),
        outline="red",
        width=5
    )

    marked_filename = (
        f"marked_{index:03d}_"
        f"{video_id}_"
        f"{timestamp:05d}_"
        f"{label}.jpg"
    )

    marked_path = os.path.join(
        OUTPUT_DIR,
        marked_filename
    )

    marked.save(
        marked_path,
        "JPEG",
        quality=95
    )

    print(
        f"{index + 1:2}. "
        f"{frame_name:20} | "
        f"{label:20} | "
        f"Box: "
        f"({left},{top})-({right},{bottom})"
    )


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print("\nOpen this folder and check the generated images.")