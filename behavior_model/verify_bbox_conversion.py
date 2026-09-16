import os
import ast
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"

CBVD_FILE = os.path.join(
    DATASET_DIR,
    "CBVD-5.csv"
)

AVA_FILE = os.path.join(
    DATASET_DIR,
    "annotations",
    "ava_train_v2.1.csv"
)


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


# ============================================================
# LOAD CBVD CSV
# ============================================================

print("=" * 70)
print("VERIFYING CBVD → AVA BOUNDING BOX CONVERSION")
print("=" * 70)

with open(
    CBVD_FILE,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    lines = f.readlines()


header = None

for line in lines:

    if line.startswith("# CSV_HEADER"):

        header = line.split(
            "=",
            1
        )[1].strip()

        break


data_lines = [
    line
    for line in lines
    if not line.startswith("#")
]

from io import StringIO

cbvd = pd.read_csv(
    StringIO(
        header + "\n" + "".join(data_lines)
    )
)


# ============================================================
# LOAD AVA
# ============================================================

ava = pd.read_csv(
    AVA_FILE,
    header=None
)

ava.columns = [
    "video_id",
    "timestamp",
    "x1",
    "y1",
    "x2",
    "y2",
    "action_id",
    "person_id"
]


# ============================================================
# CHECK FIRST 10 CBVD ANNOTATIONS
# ============================================================

print("\nChecking first 10 bounding boxes...")
print("-" * 70)

for i in range(10):

    row = cbvd.iloc[i]

    file_list = ast.literal_eval(
        row["file_list"]
    )

    filename = file_list[0]

    # Extract video ID and timestamp
    name = filename.replace(
        ".jpg",
        ""
    )

    video_id, timestamp = name.split(
        "_"
    )

    video_id = int(video_id)
    timestamp = int(timestamp)

    # CBVD rectangle:
    # [shape_id, x, y, width, height]

    box = ast.literal_eval(
        row["spatial_coordinates"]
    )

    _, x, y, w, h = box

    expected_x1 = x / IMAGE_WIDTH
    expected_y1 = y / IMAGE_HEIGHT

    expected_x2 = (
        x + w
    ) / IMAGE_WIDTH

    expected_y2 = (
        y + h
    ) / IMAGE_HEIGHT

    # Find corresponding AVA rows
    matches = ava[
        (ava["video_id"] == video_id)
        &
        (ava["timestamp"] == timestamp)
    ]

    # Find closest matching bounding box
    best_match = None
    best_error = float("inf")

    for _, arow in matches.iterrows():

        error = (
            abs(arow["x1"] - expected_x1)
            + abs(arow["y1"] - expected_y1)
            + abs(arow["x2"] - expected_x2)
            + abs(arow["y2"] - expected_y2)
        )

        if error < best_error:

            best_error = error
            best_match = arow

    print(f"\n{i + 1}. {filename}")

    print(
        f"CBVD pixel box: "
        f"({x:.3f}, {y:.3f}, "
        f"{w:.3f}, {h:.3f})"
    )

    print(
        f"Expected AVA: "
        f"({expected_x1:.6f}, "
        f"{expected_y1:.6f}, "
        f"{expected_x2:.6f}, "
        f"{expected_y2:.6f})"
    )

    if best_match is not None:

        print(
            f"Actual AVA:   "
            f"({best_match['x1']:.6f}, "
            f"{best_match['y1']:.6f}, "
            f"{best_match['x2']:.6f}, "
            f"{best_match['y2']:.6f})"
        )

        print(
            f"Difference: {best_error:.10f}"
        )

        print(
            f"Action ID: {best_match['action_id']}"
        )

    else:

        print("No AVA match found.")


print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)