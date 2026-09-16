import os
import pandas as pd

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"

FRAME_DIR = os.path.join(
    DATASET_DIR,
    "labelframes",
    "labelframes"
)

ANNOTATION_DIR = os.path.join(
    DATASET_DIR,
    "annotations"
)

ANNOTATION_FILE = os.path.join(
    ANNOTATION_DIR,
    "ava_train_v2.1.csv"
)

print("=" * 70)
print("FRAME MAPPING CHECK")
print("=" * 70)

print("\nFrame directory:")
print(FRAME_DIR)

print("\nAnnotation file:")
print(ANNOTATION_FILE)

# Load annotations
df = pd.read_csv(
    ANNOTATION_FILE,
    header=None
)

print("\nTotal annotations:", len(df))

print("\nChecking first 20 annotations...")
print("-" * 70)

found = 0
missing = 0

for i in range(min(20, len(df))):

    video_id = int(df.iloc[i, 0])
    timestamp = int(df.iloc[i, 1])

    filename = f"{video_id}_{timestamp:05d}.jpg"

    path = os.path.join(
        FRAME_DIR,
        filename
    )

    exists = os.path.exists(path)

    status = "FOUND" if exists else "MISSING"

    print(
        f"{video_id:4d} | "
        f"{timestamp:5d} | "
        f"{filename:20} | "
        f"{status}"
    )

    if exists:
        found += 1
    else:
        missing += 1

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("Found :", found)
print("Missing:", missing)

print("\nFrame files in directory:")

files = [
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith(".jpg")
]

print("Total JPG files:", len(files))

print("\nFirst 20 JPG files:")

for filename in sorted(files)[:20]:
    print(filename)

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)