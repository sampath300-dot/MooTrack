import os

from behavior_model.config import DATASET_DIR
LABELMAP_PATH = os.path.join(
    DATASET_DIR,
    "annotations",
    "labelmap.txt"
)

print("=" * 70)
print("CBVD-5 LABEL MAP")
print("=" * 70)

print("\nFile:")
print(LABELMAP_PATH)

print("\nContents:")
print("-" * 70)

with open(LABELMAP_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line_number, line in enumerate(f, start=1):
        print(f"{line_number}: {line.rstrip()}")

print("-" * 70)