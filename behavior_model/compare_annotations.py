import os
import pandas as pd


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


print("=" * 70)
print("CBVD-5 vs AVA ANNOTATION COMPARISON")
print("=" * 70)


# ============================================================
# READ CBVD-5 CSV
# ============================================================

print("\nReading CBVD-5.csv...")

with open(
    CBVD_FILE,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    lines = f.readlines()


# Find CSV header
header_line = None

for line in lines:

    if line.startswith("# CSV_HEADER"):

        header_line = line.split("=", 1)[1].strip()

        break


print("\nCBVD header:")
print(header_line)


# Remove comment lines
data_lines = [
    line
    for line in lines
    if not line.startswith("#")
]

from io import StringIO

cbvd = pd.read_csv(
    StringIO("".join(
        [header_line + "\n"] + data_lines
    ))
)

print(
    f"\nCBVD annotation rows: {len(cbvd):,}"
)

print("\nCBVD columns:")

for column in cbvd.columns:
    print(" -", column)


print("\nFirst 10 CBVD annotations:")

print(
    cbvd.head(10).to_string(
        index=False
    )
)


# ============================================================
# READ AVA
# ============================================================

print("\n" + "=" * 70)
print("AVA ANNOTATIONS")
print("=" * 70)

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

print(
    f"\nAVA annotation rows: {len(ava):,}"
)

print("\nFirst 20 AVA annotations:")

print(
    ava.head(20).to_string(
        index=False
    )
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("AVA ACTION DISTRIBUTION")
print("=" * 70)

print(
    ava["action_id"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAMPLE FILE REFERENCES
# ============================================================

print("\n" + "=" * 70)
print("CBVD FILE REFERENCES")
print("=" * 70)

if "file_list" in cbvd.columns:

    for value in cbvd["file_list"].head(10):

        print(value)


# ============================================================
# SAMPLE METADATA
# ============================================================

print("\n" + "=" * 70)
print("CBVD METADATA EXAMPLES")
print("=" * 70)

if "metadata" in cbvd.columns:

    for value in cbvd["metadata"].head(20):

        print(value)


print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)