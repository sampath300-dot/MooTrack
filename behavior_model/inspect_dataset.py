import os
import json
import ast
import pandas as pd
from collections import Counter

# =========================================================
# CBVD-5 DATASET INSPECTION
# =========================================================

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"

CSV_PATH = os.path.join(DATASET_DIR, "CBVD-5.csv")

FRAME_DIR = os.path.join(
    DATASET_DIR,
    "labelframes",
    "labelframes"
)

# =========================================================
# Official label mapping from CSV
# =========================================================

LABEL_MAP = {
    "0": "Stand",
    "1": "Lying down",
    "2": "Foraging",
    "3": "Drinking water",
    "4": "Rumination",
}

print("=" * 70)
print("CBVD-5 DATASET INSPECTION")
print("=" * 70)

# =========================================================
# 1. Check paths
# =========================================================

print("\n[1] Checking dataset paths...")

print("Dataset directory:", DATASET_DIR)
print("CSV exists:", os.path.exists(CSV_PATH))
print("Frame directory exists:", os.path.exists(FRAME_DIR))

# =========================================================
# 2. Read raw CSV header information
# =========================================================

print("\n[2] Reading CSV header information...")

header_line_number = None
column_names = None

with open(
    CSV_PATH,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    for line_number, line in enumerate(f):

        if line.startswith("# CSV_HEADER"):

            header_line_number = line_number

            header_text = line.split("=", 1)[1].strip()

            column_names = [
                x.strip()
                for x in header_text.split(",")
            ]

            print("Header found at line:", line_number + 1)
            print("Columns:", column_names)

            break

if column_names is None:

    print("ERROR: CSV header not found.")
    raise SystemExit

# =========================================================
# 3. Load actual data
# =========================================================

print("\n[3] Loading annotation data...")

# header_line_number is zero-indexed.
# Skip all lines through the comment/header line.
df = pd.read_csv(
    CSV_PATH,
    skiprows=header_line_number + 1,
    header=None,
    names=column_names
)

print("Number of annotation rows:", len(df))
print("Number of columns:", len(df.columns))

# =========================================================
# 4. Display columns
# =========================================================

print("\n[4] Columns:")

for column in df.columns:
    print(" -", column)

# =========================================================
# 5. Display first rows
# =========================================================

print("\n[5] First 5 annotation rows:")

print(
    df.head().to_string(index=False)
)

# =========================================================
# 6. Check labelmap
# =========================================================

print("\n[6] Behaviour label mapping:")

for label_id, behaviour in LABEL_MAP.items():

    print(
        f" - {label_id} -> {behaviour}"
    )

# =========================================================
# 7. Extract behaviour IDs from metadata
# =========================================================

print("\n[7] Extracting behaviour labels...")

label_counter = Counter()

invalid_labels = Counter()

rows_with_multiple_labels = 0

for metadata in df["metadata"]:

    try:

        # Example:
        # {"1":"0,4"}

        data = json.loads(metadata)

        value = data.get("1", "")

        labels = [
            x.strip()
            for x in str(value).split(",")
            if x.strip() != ""
        ]

        if len(labels) > 1:
            rows_with_multiple_labels += 1

        for label in labels:

            if label in LABEL_MAP:

                label_counter[label] += 1

            else:

                invalid_labels[label] += 1

    except Exception:

        invalid_labels["PARSE_ERROR"] += 1

# =========================================================
# 8. Label distribution
# =========================================================

print("\n[8] Behaviour label distribution:")

for label_id in sorted(
    LABEL_MAP.keys(),
    key=lambda x: int(x)
):

    count = label_counter[label_id]

    print(
        f" - {label_id} ({LABEL_MAP[label_id]}): {count}"
    )

# =========================================================
# 9. Invalid / unknown labels
# =========================================================

print("\n[9] Unknown / invalid labels:")

if invalid_labels:

    for label, count in invalid_labels.items():

        print(
            f" - {label}: {count}"
        )

else:

    print(" - None")

# =========================================================
# 10. Multiple behaviour annotations
# =========================================================

print("\n[10] Multiple-label annotations:")

print(
    "Rows containing more than one behaviour label:",
    rows_with_multiple_labels
)

# =========================================================
# 11. Check frame files
# =========================================================

print("\n[11] Checking labelframes...")

if os.path.exists(FRAME_DIR):

    frame_files = [
        f
        for f in os.listdir(FRAME_DIR)
        if os.path.isfile(
            os.path.join(FRAME_DIR, f)
        )
    ]

    print(
        "Number of frame files:",
        len(frame_files)
    )

    print("\nFirst 20 frame files:")

    for filename in frame_files[:20]:

        print(
            " -",
            filename
        )

else:

    print("Frame directory not found!")

# =========================================================
# 12. Unique frames in annotations
# =========================================================

print("\n[12] Checking annotated frame count...")

def extract_filename(value):

    try:

        # Example:
        # ["618_00002.jpg"]

        value = ast.literal_eval(value)

        if isinstance(value, list) and len(value) > 0:

            return value[0]

    except Exception:

        pass

    return None


annotated_frames = set()

for value in df["file_list"]:

    filename = extract_filename(value)

    if filename:

        annotated_frames.add(filename)

print(
    "Unique frames referenced by annotations:",
    len(annotated_frames)
)

# =========================================================
# 13. Check annotation frames against actual files
# =========================================================

print("\n[13] Checking annotation → image mapping...")

existing_frames = set(frame_files)

matched_frames = annotated_frames.intersection(
    existing_frames
)

missing_frames = annotated_frames - existing_frames

print(
    "Annotated frames found in labelframes:",
    len(matched_frames)
)

print(
    "Annotated frames missing from labelframes:",
    len(missing_frames)
)

if missing_frames:

    print("\nFirst 20 missing frames:")

    for filename in list(missing_frames)[:20]:

        print(
            " -",
            filename
        )

# =========================================================
# 14. Summary
# =========================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    "Annotation rows:",
    len(df)
)

print(
    "Unique annotated frames:",
    len(annotated_frames)
)

print(
    "Actual labelframes:",
    len(existing_frames)
)

print(
    "Matched frames:",
    len(matched_frames)
)

print(
    "Rows with multiple behaviour labels:",
    rows_with_multiple_labels
)

print("\nBehaviour classes:")

for label_id, behaviour in LABEL_MAP.items():

    print(
        f"{label_id} = {behaviour}: "
        f"{label_counter[label_id]} annotations"
    )

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)