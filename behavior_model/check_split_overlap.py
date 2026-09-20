import os
import pandas as pd

from behavior_model.config import DATASET_DIR
ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations")

FILES = {
    "train": "ava_train_v2.1.csv",
    "val": "ava_val_v2.1.csv",
    "test": "ava_test_v2.1.csv",
}

data = {}

print("=" * 70)
print("TRAIN / VALIDATION / TEST OVERLAP CHECK")
print("=" * 70)

# Load files
for split, filename in FILES.items():

    path = os.path.join(ANNOTATION_DIR, filename)

    df = pd.read_csv(path, header=None)

    # Column 0 = video ID
    # Column 1 = timestamp

    data[split] = df

    videos = set(df[0].unique())

    timestamps = set(
        zip(df[0], df[1])
    )

    print(f"\n{split.upper()}")
    print("-" * 70)
    print(f"Annotation rows : {len(df):,}")
    print(f"Unique videos   : {len(videos):,}")
    print(f"Unique timestamps: {len(timestamps):,}")


# --------------------------------------------------
# Video-level overlap
# --------------------------------------------------

train_videos = set(data["train"][0].unique())
val_videos = set(data["val"][0].unique())
test_videos = set(data["test"][0].unique())

print("\n" + "=" * 70)
print("VIDEO-LEVEL OVERLAP")
print("=" * 70)

print(
    "Train ∩ Validation:",
    len(train_videos & val_videos)
)

print(
    "Train ∩ Test:",
    len(train_videos & test_videos)
)

print(
    "Validation ∩ Test:",
    len(val_videos & test_videos)
)


# --------------------------------------------------
# Timestamp-level overlap
# --------------------------------------------------

train_ts = set(zip(data["train"][0], data["train"][1]))
val_ts = set(zip(data["val"][0], data["val"][1]))
test_ts = set(zip(data["test"][0], data["test"][1]))

print("\n" + "=" * 70)
print("VIDEO + TIMESTAMP OVERLAP")
print("=" * 70)

print(
    "Train ∩ Validation:",
    len(train_ts & val_ts)
)

print(
    "Train ∩ Test:",
    len(train_ts & test_ts)
)

print(
    "Validation ∩ Test:",
    len(val_ts & test_ts)
)


# --------------------------------------------------
# Show overlapping video IDs
# --------------------------------------------------

print("\n" + "=" * 70)
print("OVERLAPPING VIDEO IDs")
print("=" * 70)

val_test_overlap = sorted(val_videos & test_videos)

print("\nValidation ∩ Test video IDs:")

print(val_test_overlap)

print("\nNumber of overlapping videos:", len(val_test_overlap))

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)