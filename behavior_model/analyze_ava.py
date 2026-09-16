import os
import pandas as pd

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"
ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations")

FILES = {
    "train": "ava_train_v2.1.csv",
    "val": "ava_val_v2.1.csv",
    "test": "ava_test_v2.1.csv",
}

LABELS = {
    1: "Stand",
    2: "Lying down",
    3: "Foraging",
    4: "Drinking water",
    5: "Rumination",
}

print("=" * 70)
print("CBVD-5 AVA ANNOTATION ANALYSIS")
print("=" * 70)

for split, filename in FILES.items():

    path = os.path.join(ANNOTATION_DIR, filename)

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATA")
    print("=" * 70)

    df = pd.read_csv(path, header=None)

    print(f"File: {filename}")
    print(f"Total annotation rows: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")

    print("\nBehaviour distribution:")

    counts = df[6].value_counts().sort_index()

    for action_id, count in counts.items():
        try:
            action_id = int(action_id)
            name = LABELS.get(action_id, "Unknown")
        except:
            name = "Unknown"

        print(f"{action_id}: {name:20} {count:,}")

    print("\nUnique video IDs:")
    print(df[0].nunique())

    print("Video ID range:")
    print(df[0].min(), "to", df[0].max())

    print("\nUnique timestamps:")
    print(df[[0, 1]].drop_duplicates().shape[0])

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)