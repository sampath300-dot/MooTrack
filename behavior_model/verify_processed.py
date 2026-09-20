import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

from behavior_model.config import DATASET_DIR

CROP_DIR = os.path.join(
    DATASET_DIR,
    "processed_behavior",
    "crops"
)

LABELS = {
    0: "Stand",
    1: "Lying down",
    2: "Foraging",
    3: "Drinking water",
    4: "Rumination"
}

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# VERIFY EACH SPLIT
# ============================================================

print("=" * 70)
print("PROCESSED DATASET VERIFICATION")
print("=" * 70)

total_crops = 0


for split in SPLITS:

    print("\n" + "=" * 70)
    print(f"{split.upper()} DATASET")
    print("=" * 70)

    metadata_path = os.path.join(
        CROP_DIR,
        split,
        "metadata.csv"
    )

    if not os.path.exists(metadata_path):

        print(
            f"ERROR: Metadata not found:\n"
            f"{metadata_path}"
        )

        continue

    df = pd.read_csv(metadata_path)

    print(
        f"Metadata records: {len(df):,}"
    )

    # --------------------------------------------------------
    # Count labels
    # --------------------------------------------------------

    label_counts = {
        label_id: 0
        for label_id in LABELS
    }

    multi_label_count = 0

    for value in df["label_vector"]:

        values = [
            int(x)
            for x in str(value).split(",")
        ]

        active_labels = sum(values)

        if active_labels > 1:
            multi_label_count += 1

        for label_id, active in enumerate(values):

            if active == 1:
                label_counts[label_id] += 1

    print("\nBehaviour label counts:")

    for label_id, name in LABELS.items():

        print(
            f"{label_id}: "
            f"{name:20} "
            f"{label_counts[label_id]:,}"
        )

    print(
        f"\nMulti-label samples: "
        f"{multi_label_count:,}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    missing_files = 0

    for image_path in df["image_path"]:

        if not os.path.exists(image_path):

            missing_files += 1

    print(
        f"Missing crop files: "
        f"{missing_files:,}"
    )

    total_crops += len(df)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Total processed crops: "
    f"{total_crops:,}"
)

print("\nExpected:")
print("25,324 crops")

if total_crops == 25324:

    print(
        "\n✓ Total crop count is correct."
    )

else:

    print(
        "\nWARNING: Crop count differs "
        "from expected value."
    )

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)