import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"

PROCESSED_DIR = os.path.join(
    DATASET_DIR,
    "processed_behavior"
)

FILES = {
    "train": "train_annotations.csv",
    "val": "val_annotations.csv",
    "test": "test_annotations.csv"
}


LABELS = {
    1: "Stand",
    2: "Lying down",
    3: "Foraging",
    4: "Drinking water",
    5: "Rumination"
}


# ============================================================
# ANALYZE
# ============================================================

for split, filename in FILES.items():

    path = os.path.join(
        PROCESSED_DIR,
        filename
    )

    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print(f"{split.upper()} MULTI-LABEL ANALYSIS")
    print("=" * 70)

    print(
        f"Total annotation rows: {len(df):,}"
    )

    # Same video + timestamp + bounding box
    # represents one cattle annotation.
    #
    # Grouping by these fields allows us to see
    # whether the same box has multiple actions.

    group_columns = [
        "video_id",
        "timestamp",
        "x1",
        "y1",
        "x2",
        "y2",
        "person_id"
    ]

    grouped = (
        df.groupby(group_columns)["action_id"]
        .agg(list)
        .reset_index()
    )

    single_label = 0
    multi_label = 0

    behaviour_counts = {
        label_id: 0
        for label_id in LABELS
    }

    for actions in grouped["action_id"]:

        unique_actions = set(
            int(x) for x in actions
        )

        if len(unique_actions) == 1:

            single_label += 1

            action_id = list(
                unique_actions
            )[0]

            if action_id in behaviour_counts:
                behaviour_counts[action_id] += 1

        else:

            multi_label += 1

    print(
        f"\nUnique cattle annotations: "
        f"{len(grouped):,}"
    )

    print(
        f"Single-label boxes: "
        f"{single_label:,}"
    )

    print(
        f"Multi-label boxes: "
        f"{multi_label:,}"
    )

    print("\nSingle-label behaviour distribution:")

    for action_id, count in behaviour_counts.items():

        print(
            f"{action_id}: "
            f"{LABELS[action_id]:20} "
            f"{count:,}"
        )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)