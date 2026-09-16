import os
import random
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = r"C:\Users\Manikanta\Downloads\archive (1)"

ANNOTATION_DIR = os.path.join(
    DATASET_DIR,
    "annotations"
)

TRAIN_FILE = os.path.join(
    ANNOTATION_DIR,
    "ava_train_v2.1.csv"
)

VAL_FILE = os.path.join(
    ANNOTATION_DIR,
    "ava_val_v2.1.csv"
)

TEST_FILE = os.path.join(
    ANNOTATION_DIR,
    "ava_test_v2.1.csv"
)

OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "processed_behavior"
)

SEED = 42


# ============================================================
# LOAD ANNOTATIONS
# ============================================================

def load_annotations(path):

    df = pd.read_csv(
        path,
        header=None
    )

    df.columns = [
        "video_id",
        "timestamp",
        "x1",
        "y1",
        "x2",
        "y2",
        "action_id",
        "person_id"
    ]

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CREATING LEAKAGE-SAFE VIDEO SPLIT")
    print("=" * 70)

    print("\nLoading annotations...")

    train_df = load_annotations(TRAIN_FILE)
    val_df = load_annotations(VAL_FILE)
    test_df = load_annotations(TEST_FILE)

    # --------------------------------------------------------
    # Combine all annotation information
    # --------------------------------------------------------

    all_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    print(
        f"Total annotation rows: {len(all_df):,}"
    )

    # --------------------------------------------------------
    # Get unique video IDs
    # --------------------------------------------------------

    video_ids = sorted(
        all_df["video_id"].unique()
    )

    print(
        f"Total unique videos: {len(video_ids):,}"
    )

    # --------------------------------------------------------
    # Shuffle videos
    # --------------------------------------------------------

    random.seed(SEED)

    random.shuffle(video_ids)

    total_videos = len(video_ids)

    train_count = int(total_videos * 0.80)
    val_count = int(total_videos * 0.10)

    train_videos = video_ids[
        :train_count
    ]

    val_videos = video_ids[
        train_count:
        train_count + val_count
    ]

    test_videos = video_ids[
        train_count + val_count:
    ]

    # --------------------------------------------------------
    # Convert to sets
    # --------------------------------------------------------

    train_videos = set(train_videos)
    val_videos = set(val_videos)
    test_videos = set(test_videos)

    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    train_val = train_videos & val_videos
    train_test = train_videos & test_videos
    val_test = val_videos & test_videos

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print(
        f"Train videos      : {len(train_videos):,}"
    )

    print(
        f"Validation videos : {len(val_videos):,}"
    )

    print(
        f"Test videos       : {len(test_videos):,}"
    )

    print("\nOverlap check:")

    print(
        f"Train ∩ Validation : {len(train_val)}"
    )

    print(
        f"Train ∩ Test       : {len(train_test)}"
    )

    print(
        f"Validation ∩ Test  : {len(val_test)}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save video IDs
    # --------------------------------------------------------

    pd.DataFrame(
        sorted(train_videos),
        columns=["video_id"]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train_videos.csv"
        ),
        index=False
    )

    pd.DataFrame(
        sorted(val_videos),
        columns=["video_id"]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "val_videos.csv"
        ),
        index=False
    )

    pd.DataFrame(
        sorted(test_videos),
        columns=["video_id"]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_videos.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Filter annotations according to video split
    # --------------------------------------------------------

    train_split = all_df[
        all_df["video_id"].isin(train_videos)
    ]

    val_split = all_df[
        all_df["video_id"].isin(val_videos)
    ]

    test_split = all_df[
        all_df["video_id"].isin(test_videos)
    ]

    # --------------------------------------------------------
    # Save annotations
    # --------------------------------------------------------

    train_split.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "train_annotations.csv"
        ),
        index=False
    )

    val_split.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "val_annotations.csv"
        ),
        index=False
    )

    test_split.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_annotations.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Print annotation counts
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANNOTATION COUNTS")
    print("=" * 70)

    print(
        f"Train annotations      : {len(train_split):,}"
    )

    print(
        f"Validation annotations : {len(val_split):,}"
    )

    print(
        f"Test annotations       : {len(test_split):,}"
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    assert len(train_val) == 0
    assert len(train_test) == 0
    assert len(val_test) == 0

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "\nNo video appears in more than one split."
    )

    print(
        f"\nSplit files saved to:\n{OUTPUT_DIR}"
    )