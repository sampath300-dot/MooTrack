import os
import ast
import pandas as pd
from PIL import Image

from .config import (
    DATASET_DIR,
    FRAME_DIR,
    PROCESSED_DIR,
    CROP_DIR
)

# ============================================================
# LABEL MAPPING
# ============================================================

# Original CBVD labels:
#
# 0 = Stand
# 1 = Lying down
# 2 = Foraging
# 3 = Drinking water
# 4 = Rumination

LABELS = {
    0: "stand",
    1: "lying_down",
    2: "foraging",
    3: "drinking_water",
    4: "rumination"
}


# ============================================================
# LOAD ORIGINAL CBVD ANNOTATIONS
# ============================================================

def load_cbvd_annotations():

    csv_path = os.path.join(
        DATASET_DIR,
        "CBVD-5.csv"
    )

    with open(
        csv_path,
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

    if header is None:

        raise ValueError(
            "CSV_HEADER not found in CBVD-5.csv"
        )

    data_lines = [
        line
        for line in lines
        if not line.startswith("#")
    ]

    from io import StringIO

    df = pd.read_csv(
        StringIO(
            header + "\n" + "".join(data_lines)
        )
    )

    return df


# ============================================================
# LOAD VIDEO SPLITS
# ============================================================

def load_video_split(split_name):

    path = os.path.join(
        PROCESSED_DIR,
        f"{split_name}_videos.csv"
    )

    df = pd.read_csv(path)

    return set(
        df["video_id"].astype(int)
    )


# ============================================================
# PARSE LABELS
# ============================================================

def parse_behaviour_labels(metadata):

    """
    Example:

    {"1":"0,4"}

    becomes:

    [0, 4]

    Meaning:

    Stand + Rumination
    """

    try:

        metadata_dict = ast.literal_eval(
            metadata
        )

        value = metadata_dict.get(
            "1",
            ""
        )

        labels = [
            int(x.strip())
            for x in value.split(",")
            if x.strip() != ""
        ]

        return labels

    except Exception:

        return []


# ============================================================
# GET FRAME NAME
# ============================================================

def get_frame_name(file_list):

    try:

        files = ast.literal_eval(
            file_list
        )

        if len(files) == 0:
            return None

        return files[0]

    except Exception:

        return None


# ============================================================
# CREATE MULTI-LABEL VECTOR
# ============================================================

def create_label_vector(labels):

    vector = [0] * 5

    for label in labels:

        if label in LABELS:

            vector[label] = 1

    return vector


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    df,
    split_name,
    allowed_videos
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 70)

    output_dir = os.path.join(
        CROP_DIR,
        split_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    records = []

    processed = 0
    missing = 0
    errors = 0

    # --------------------------------------------------------
    # Filter videos
    # --------------------------------------------------------

    df = df[
        df["file_list"]
        .apply(
            lambda x:
            get_frame_name(x) is not None
        )
    ].copy()

    # --------------------------------------------------------
    # Process annotations
    # --------------------------------------------------------

    for index, row in df.iterrows():

        frame_name = get_frame_name(
            row["file_list"]
        )

        if frame_name is None:
            continue

        # Extract video ID from filename

        try:

            video_id = int(
                frame_name.split("_")[0]
            )

        except Exception:

            errors += 1
            continue

        # Ensure video belongs to this split

        if video_id not in allowed_videos:

            continue

        frame_path = os.path.join(
            FRAME_DIR,
            frame_name
        )

        if not os.path.exists(frame_path):

            missing += 1
            continue

        labels = parse_behaviour_labels(
            row["metadata"]
        )

        labels = [
            label
            for label in labels
            if label in LABELS
        ]

        if not labels:

            continue

        try:

            # ------------------------------------------------
            # Open frame
            # ------------------------------------------------

            image = Image.open(
                frame_path
            ).convert("RGB")

            width, height = image.size

            # ------------------------------------------------
            # Original CBVD box:
            #
            # [2, x, y, width, height]
            # ------------------------------------------------

            box = ast.literal_eval(
                row["spatial_coordinates"]
            )

            shape_id, x, y, w, h = box

            # Pixel coordinates

            left = int(x)
            top = int(y)

            right = int(
                x + w
            )

            bottom = int(
                y + h
            )

            # Clamp

            left = max(
                0,
                min(left, width - 1)
            )

            top = max(
                0,
                min(top, height - 1)
            )

            right = max(
                left + 1,
                min(right, width)
            )

            bottom = max(
                top + 1,
                min(bottom, height)
            )

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            crop = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )

            # ------------------------------------------------
            # Filename
            # ------------------------------------------------

            label_string = "_".join(
                str(x)
                for x in labels
            )

            output_name = (
                f"{video_id}_"
                f"{frame_name.split('_')[1].replace('.jpg', '')}_"
                f"{index:06d}.jpg"
            )

            output_path = os.path.join(
                output_dir,
                output_name
            )

            crop.save(
                output_path,
                "JPEG",
                quality=95
            )

            # ------------------------------------------------
            # Save metadata record
            # ------------------------------------------------

            vector = create_label_vector(
                labels
            )

            records.append({
                "image_path": output_path,
                "video_id": video_id,
                "frame": frame_name,
                "labels": ",".join(
                    str(x) for x in labels
                ),
                "label_vector": ",".join(
                    str(x) for x in vector
                )
            })

            processed += 1

        except Exception as e:

            errors += 1

            print(
                f"Error at row {index}: {e}"
            )

    # --------------------------------------------------------
    # Save metadata CSV
    # --------------------------------------------------------

    metadata_path = os.path.join(
        output_dir,
        "metadata.csv"
    )

    pd.DataFrame(
        records
    ).to_csv(
        metadata_path,
        index=False
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nResults")
    print("-" * 70)

    print(
        f"Crops generated : {processed:,}"
    )

    print(
        f"Missing frames  : {missing:,}"
    )

    print(
        f"Errors          : {errors:,}"
    )

    print(
        f"Metadata saved  : {metadata_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CBVD-5 MULTI-LABEL CATTLE CROP GENERATION")
    print("=" * 70)

    # Load original annotations

    df = load_cbvd_annotations()

    print(
        f"\nOriginal annotations: "
        f"{len(df):,}"
    )

    # Load our leakage-safe video splits

    train_videos = load_video_split(
        "train"
    )

    val_videos = load_video_split(
        "val"
    )

    test_videos = load_video_split(
        "test"
    )

    print(
        f"Train videos: {len(train_videos)}"
    )

    print(
        f"Validation videos: {len(val_videos)}"
    )

    print(
        f"Test videos: {len(test_videos)}"
    )

    # Process

    process_split(
        df,
        "train",
        train_videos
    )

    process_split(
        df,
        "val",
        val_videos
    )

    process_split(
        df,
        "test",
        test_videos
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)