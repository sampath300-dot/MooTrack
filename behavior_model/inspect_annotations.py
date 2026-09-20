import os

from behavior_model.config import DATASET_DIR

ANNOTATION_DIR = os.path.join(
    DATASET_DIR,
    "annotations"
)

print("=" * 70)
print("CBVD-5 ANNOTATION FILE INSPECTION")
print("=" * 70)

print("\nAnnotation directory:")
print(ANNOTATION_DIR)

print("\nFiles:")

for filename in sorted(os.listdir(ANNOTATION_DIR)):

    path = os.path.join(
        ANNOTATION_DIR,
        filename
    )

    if os.path.isfile(path):

        size = os.path.getsize(path)

        print(
            f"{filename:60} {size:,} bytes"
        )

print("\n" + "=" * 70)

# Inspect selected annotation files
selected_files = [
    "ava_train_v2.1.csv",
    "ava_val_v2.1.csv",
    "ava_test_v2.1.csv",
    "ava_train_excluded_timestamps_v2.1.csv",
    "ava_val_excluded_timestamps_v2.1.csv",
    "ava_test_excluded_timestamps_v2.1.csv",
]

for filename in selected_files:

    path = os.path.join(
        ANNOTATION_DIR,
        filename
    )

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    if not os.path.exists(path):

        print("File not found.")
        continue

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as f:

            for i in range(10):

                line = f.readline()

                if not line:
                    break

                print(
                    f"{i + 1}: {line.rstrip()}"
                )

    except Exception as e:

        print(
            "Could not read file:",
            e
        )