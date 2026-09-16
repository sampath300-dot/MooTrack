import os
import numpy as np
import matplotlib.pyplot as plt

from behavior_model.config import (
    CLASS_NAMES,
    RESULTS_DIR,
    CONFUSION_MATRIX_PATH
)


def main():

    print("=" * 70)
    print("CREATING CONFUSION MATRIX IMAGE")
    print("=" * 70)

    # Load saved confusion matrices
    matrices = np.load(CONFUSION_MATRIX_PATH)

    print("\nLoaded confusion matrices:")
    print("Shape:", matrices.shape)

    # Create one figure containing all 5 binary confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for i, class_name in enumerate(CLASS_NAMES):

        cm = matrices[i]

        ax = axes[i]

        image = ax.imshow(cm)

        ax.set_title(class_name, fontsize=11)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])

        # Write values inside matrix
        for row in range(2):
            for col in range(2):

                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    fontsize=12
                )

    fig.suptitle(
        "ResNet18 Multi-Label Cattle Behaviour Confusion Matrices",
        fontsize=15
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrices.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion matrix image saved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()