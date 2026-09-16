import os

import torch
from PIL import Image
from torchvision import models, transforms

from .config import (
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    IMAGE_MEAN,
    IMAGE_STD,
    CLASS_NAMES,
    NUM_CLASSES,
    PREDICTION_THRESHOLD
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGE_MEAN,
        std=IMAGE_STD
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(
        BEST_MODEL_PATH
    ):

        raise FileNotFoundError(
            "Trained model checkpoint not found:\n"
            f"{BEST_MODEL_PATH}"
        )

    model = models.resnet18(
        weights=None
    )

    input_features = (
        model.fc.in_features
    )

    model.fc = torch.nn.Linear(
        input_features,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD MODEL ONCE
# ============================================================

MODEL = load_model()


# ============================================================
# PREDICT BEHAVIOUR
# ============================================================

def predict_behavior(image_path):

    """
    Predict the primary cattle behaviour.

    Parameters
    ----------
    image_path : str
        Path to a cattle crop image.

    Returns
    -------
    dict
        {
            "class": "...",
            "confidence": 0.0
        }
    """

    if not os.path.exists(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = MODEL(
            image_tensor
        )

        probabilities = torch.sigmoid(
            outputs
        )[0]

    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    confidence, class_index = torch.max(
        probabilities,
        dim=0
    )

    class_name = CLASS_NAMES[
        class_index.item()
    ]

    return {
        "class": class_name,
        "confidence": round(
            confidence.item(),
            4
        )
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m behavior_model.predict "
            "<image_path>"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    result = predict_behavior(
        image_path
    )

    print("\n" + "=" * 60)
    print("CATTLE BEHAVIOUR PREDICTION")
    print("=" * 60)

    print(
        f"\nBehaviour : "
        f"{result['class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print("=" * 60)