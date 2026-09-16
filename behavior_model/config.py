import os

# ============================================================
# DATASET
# ============================================================

# Set the dataset location using an environment variable.
# Example on Windows:
# $env:MOOTRACK_DATASET = "C:\path\to\CBVD-5"

DATASET_DIR = os.environ.get(
    "MOOTRACK_DATASET",
    r"C:\Users\Manikanta\Downloads\archive (1)"
)
FRAME_DIR = os.path.join(
    DATASET_DIR,
    "labelframes",
    "labelframes"
)

PROCESSED_DIR = os.path.join(
    DATASET_DIR,
    "processed_behavior"
)

CROP_DIR = os.path.join(
    PROCESSED_DIR,
    "crops"
)


# ============================================================
# BEHAVIOUR CLASSES
# ============================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Stand",
    "Lying down",
    "Foraging",
    "Drinking water",
    "Rumination"
]


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = 224

# ImageNet normalization used for pretrained ResNet18
IMAGE_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGE_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 32

LEARNING_RATE = 0.0001

EPOCHS = 15

RANDOM_SEED = 42

# Number of CPU workers for DataLoader
# 0 is safest on Windows
NUM_WORKERS = 0


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = "ResNet18"

# Use pretrained ImageNet weights
PRETRAINED = True


# ============================================================
# MULTI-LABEL CLASSIFICATION
# ============================================================

# Each cattle crop can have more than one behaviour.
# Example:
# Stand + Rumination
# Stand + Foraging

MULTI_LABEL = True

# Probability threshold used to convert sigmoid
# probabilities into predicted labels.

PREDICTION_THRESHOLD = 0.5


# ============================================================
# MODEL CHECKPOINT
# ============================================================

MODEL_DIR = os.path.join(
    PROCESSED_DIR,
    "models"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_resnet18.pth"
)


# ============================================================
# TRAINING RESULTS
# ============================================================

RESULTS_DIR = os.path.join(
    MODEL_DIR,
    "results"
)

LOSS_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)

F1_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "f1_curve.png"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "multilabel_confusion_matrices.npy"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)