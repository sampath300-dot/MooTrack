import os
from pathlib import Path

# ============================================================
# DIRECTORY PATHS (Project-Relative by default)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Dataset Configuration (CBVD-5 Cow Behavior Video Dataset)
DATASET_NAME = "CBVD-5 Cow Behavior Video Dataset"
DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset"

# Dataset root: checks environment variable 'MOOTRACK_DATASET' or defaults to local dataset folder
DATASET_DIR = Path(os.environ.get("MOOTRACK_DATASET", str(BASE_DIR / "dataset"))).resolve()
FRAME_DIR = DATASET_DIR / "labelframes" / "labelframes"
PROCESSED_DIR = DATASET_DIR / "processed_behavior"
CROP_DIR = PROCESSED_DIR / "crops"
TEST_SAMPLES_DIR = BASE_DIR / "test_samples"

# Storage Paths
SAVED_MODEL_DIR = BASE_DIR / "model"
SAVED_MODEL_PATH = SAVED_MODEL_DIR / "best_model.pth"
ALT_MODEL_PATH = SAVED_MODEL_DIR / "best_resnet18.pth"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET CLASSES & BIDIRECTIONAL MAPPINGS
# ============================================================

# Standard 5 MooTrack Classes
BEHAVIOR_CLASSES = ["drinking", "feeding", "lying", "rumination", "standing"]
CLASS_NAMES = BEHAVIOR_CLASSES
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrices.npy"
NUM_CLASSES = len(BEHAVIOR_CLASSES)

ID2LABEL = {
    0: "drinking",
    1: "feeding",
    2: "lying",
    3: "rumination",
    4: "standing",
}

LABEL2ID = {
    "drinking": 0,
    "feeding": 1,
    "lying": 2,
    "rumination": 3,
    "standing": 4,
}

# Original CBVD-5 naming conventions
CBVD_CLASS_NAMES = [
    "Drinking water",
    "Foraging",
    "Lying down",
    "Rumination",
    "Stand",
]

# Normalization maps
CBVD_TO_STANDARD = {
    "drinking water": "drinking",
    "drinking": "drinking",
    "drinking_water": "drinking",
    "foraging": "feeding",
    "feeding": "feeding",
    "lying down": "lying",
    "lying_down": "lying",
    "lying": "lying",
    "rumination": "rumination",
    "stand": "standing",
    "standing": "standing",
}

STANDARD_TO_CBVD = {
    "drinking": "Drinking water",
    "feeding": "Foraging",
    "lying": "Lying down",
    "rumination": "Rumination",
    "standing": "Stand",
}

# Ethological Behavior Descriptions
BEHAVIOR_DESCRIPTIONS = {
    "standing": "Cattle is upright in an alert, resting, or socializing posture. Standard baseline posture observed throughout daylight hours.",
    "lying": "Cattle is resting in sternal or lateral recumbency. Critical for rumination, hoof health, and overall dairy herd welfare (typically 10-14 hours/day).",
    "feeding": "Cattle is actively consuming forage, silage, or concentrate from the feed bunk or pasture. Key indicator of dry matter intake and metabolic health.",
    "drinking": "Cattle is ingesting water at the trough or drinker. Water intake is vital for thermoregulation and milk yield (typically 60-120L/day).",
    "rumination": "Cattle is chewing regurgitated cud (chewing rhythm). Primary indicator of digestive health, microbial fermentation, and welfare comfort.",
}


# ============================================================
# IMAGE & PREPROCESSING SETTINGS
# ============================================================

BASE_MODEL_NAME = "resnet18"
PRETRAINED_WEIGHTS = "DEFAULT"

IMAGE_SIZE = 224
CROP_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# VIDEO PROCESSING HYPERPARAMETERS
# ============================================================

VIDEO_SAMPLE_FPS = 1.0          # Sample 1 frame per second
VIDEO_MAX_FRAMES = 120          # Maximum frames to analyze per video
KEYFRAME_THUMBNAIL_WIDTH = 320  # Preview thumbnail width for UI
PREDICTION_THRESHOLD = 0.5      # Sigmoid threshold for multi-label presence

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp"
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"
}


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

RANDOM_SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
NUM_WORKERS = 0

TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
