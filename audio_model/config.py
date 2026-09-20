import os
from pathlib import Path

# Base Paths (Relative to project root, robust to execution from any folder)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Dataset Configuration
DATASET_NAME = "oliveirabruno01/openfarm-ungulate-valence"
SPECIES_FILTER = "Cow/cattle"
LABEL_COLUMN = "valence"
GROUP_COLUMN = "audio_sha256"

# Target Classes & Mapping
ID2LABEL = {0: "Negative", 1: "Positive"}
LABEL2ID = {"Negative": 0, "Positive": 1}
NUM_CLASSES = 2

# Model Configuration
PRETRAINED_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAVED_MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

# Audio Hyperparameters
SAMPLING_RATE = 16000
MAX_DURATION_SEC = 10.0
MAX_LENGTH = 1024  # AST Spectrogram max length

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Split Ratios (Leakage-safe group split: 60% Train, 20% Val, 20% Test)
TEST_SPLIT_RATIO = 0.20
VAL_SPLIT_RATIO = 0.25  # 25% of the 80% remaining = 20% of total
