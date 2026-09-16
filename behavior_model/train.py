import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torchvision import models, transforms

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    multilabel_confusion_matrix
)

import matplotlib.pyplot as plt

from .config import (
    CROP_DIR,
    MODEL_DIR,
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    RANDOM_SEED,
    NUM_CLASSES,
    CLASS_NAMES
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class CattleBehaviorDataset(Dataset):

    def __init__(
        self,
        metadata_path,
        transform=None
    ):

        self.data = pd.read_csv(metadata_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = row["image_path"]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label_vector = [
            float(x)
            for x in str(
                row["label_vector"]
            ).split(",")
        ]

        labels = torch.tensor(
            label_vector,
            dtype=torch.float32
        )

        return image, labels


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

def create_datasets():

    train_metadata = os.path.join(
        CROP_DIR,
        "train",
        "metadata.csv"
    )

    val_metadata = os.path.join(
        CROP_DIR,
        "val",
        "metadata.csv"
    )

    test_metadata = os.path.join(
        CROP_DIR,
        "test",
        "metadata.csv"
    )

    train_dataset = CattleBehaviorDataset(
        train_metadata,
        train_transform
    )

    val_dataset = CattleBehaviorDataset(
        val_metadata,
        val_test_transform
    )

    test_dataset = CattleBehaviorDataset(
        test_metadata,
        val_test_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# CREATE RESNET18
# ============================================================

def create_model():

    print("\nLoading pretrained ResNet18...")

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # Replace final classification layer
    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model


# ============================================================
# CALCULATE POSITIVE WEIGHTS
# ============================================================

def calculate_pos_weights(dataset):

    labels = np.array([
        [
            float(x)
            for x in str(
                row["label_vector"]
            ).split(",")
        ]
        for _, row in dataset.data.iterrows()
    ])

    positive = labels.sum(axis=0)

    negative = (
        len(labels) - positive
    )

    pos_weight = (
        negative /
        np.maximum(
            positive,
            1
        )
    )

    print("\nClass positive weights:")

    for i, weight in enumerate(pos_weight):

        print(
            f"{CLASS_NAMES[i]:20} "
            f"{weight:.3f}"
        )

    return torch.tensor(
        pos_weight,
        dtype=torch.float32
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    return epoch_loss


# ============================================================
# VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).int()

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    # Exact-match / subset accuracy
    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def plot_training_curves(
    train_losses,
    val_losses,
    val_accuracy_scores,
    val_f1_scores
):

    results_dir = os.path.join(
        MODEL_DIR,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "ResNet18 Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            results_dir,
            "loss_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Accuracy curve
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        val_accuracy_scores,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "ResNet18 Validation Accuracy"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            results_dir,
            "accuracy_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # F1 curve
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        epochs,
        val_f1_scores,
        label="Validation Macro F1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")

    plt.title(
        "ResNet18 Validation Macro F1 Score"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            results_dir,
            "f1_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nTraining curves saved to:"
    )

    print(results_dir)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_training_history(
    train_losses,
    val_losses,
    val_accuracy_scores,
    val_precision_scores,
    val_recall_scores,
    val_f1_scores
):

    results_dir = os.path.join(
        MODEL_DIR,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    history = pd.DataFrame({
        "epoch": range(
            1,
            len(train_losses) + 1
        ),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_accuracy": val_accuracy_scores,
        "val_precision": val_precision_scores,
        "val_recall": val_recall_scores,
        "val_macro_f1": val_f1_scores
    })

    history_path = os.path.join(
        results_dir,
        "training_history.csv"
    )

    history.to_csv(
        history_path,
        index=False
    )

    print(
        f"\nTraining history saved: "
        f"{history_path}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def final_test_evaluation(
    model,
    loader,
    device
):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(
                images
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= 0.5
            ).int()

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Overall subset accuracy
    # --------------------------------------------------------

    test_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------------

    precision = precision_score(
        all_labels,
        all_predictions,
        average=None,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average=None,
        zero_division=0
    )

    print(
        f"\nSubset Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        "\nPer-class metrics:"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"\n{name}"
        )

        print(
            f"  Precision: "
            f"{precision[i]:.4f}"
        )

        print(
            f"  Recall:    "
            f"{recall[i]:.4f}"
        )

        print(
            f"  F1 Score:  "
            f"{f1[i]:.4f}"
        )

    # --------------------------------------------------------
    # Macro averages
    # --------------------------------------------------------

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    print(
        "\nMacro averages:"
    )

    print(
        f"Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{macro_recall:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # Save final metrics
    # --------------------------------------------------------

    results_dir = os.path.join(
        MODEL_DIR,
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    metrics = []

    for i, name in enumerate(
        CLASS_NAMES
    ):

        metrics.append({
            "class": name,
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1[i]
        })

    metrics.append({
        "class": "Macro Average",
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1
    })

    metrics.append({
        "class": "Subset Accuracy",
        "precision": np.nan,
        "recall": np.nan,
        "f1_score": test_accuracy
    })

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_path = os.path.join(
        results_dir,
        "test_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\nTest metrics saved: "
        f"{metrics_path}"
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    matrices = multilabel_confusion_matrix(
        all_labels,
        all_predictions
    )

    confusion_path = os.path.join(
        results_dir,
        "multilabel_confusion_matrices.npy"
    )

    np.save(
        confusion_path,
        matrices
    )

    print(
        "\nPer-class confusion matrices saved:"
    )

    print(confusion_path)

    return (
        test_accuracy,
        macro_precision,
        macro_recall,
        macro_f1
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    print(
        "=" * 70
    )

    print(
        "CBVD-5 RESNET18 "
        "CATTLE BEHAVIOUR TRAINING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets()

    print(
        f"\nTrain samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset):,}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = calculate_pos_weights(
        train_dataset
    )

    pos_weight = pos_weight.to(
        device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    train_losses = []
    val_losses = []
    val_accuracy_scores = []
    val_precision_scores = []
    val_recall_scores = []
    val_f1_scores = []

    best_f1 = -1.0

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        val_accuracy_scores.append(
            val_accuracy
        )

        val_precision_scores.append(
            val_precision
        )

        val_recall_scores.append(
            val_recall
        )

        val_f1_scores.append(
            val_f1
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Val Precision: "
            f"{val_precision:.4f}"
        )

        print(
            f"Val Recall: "
            f"{val_recall:.4f}"
        )

        print(
            f"Val F1: "
            f"{val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "num_classes":
                        NUM_CLASSES,

                    "class_names":
                        CLASS_NAMES,

                    "image_size":
                        IMAGE_SIZE,

                    "best_val_f1":
                        best_f1
                },
                BEST_MODEL_PATH
            )

            print(
                "Best model saved."
            )

    # --------------------------------------------------------
    # Save curves
    # --------------------------------------------------------

    plot_training_curves(
        train_losses,
        val_losses,
        val_accuracy_scores,
        val_f1_scores
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_training_history(
        train_losses,
        val_losses,
        val_accuracy_scores,
        val_precision_scores,
        val_recall_scores,
        val_f1_scores
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "\nLoaded best model "
        f"(Val F1: {best_f1:.4f})"
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    final_test_evaluation(
        model,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest model:"
        f"\n{BEST_MODEL_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()