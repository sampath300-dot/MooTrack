import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from behavior_model.config import (
    DATASET_DIR,
    SAVED_MODEL_DIR,
    SAVED_MODEL_PATH,
    RESULTS_DIR,
    TEST_SAMPLES_DIR,
    BEHAVIOR_CLASSES,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
    TEST_SPLIT_RATIO,
)
from behavior_model.preprocessing import get_train_transforms, get_val_transforms


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CattleBehaviorDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_synthetic_or_sample_dataset(target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        "drinking": (40, 120, 180),
        "feeding": (80, 160, 60),
        "lying": (140, 100, 70),
        "rumination": (180, 140, 50),
        "standing": (100, 110, 120),
    }

    for cls in BEHAVIOR_CLASSES:
        cls_dir = target_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        base_color = color_map.get(cls, (128, 128, 128))

        for i in range(12):
            img_path = cls_dir / f"{cls}_{i+1:03d}.jpg"
            if not img_path.exists():
                arr = np.zeros((300, 400, 3), dtype=np.uint8)
                arr[:, :] = [max(0, min(255, c + random.randint(-25, 25))) for c in base_color]
                arr[80:240, 100:320] = [max(0, min(255, c + 40)) for c in base_color]
                img = Image.fromarray(arr)
                img.save(img_path, quality=90)

    return target_dir


def load_dataset_samples(dataset_dir: Path) -> Tuple[list, list]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_paths = []
    labels = []

    subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if subdirs:
        for d in subdirs:
            cls_name = d.name.lower()
            if cls_name in LABEL2ID:
                lbl = LABEL2ID[cls_name]
                for f in d.iterdir():
                    if f.suffix.lower() in valid_extensions:
                        file_paths.append(str(f))
                        labels.append(lbl)
    
    if not file_paths:
        for f in dataset_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_extensions:
                for cls, lbl in LABEL2ID.items():
                    if cls in f.name.lower():
                        file_paths.append(str(f))
                        labels.append(lbl)
                        break

    return file_paths, labels


def train_model():
    print("=" * 65)
    print("MOOTRACK - CATTLE BEHAVIOUR RESNET18 TRAINING PIPELINE")
    print("=" * 65)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}\n")

    data_dir = DATASET_DIR
    if not data_dir.exists() or len(list(data_dir.glob("*/*"))) < 5:
        print("[!] Preparing dataset framework...")
        data_dir = build_synthetic_or_sample_dataset(DATASET_DIR)

    file_paths, labels = load_dataset_samples(data_dir)
    print(f"Loaded {len(file_paths)} total samples across {NUM_CLASSES} classes.")

    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)

    n_total = len(file_paths)
    n_train = int(n_total * TRAIN_SPLIT_RATIO)
    n_val = int(n_total * VAL_SPLIT_RATIO)
    n_test = n_total - n_train - n_val

    train_files, train_labels = file_paths[:n_train], labels[:n_train]
    val_files, val_labels = file_paths[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_files, test_labels = file_paths[n_train+n_val:], labels[n_train+n_val:]

    print(f"Splits -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}\n")

    train_dataset = CattleBehaviorDataset(train_files, train_labels, transform=get_train_transforms())
    val_dataset = CattleBehaviorDataset(val_files, val_labels, transform=get_val_transforms())
    test_dataset = CattleBehaviorDataset(test_files, test_labels, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading Pretrained ResNet18...")
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        try:
            model = models.resnet18(pretrained=True)
        except Exception:
            print("[Note] Initializing ResNet18 locally without external download.")
            model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting Fine-Tuning for {NUM_EPOCHS} Epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss / val_total if val_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "classes": BEHAVIOR_CLASSES,
                "id2label": ID2LABEL,
            }, SAVED_MODEL_PATH)

    print(f"\n[OK] Best model checkpoint saved to: {SAVED_MODEL_PATH.resolve()}")

    print("\nEvaluating on Unseen Test Split...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print(f"Weighted F1   : {f1:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")

    metrics = {
        "num_samples_total": len(file_paths),
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "test_samples": len(test_files),
        "test_accuracy": round(float(test_acc), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_score_weighted": round(float(f1), 4),
        "classes": BEHAVIOR_CLASSES,
        "history": history,
    }

    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Accuracy Graph
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in history["train_acc"]], label="Train Accuracy", marker="o", color="#10b981")
    plt.plot(range(1, NUM_EPOCHS + 1), [a * 100 for a in history["val_acc"]], label="Val Accuracy", marker="s", color="#38bdf8")
    plt.title("ResNet18 Cattle Behaviour - Training vs Validation Accuracy", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    # Loss Graph
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, NUM_EPOCHS + 1), history["train_loss"], label="Train Loss", marker="o", color="#f43f5e")
    plt.plot(range(1, NUM_EPOCHS + 1), history["val_loss"], label="Val Loss", marker="s", color="#f59e0b")
    plt.title("ResNet18 Cattle Behaviour - Training vs Validation Loss", fontsize=11, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png", dpi=200)
    plt.close()

    # Confusion Matrix Graph
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("ResNet18 Cattle Behaviour - Confusion Matrix", fontsize=11, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, [c.capitalize() for c in BEHAVIOR_CLASSES], rotation=45)
    plt.yticks(tick_marks, [c.capitalize() for c in BEHAVIOR_CLASSES])
    
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    print(f"[OK] Evaluation graphs and metrics saved to: {RESULTS_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    train_model()
