import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_model.config import (
    SAVED_MODEL_DIR,
    RESULTS_DIR,
    PRETRAINED_MODEL_NAME,
    ID2LABEL,
    LABEL2ID,
    BATCH_SIZE,
    LABEL_COLUMN,
    SAMPLING_RATE,
    MAX_DURATION_SEC,
    MAX_LENGTH,
)
from audio_model.train import CattleAudioDataset, prepare_cattle_splits


def plot_training_curves(history_path: Path, output_dir: Path):
    """Plot and save Loss and Accuracy curves from training history."""
    if not history_path.exists():
        print(f"[!] Warning: Training history file not found at {history_path}. Skipping curve generation.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "r--s", label="Validation Loss", linewidth=2)
    plt.title("MooTrack AST Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Cross-Entropy)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    loss_curve_path = output_dir / "loss_curve.png"
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"✓ Saved loss curve to: {loss_curve_path}")

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "g-o", label="Training Accuracy", linewidth=2)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "m--s", label="Validation Accuracy", linewidth=2)
    plt.title("MooTrack AST Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    acc_curve_path = output_dir / "accuracy_curve.png"
    plt.savefig(acc_curve_path, dpi=300)
    plt.close()
    print(f"✓ Saved accuracy curve to: {acc_curve_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plot and save Confusion Matrix."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("MooTrack Cattle Valence Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Annotate numbers inside matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:d}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    plt.ylabel("True Emotional Valence", fontsize=12)
    plt.xlabel("Predicted Emotional Valence", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✓ Saved confusion matrix plot to: {output_path}")


def evaluate_model():
    """
    Evaluates the fine-tuned AST model on the unseen test split.
    Calculates actual Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating using device: {device}")

    # 1. Determine model directory (fine-tuned model preferred)
    if SAVED_MODEL_DIR.exists() and (SAVED_MODEL_DIR / "config.json").exists():
        model_source = SAVED_MODEL_DIR
        print(f"Loading fine-tuned model from '{SAVED_MODEL_DIR}'...")
    else:
        print(f"[!] No fine-tuned checkpoint found at '{SAVED_MODEL_DIR}'.")
        print(f"    Evaluating with base checkpoint '{PRETRAINED_MODEL_NAME}' as baseline...")
        model_source = PRETRAINED_MODEL_NAME

    # 2. Load model and feature extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_source)
    model = ASTForAudioClassification.from_pretrained(
        model_source,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.eval()

    # 3. Load held-out test split
    print("Preparing held-out test split...")
    _, _, test_df = prepare_cattle_splits()
    test_dataset = CattleAudioDataset(test_df, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Running evaluation on {len(test_df)} test samples...")
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 4. Calculate real metrics
    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds)
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=[ID2LABEL[0], ID2LABEL[1]],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_samples": len(test_df),
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    # 5. Save metrics JSON
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics successfully saved to: {metrics_path}")

    # 6. Plot & Save Curves and Confusion Matrix
    class_names = [ID2LABEL[0], ID2LABEL[1]]
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    plot_training_curves(RESULTS_DIR / "training_history.json", RESULTS_DIR)

    # 7. Print Console Summary
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Test Samples: {len(test_df)}")
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Macro Precision:    {p_macro:.4f}")
    print(f"Macro Recall:       {r_macro:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()
