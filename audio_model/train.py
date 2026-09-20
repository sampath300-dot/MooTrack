import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    ASTForAudioClassification,
    ASTFeatureExtractor,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

from audio_model.config import (
    DATASET_NAME,
    SPECIES_FILTER,
    LABEL_COLUMN,
    GROUP_COLUMN,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    PRETRAINED_MODEL_NAME,
    SAVED_MODEL_DIR,
    RESULTS_DIR,
    SAMPLING_RATE,
    MAX_DURATION_SEC,
    MAX_LENGTH,
    RANDOM_SEED,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
)
from audio_model.preprocessing import load_and_resample_audio, extract_ast_features


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CattleAudioDataset(Dataset):
    """
    PyTorch Dataset for Cattle Vocalizations.
    Applies unified preprocessing consistent with inference pipeline.
    """
    def __init__(self, df: pd.DataFrame, feature_extractor: ASTFeatureExtractor):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_data = row["audio"]
        label_str = row[LABEL_COLUMN]
        label_id = LABEL2ID[label_str]

        # 1. Load and resample audio
        waveform, sr = load_and_resample_audio(
            audio_input=audio_data,
            target_sr=SAMPLING_RATE,
            max_duration_sec=MAX_DURATION_SEC,
        )

        # 2. Extract AST spectrogram feature tensor [1, 1024, 128] -> squeeze to [1024, 128]
        input_values = extract_ast_features(
            audio_waveform=waveform,
            feature_extractor=self.feature_extractor,
            sampling_rate=sr,
            max_length=MAX_LENGTH,
        ).squeeze(0)

        return {
            "input_values": input_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def prepare_cattle_splits():
    """
    Loads dataset from Hugging Face, filters for cattle,
    and applies leakage-safe Stratified Group Splitting.
    """
    print(f"Loading '{DATASET_NAME}'...")
    ds = load_dataset(DATASET_NAME)

    # Combine raw partitions to get full cattle cohort (1,254 calls)
    train_raw = ds["train_raw"].to_pandas()
    test_raw = ds["test_raw"].to_pandas()
    all_raw = pd.concat([train_raw, test_raw], ignore_index=True)

    # Filter cattle only
    cow_df = all_raw[all_raw["species"] == SPECIES_FILTER].copy().reset_index(drop=True)
    print(f"Filtered {len(cow_df)} cattle vocalization recordings.")

    # 1. Split Train+Val vs Test (Grouped on physical recording file: audio_sha256)
    sgkf_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(sgkf_test.split(cow_df, cow_df[LABEL_COLUMN], groups=cow_df[GROUP_COLUMN]))

    train_val_df = cow_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = cow_df.iloc[test_idx].reset_index(drop=True)

    # 2. Split Train vs Val (Grouped on audio_sha256)
    sgkf_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sgkf_val.split(train_val_df, train_val_df[LABEL_COLUMN], groups=train_val_df[GROUP_COLUMN]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"Split Summary (Grouped on '{GROUP_COLUMN}'):")
    print(f"  • Train: {len(train_df)} calls (Pos: {(train_df[LABEL_COLUMN]=='Positive').sum()}, Neg: {(train_df[LABEL_COLUMN]=='Negative').sum()})")
    print(f"  • Val:   {len(val_df)} calls (Pos: {(val_df[LABEL_COLUMN]=='Positive').sum()}, Neg: {(val_df[LABEL_COLUMN]=='Negative').sum()})")
    print(f"  • Test:  {len(test_df)} calls (Pos: {(test_df[LABEL_COLUMN]=='Positive').sum()}, Neg: {(test_df[LABEL_COLUMN]=='Negative').sum()})")

    # Save test metadata for downstream evaluation without re-splitting
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_meta_path = RESULTS_DIR / "test_split_metadata.parquet"
    # Exclude heavy raw audio object when saving metadata table
    meta_cols = [c for c in test_df.columns if c != "audio"]
    test_df[meta_cols].to_parquet(test_meta_path, index=False)
    print(f"Saved test split metadata to {test_meta_path}")

    return train_df, val_df, test_df


def train_model(epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 1. Feature Extractor
    print(f"Loading AST Feature Extractor from '{PRETRAINED_MODEL_NAME}'...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)

    # 2. Prepare Data Splits
    train_df, val_df, test_df = prepare_cattle_splits()

    train_dataset = CattleAudioDataset(train_df, feature_extractor)
    val_dataset = CattleAudioDataset(val_df, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Handle Class Imbalance with Weighted Cross-Entropy Loss
    neg_count = (train_df[LABEL_COLUMN] == "Negative").sum()
    pos_count = (train_df[LABEL_COLUMN] == "Positive").sum()
    weight_neg = 1.0
    weight_pos = float(neg_count / max(pos_count, 1))
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(device)
    print(f"Loss Class Weights (Negative={weight_neg:.2f}, Positive={weight_pos:.2f})")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Load Pre-trained AST Model and modify classification head for 2 classes
    print(f"Initializing ASTForAudioClassification from '{PRETRAINED_MODEL_NAME}' for {NUM_CLASSES} classes...")
    model = ASTForAudioClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # 5. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    # History tracker
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("STARTING AST FINE-TUNING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        train_preds, train_targets = [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = accuracy_score(train_targets, train_preds)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(inputs)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

                running_val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = accuracy_score(val_targets, val_preds)
        epoch_val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_f1"].append(epoch_val_f1)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%, Val Macro F1: {epoch_val_f1:.4f}"
        )

        # Save checkpoint if best validation F1
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            print(f"  -> New best validation Macro F1 ({epoch_val_f1:.4f})! Saving model to {SAVED_MODEL_DIR}...")
            model.save_pretrained(SAVED_MODEL_DIR)
            feature_extractor.save_pretrained(SAVED_MODEL_DIR)

    # Save training history
    history_path = RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. History saved to {history_path}")
    print(f"Best model weights saved in: {SAVED_MODEL_DIR}")


if __name__ == "__main__":
    train_model()
