# Cattle Behaviour Recognition

## Overview

This module implements cattle behaviour recognition for the MooTrack project using a deep learning model.

A pretrained ResNet18 CNN is used with transfer learning and fine-tuning to classify cattle behaviour from cropped cattle images.

The model supports multi-label behaviour annotations because a cattle image can contain more than one behaviour label.

## Dataset

Dataset used:

**CBVD-5 Cow Behavior Video Dataset**

The dataset contains five behaviour classes:

| ID | Behaviour |
|---|---|
| 0 | Stand |
| 1 | Lying down |
| 2 | Foraging |
| 3 | Drinking water |
| 4 | Rumination |

The original dataset annotations were processed into cattle image crops using the annotated bounding boxes.

The dataset itself is not included in this repository.

## Model

**Architecture:** ResNet18

**Approach:**
- Transfer learning using pretrained ResNet18
- Fine-tuning on CBVD-5 cattle behaviour data
- Five output neurons
- Sigmoid activation
- Binary Cross Entropy with Logits Loss
- Multi-label classification

## Dataset Processing

The preprocessing pipeline:

1. Reads CBVD-5 annotations.
2. Extracts cattle bounding boxes.
3. Converts annotations into behaviour label vectors.
4. Crops cattle regions from the original frames.
5. Resizes images to 224 × 224.
6. Applies ImageNet normalization.
7. Creates train, validation, and test datasets.

Processed dataset:

- Training samples: 20,041
- Validation samples: 2,702
- Test samples: 2,581
- Total samples: 25,324

## Training

Training configuration:

- Model: ResNet18
- Image size: 224 × 224
- Batch size: 32
- Learning rate: 0.0001
- Epochs: 15
- Random seed: 42

The best model checkpoint is selected using validation macro F1-score.

## Test Results

Final test performance:

| Behaviour | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Stand | 99.39% | 98.62% | 99.00% |
| Lying down | 97.42% | 98.80% | 98.11% |
| Foraging | 83.17% | 91.13% | 86.97% |
| Drinking water | 61.42% | 87.64% | 72.22% |
| Rumination | 64.65% | 77.58% | 70.52% |
| **Macro Average** | **81.21%** | **90.75%** | **85.37%** |

Subset accuracy: **80.05%**

The best validation macro F1-score was **82.47% at Epoch 8**.

## Results

The `results/` directory contains:

- `loss_curve.png` — training and validation loss
- `accuracy_curve.png` — validation accuracy
- `f1_curve.png` — validation F1-score
- `confusion_matrices.png` — per-class confusion matrices
- `training_history.csv` — epoch-wise training history
- `test_metrics.csv` — final test metrics

## Prediction

Prediction can be performed using:

```python
from behavior_model.predict import predict_behavior

result = predict_behavior(image_path)

print(result)