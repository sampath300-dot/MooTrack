
---

# 2. `behavior_model/README.md`

Use this for Person 2:

```markdown
# MooTrack - Behaviour Deep Learning Module

## Overview

This module is responsible for cattle behaviour recognition in the MooTrack project.

The module uses a pre-trained ResNet18 Convolutional Neural Network (CNN) and fine-tunes it on cattle behaviour data.

The model takes a cattle image or labelled keyframe as input and predicts the observed behaviour.

---

## Project Pipeline

Cattle Image / Keyframe
↓
Image Preprocessing
↓
Pre-trained ResNet18
↓
Fine-Tuning
↓
Behaviour Classification
↓
Confidence Score

---

## Dataset

### CBVD-5 Cow Behavior Video Dataset

Dataset:
CBVD-5 Cow Behavior Video Dataset

Source:
https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset

The dataset contains labelled cattle behaviour samples obtained from cattle monitoring/video data.

Before training, the actual downloaded dataset structure and annotation format must be inspected.

The exact class names used by the implementation must come from the dataset itself.

---

## Behaviour Classes

The commonly documented CBVD-5 behaviour categories include:

- Standing
- Lying
- Feeding
- Drinking
- Rumination

The implementation must verify the actual labels present in the downloaded dataset before training.

---

## Deep Learning Model

### ResNet18

MooTrack uses a pre-trained ResNet18 Convolutional Neural Network.

ResNet18 is an image-based Deep Learning model suitable for image classification.

The model is not trained from scratch.

Instead, a pre-trained ResNet18 is adapted to the cattle behaviour classification task through transfer learning and fine-tuning.

### Learning Approach

- Transfer Learning
- Fine-Tuning

---

## Image Preprocessing

Typical preprocessing includes:

1. Loading the cattle image/keyframe
2. Resizing the image
3. Converting the image into the required tensor format
4. Normalizing the image
5. Passing the processed image to ResNet18

Training and inference should use compatible preprocessing.

---

## Important Dataset Handling

The dataset may contain multiple frames originating from the same video.

To reduce data leakage:

- Avoid placing closely related frames from the same original video into both training and test sets when source metadata allows this.
- Keep the test set unseen during training and model selection.
- Use the actual annotations provided by the dataset.

---

## Project Structure

```text
behavior_model/
├── train.py
├── preprocessing.py
├── predict.py
├── config.py
├── README.md
├── model/
└── results/
