# MooTrack — Cattle Behaviour Recognition & Video Prediction Module

## 1. Project Overview
The **behavior_model** module is the computer vision engine of **MooTrack**, an intelligent cattle monitoring and precision livestock farming system. 

It provides automated classification of cattle postures and ethological behaviors from camera feeds, static images, and continuous video streams:
- **Single Image / Keyframe Prediction**: Fast single-frame posture classification with confidence scoring and probability distributions.
- **Video Stream Ingestion & Temporal Analysis**: Decodes video feeds (`.mp4`, `.avi`, `.mov`, `.webm`), extracts keyframes at a sampled rate (e.g. 1 FPS), runs deep learning inference per frame, and aggregates temporal time-budgets (% time standing, % feeding, % lying, % drinking, % rumination) and timeline transitions.

> **Project Disclaimer:** This model is a **behaviour recognition prototype** for research and livestock monitoring assistance. It is not a veterinary diagnostic system and does not detect cattle disease or provide medical diagnosis.

---

## 2. Dataset: CBVD-5 (Cow Behavior Video Dataset)
- **Source:** [Kaggle: CBVD-5 Cow Behavior Video Dataset](https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset)
- **Modality:** High-resolution video recordings and annotated bounding-box crops across dairy and beef cattle in agricultural and open barn environments.
- **Dataset Configuration:** The dataset root can be configured via the `MOOTRACK_DATASET` environment variable or placed in `behavior_model/dataset/`.

### The 5 Target Cattle Behaviors:
| ID | Standard Key | CBVD Label | Ethological Meaning & Farm Importance |
|:---:|:---:|:---:|:---|
| 0 | `drinking` | Drinking water | Water ingestion at trough/drinker. Crucial for thermoregulation and milk yield (typically 60–120 L/day). |
| 1 | `feeding` | Foraging | Active consumption of forage, silage, or concentrate. Primary indicator of dry matter intake (DMI). |
| 2 | `lying` | Lying down | Sternal or lateral recumbency rest. Essential for rumen digestion, hoof health, and milk synthesis (10–14 hrs/day). |
| 3 | `rumination` | Rumination | Rhythmic chewing of regurgitated cud. Direct indicator of rumen microbial fermentation and welfare comfort. |
| 4 | `standing` | Stand | Upright alert, socialization, or resting posture. Standard daylight baseline activity. |

---

## 3. Deep Learning Architecture: Fine-Tuned ResNet18
- **Base Architecture:** Pre-trained 18-layer Deep Residual Network (`ResNet18`) with residual skip connections.
- **Transfer Learning:** Initialized with ImageNet weights to leverage rich generalized visual features (body contours, textures, postures).
- **Classification Head:** Linear classifier mapped to the 5 target behaviors (`Linear(in_features=512, out_features=5)`).
- **Normalization:** ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and standard deviation ($\sigma = [0.229, 0.224, 0.225]$) resized to $224 \times 224$.

---

## 4. Video Processing & Temporal Prediction Engine

The video inference engine (`behavior_model/video_processor.py` & `behavior_model/predict.py`) executes:
1. **Video Ingestion:** Opens `.mp4`, `.avi`, `.mov`, `.webm`, or `.mkv` files using OpenCV (`cv2.VideoCapture`).
2. **Keyframe Sampling:** Extracts frames at a configurable sampling rate (default: `1.0 FPS` / 1 frame per second) up to a maximum limit (`max_frames=120`).
3. **Per-Frame Inference:** Normalizes and evaluates each extracted frame through the ResNet18 model.
4. **Temporal Aggregation:**
   - Calculates **Dominant Behavior** and average confidence across the video duration.
   - Calculates **Activity Time-Budget Breakdown** (% time spent in each behavior).
   - Generates an interactive **Frame-by-Frame Timeline** with formatted timestamps (`00:01.0`, `00:02.0`, etc.) and lightweight base64 thumbnail previews.
   - Detects **Behavioral Transitions** (e.g., transition from standing to feeding).

```
   [ Video File (.mp4/.mov) ]
              │
              ▼
   [ OpenCV Frame Sampling ] ──> Extracts 1 FPS keyframes
              │
              ▼
   [ ResNet18 Inference ]    ──> Computes per-frame probabilities
              │
              ▼
   [ Temporal Aggregator ]   ──> Dominant Class + Time-Budget % + Frame Timeline
```

---

## 5. Test Results & Metrics

Evaluated on the CBVD-5 cattle dataset:

| Behaviour | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| **Stand** | 99.39% | 98.62% | 99.00% |
| **Lying down** | 97.42% | 98.80% | 98.11% |
| **Foraging (Feeding)** | 83.17% | 91.13% | 86.97% |
| **Drinking water** | 61.42% | 87.64% | 72.22% |
| **Rumination** | 64.65% | 77.58% | 70.52% |
| **Macro Average** | **81.21%** | **90.75%** | **85.37%** |

Artifacts in `behavior_model/results/`:
- `loss_curve.png` & `accuracy_curve.png`
- `f1_curve.png`
- `confusion_matrices.png` & `confusion_matrix.png`
- `test_metrics.csv` & `training_history.csv`

---

## 6. How to Run Predictions

### A. Python API Usage
```python
from behavior_model.predict import predict_behavior

# 1. Image Prediction
img_result = predict_behavior("behavior_model/test_samples/sample_standing.jpg")
print(img_result)
# Output: {'class': 'standing', 'confidence': 0.72, 'probabilities': {...}, ...}

# 2. Video Prediction (Automatic video detection)
vid_result = predict_behavior("behavior_model/test_samples/sample_cattle_video.mp4")
print("Dominant Behavior:", vid_result["class"])
print("Activity Breakdown:", vid_result["activity_breakdown"])
print("Timeline:", vid_result["timeline"][:3])
```

### B. Command-Line Interface (CLI)
```powershell
# Analyze an image:
python -m behavior_model.predict behavior_model/test_samples/sample_feeding.jpg

# Analyze a video:
python -m behavior_model.predict behavior_model/test_samples/sample_cattle_video.mp4
```

### C. Run Verification Suite
```powershell
python -m behavior_model.test_inference
```

---

## 7. Web Dashboard Integration
The model is connected directly to the MooTrack Web Dashboard (`app/web_dashboard.py`):
- **Live Camera / Photo Upload**: Instant activity classification with visual probability bars.
- **Video Upload & Analysis**: Upload cow video clips (`.mp4`, `.webm`, `.mov`), view interactive video preview, inspect activity time-budget bars, and browse frame-by-frame snapshot timelines.
- **Unified Farm Well-being Index**: Combines vision behavior with cow vocal sound analysis.

Run dashboard:
```powershell
python app/web_dashboard.py
# Open: http://127.0.0.1:8000
```


