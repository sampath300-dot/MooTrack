# MooTrack - Audio Deep Learning Module

## 1. Overview
The **audio_model** module is the acoustic analysis component of **MooTrack**, an intelligent cattle monitoring system. It provides automated **emotional valence classification** from cattle vocalizations (calls) using state-of-the-art Deep Learning audio transformers.

> **Note:** This module performs **emotional valence classification** based on behavioral contexts. It is not designed for, nor should it be described as, disease detection, medical diagnosis, or guaranteed emotion recognition.

---

## 2. Project Objective
To build a robust, leakage-free Deep Learning audio analysis pipeline that takes an audio recording of a cow vocalization and classifies its emotional valence as either:
- **Positive** (associated with social reunion contexts)
- **Negative** (associated with physical/visual social separation contexts)

along with an associated confidence score:
```python
{
    "class": "Positive",
    "confidence": 0.91
}
```

---

## 3. Dataset
This module uses the **OpenFarm Ungulate Valence** dataset.
- **Scope:** Across all ungulate species, the dataset contains 3,181 call-level recordings from pigs, wild boars, cows, sheep, horses, and goats.
- **Cattle Subset:** **1,254** call-level vocalizations from cattle (*Bos taurus*).

---

## 4. Dataset Source
- **Hugging Face Hub:** [`oliveirabruno01/openfarm-ungulate-valence`](https://huggingface.co/datasets/oliveirabruno01/openfarm-ungulate-valence)
- **Primary Source Data:** Zenodo Repository ([DOI: 10.5281/zenodo.14636641](https://zenodo.org/records/14636641))

---

## 5. License & Attribution Note
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Original Research & Attribution:**
  > Padilla de la Torre, M., Hillmann, E., and Briefer, E.F. (2015). *Vocal expression of emotion in cattle*. In Proceedings of the 25th International Congress of the Bioacoustics Council, Murnau, Germany, 7–12 September 2015.
- **Redistribution Policy:** Under CC BY 4.0, raw audio dataset files are not redistributed through GitHub; rather, the dataset is loaded programmatically via Hugging Face Hub.

---

## 6. Cattle-Only Filtering
The dataset contains multiple ungulate species. For MooTrack:
- Only records with `species == "Cow/cattle"` are retained.
- This isolates **1,254** cattle vocalization calls across 32 unique cattle individuals.

---

## 7. Positive / Negative Valence Explanation
Valence labels in the dataset correspond to specific ethological conditions:
- **Negative Valence (1,179 calls, 94.02%):** Recorded during *Full separation* (1,078 calls) and *Physical separation* (101 calls) from the herd/pen mates.
- **Positive Valence (75 calls, 5.98%):** Recorded during *Visual reunion* (54 calls) and *Physical reunion* (21 calls) with conspecifics.

Because the data reflects natural separation vs. reunion situations, predictions reflect **emotional valence associated with social context**, not an absolute or permanent emotional state.

---

## 8. Deep Learning Model (AST)
We use the **Audio Spectrogram Transformer (AST)**:
- **Architecture:** An attention-based, vision-transformer architecture adapted for audio spectrogram patches.
- **Base Checkpoint:** `MIT/ast-finetuned-audioset-10-10-0.4593`.
- **Input Dimensions:** 128 Mel-frequency bins × 1024 time frames.

---

## 9. Transfer Learning
Training a transformer from scratch requires tens of thousands of audio hours. By using an AST pre-trained on ImageNet-21k and AudioSet (over 2 million sound events), the network already understands complex acoustic representations such as harmonic structures, timbre, and spectral energy transitions.

---

## 10. Fine-Tuning
The pre-trained AudioSet classification head (527 classes) is replaced with a custom linear classification head for **2 classes** (`Negative: 0`, `Positive: 1`). The entire network is fine-tuned end-to-end on the cattle vocalization dataset using class-weighted Cross-Entropy Loss to handle the natural class imbalance.

---

## 11. Audio Preprocessing
Both training and inference share the exact same preprocessing pipeline in `audio_model/preprocessing.py`:
1. **Audio Loading:** Supports file paths (`.wav`, `.flac`), raw audio `bytes`, or Hugging Face audio dictionaries.
2. **Channel Normalization:** Converts stereo/multi-channel audio to mono.
3. **Resampling:** Resamples all audio from native rates (44.1 kHz / 48 kHz) to the AST standard **16,000 Hz**.
4. **Spectrogram Generation:** Computes 128-band log Mel filterbank features padded or truncated to 1024 time frames (`[1, 1024, 128]`).

---

## 12. Dataset Splitting Strategy
To ensure sound statistical evaluation:
- **Train Set (60%):** 754 samples (51 Positive, 703 Negative).
- **Validation Set (20%):** 253 samples (10 Positive, 243 Negative).
- **Held-out Test Set (20%):** 247 samples (14 Positive, 233 Negative).

---

## 13. Leakage Prevention
Naive random splitting (`train_test_split`) causes severe data leakage because multiple audio clips were extracted from the same physical recording sessions. 
- **Grouping Key:** All partitions are split using `StratifiedGroupKFold` grouped on `audio_sha256` (the recording file hash).
- **Overlap Verification:** Exactly **0% overlap** of physical audio recordings between Train, Validation, and Test sets.

---

## 14. Training Instructions

### To Train Locally:
```powershell
python -m audio_model.train
```

### To Train on Google Colab (Recommended GPU Acceleration):
1. Upload the `audio_model/` folder to Google Drive / Colab.
2. Enable GPU in Colab (`Runtime -> Change runtime type -> T4 GPU`).
3. Run `python -m audio_model.train`.
4. Download the resulting `model/` folder back into `c:\MooTrack\audio_model\model\`.

---

## 15. Evaluation
The evaluation script evaluates the model on the unseen test partition and computes real, non-fabricated metrics:
- Accuracy, Macro Precision, Macro Recall, Macro F1-Score
- Confusion Matrix visualization
- Training & validation loss/accuracy curves

Run evaluation:
```powershell
python -m audio_model.evaluate
```
Results are saved to `audio_model/results/`:
- `metrics.json`
- `confusion_matrix.png`
- `loss_curve.png`
- `accuracy_curve.png`

---

## 16. Inference & Interactive Live Web App

### Command-Line Inference:
```powershell
python -m audio_model.predict path/to/vocalization.wav
```

### Live Audio Recording & Classification Web Interface:
Launch the interactive web application to record live cattle vocalizations with a microphone, visualize real-time waveforms, and instantly classify emotional valence:
```powershell
python -m audio_model.web_app
```
Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. All recorded cattle voices are automatically stored in `audio_model/recordings/` and displayed below for immediate playback, re-analysis, and export.

---

## 17. Project Structure
```text
audio_model/
├── config.py              # Central configuration, hyperparams, class maps
├── preprocessing.py       # Audio loading, resampling, AST feature extraction
├── train.py               # Leakage-free split, AST fine-tuning pipeline
├── evaluate.py            # Test set evaluation, metrics, confusion matrix
├── predict.py             # Public inference API for MooTrack app
├── test_inference.py      # Automated unit & integration tests
├── inspect_dataset.py     # Dataset schema inspection & stats verification
├── web_app.py             # Live audio recording and classification web app
├── README.md              # Module documentation
├── test_samples/          # Extracted real cattle audio samples for testing
├── recordings/            # Vault storing all live recorded audio clips
├── model/                 # Fine-tuned model weights and feature extractor
└── results/               # Confusion matrix, loss curves, metrics.json
```

---

## 18. Example `predict_audio()`
For teammate integration (e.g. Person 3 in `app/`):
```python
from audio_model.predict import predict_audio

result = predict_audio("audio_model/test_samples/cattle_positive_sample.wav")
print(result)
# Output:
# {'class': 'Positive', 'confidence': 0.89}
```

---

## 19. Limitations
1. **Context-Specific Valence:** Valence labels represent social separation vs. reunion contexts and should not be equated with general livestock health or contentment outside these scenarios.
2. **Acoustic Backgrounds:** Extreme background machinery or non-cattle farm noise may affect confidence scores; audio preprocessing assumes vocalization segments are predominantly cattle calls.
3. **Class Imbalance:** Naturally, separation events generate significantly more frequent vocalizations than reunions, leading to fewer positive training examples.

---

## 20. MooTrack Integration
The audio module operates independently from the web dashboard and behavior modules. It only requires Person 3 to import `predict_audio`:
```python
from audio_model.predict import predict_audio
```
No frontend code or database schemas need to know about PyTorch internals or dataset loaders.
