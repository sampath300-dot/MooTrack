# MooTrack: Multimodal Cattle Monitoring System

<div align="center">

**Sahyadri College of Engineering & Management, Mangaluru**  
*(Affiliated to Visvesvaraya Technological University, Belagavi)*  
**Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)**  
**Subject:** Neural Networks and Deep Learning &bull; **Subject Code:** `AM722T2A`

---

### 👥 Project Team

| Student Name | USN | Core Focus |
|:---|:---:|:---|
| **Manikanta** | `4SF23CI076` | Acoustic Valence Deep Learning (AST) & Feature Engineering |
| **Sai Sudarshan** | `4SF23CI128` | Vision Dataset Preparation & ResNet18 Architecture |
| **Sampath** | `4SF23CI130` | System Architecture, UI/UX Redesign & Full-Stack Dashboards |
| **Dhruva Shetty** | `4SF23CI147` | Bioacoustic Telemetry & Human Speech Discrimination Gate |

---

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

</div>

---

## 📌 Project Overview

**MooTrack** is a non-invasive, contactless cattle monitoring system that combines bioacoustic spectrogram screening with computer vision to monitor dairy cattle health and welfare:

1. **Acoustic Valence Screening (AST):** Uses an Audio Spectrogram Transformer to classify cow vocalizations into **Positive** (calm contact calls) and **Negative** (separation or distress calls), backed by AudioSet-based human speech rejection.
2. **Behavior Posture Recognition (ResNet18):** Classifies five primary cattle activities from photos and video streams: **Standing**, **Lying Down**, **Feeding**, **Drinking**, and **Rumination (Cud Chewing)**.
3. **Dual Monitoring Dashboards:** Provides an analytical Streamlit dashboard alongside a standalone pure-Python web server with on-demand webcam snapshotting, WebAudio microphone capture, and explicit user-driven analysis controls.

---

## 🚀 Quickstart & Execution

### 1. Installation
```powershell
# Clone repository
git clone https://github.com/sampath300-dot/MooTrack.git
cd MooTrack

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Dashboards

#### Option A: Standalone Web Portal (Port 8000)
```powershell
python -m app.web_dashboard
# Open http://127.0.0.1:8000 in your browser
```

#### Option B: Streamlit Dashboard (Port 8501)
```powershell
streamlit run streamlit_app.py
# Open http://localhost:8501 in your browser
```

---

## 📁 Repository Structure

```
MooTrack/
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── streamlit_app.py               # Streamlit application entrypoint
│
├── app/                           # Web application & dashboards
│   ├── web_dashboard.py           # Standalone web server (Port 8000)
│   ├── streamlit_dashboard.py     # Streamlit dashboard implementation
│   ├── prediction_history.json    # Diagnostic audit log
│   └── static/                    # Logo and brand assets
│
├── audio_model/                   # Acoustic Valence Pipeline (AST)
│   ├── config.py                  # Audio hyperparameters & class mappings
│   ├── predict.py                 # Audio inference & bioacoustic telemetry
│   ├── train.py                   # AST fine-tuning pipeline
│   └── test_samples/              # Audio test samples (.wav)
│
└── behavior_model/                # Vision Behavior Pipeline (ResNet18)
    ├── config.py                  # Vision hyperparameters & class dictionary
    ├── predict.py                 # Posture classification & video keyframing
    ├── train.py                   # ResNet18 training pipeline
    └── test_samples/              # Cattle test photos (.jpg) & video (.mp4)
```

---

## 📄 License
This project is developed for academic purposes under the VTU curriculum at Sahyadri College of Engineering & Management.
