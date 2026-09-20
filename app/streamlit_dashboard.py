"""
MOotrack — A Deep Learning-Based Multimodal Cattle Monitoring System
Sahyadri College of Engineering & Management, Mangaluru
Subject: Neural Networks and Deep Learning (AM722T2A)

Project Team:
1. Manikanta (4SF23CI076)
2. Sai Sudarshan (4SF23CI128)
3. Sampath (4SF23CI130)
4. Dhruva Shetty (4SF23CI147)
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_model.predict import predict_audio, compute_acoustic_features
from behavior_model.predict import predict_behavior, predict_behavior_image, predict_behavior_video
from behavior_model.config import BEHAVIOR_CLASSES, BEHAVIOR_DESCRIPTIONS

# Paths
AUDIO_SAMPLES_DIR = PROJECT_ROOT / "audio_model" / "test_samples"
BEHAVIOR_SAMPLES_DIR = PROJECT_ROOT / "behavior_model" / "test_samples"
HISTORY_FILE = PROJECT_ROOT / "app" / "prediction_history.json"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
LOGO_PATH = STATIC_DIR / "logo_clean.png" if (STATIC_DIR / "logo_clean.png").exists() else STATIC_DIR / "logo.png"

# Page Configuration
st.set_page_config(
    page_title="MOotrack — Multimodal Cattle Monitoring",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling matching modern aesthetic
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e0b2e 0%, #2e1047 50%, #180826 100%);
        padding: 24px 30px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
        border: 1px solid rgba(204, 255, 0, 0.2);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .college-sub {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #CCFF00;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .project-title {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: #FFFFFF;
    }
    .project-title span {
        color: #CCFF00;
    }
    .project-meta {
        font-size: 0.88rem;
        color: rgba(255, 255, 255, 0.75);
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-negative {
        background-color: rgba(244, 63, 94, 0.15);
        color: #E11D48;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    .behavior-pill {
        background: rgba(40, 17, 59, 0.08);
        color: #28113B;
        border: 1px solid rgba(40, 17, 59, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# History helper functions
def load_history_records():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history_record(record):
    records = load_history_records()
    records.insert(0, record)
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records[:200], f, indent=2)
    except Exception as e:
        st.error(f"Error saving history: {e}")


# Header Banner
st.markdown("""
<div class="main-header">
    <div class="college-sub">Sahyadri College of Engineering & Management, Mangaluru (VTU Belagavi)</div>
    <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6); margin-bottom: 8px;">Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)</div>
    <div class="project-title">MO<span>O</span>track</div>
    <div style="font-size: 1.15rem; font-weight: 600; color: #FFFFFF; margin-bottom: 8px;">A Deep Learning-Based Multimodal Cattle Monitoring System</div>
    <div class="project-meta">
        <strong>Subject:</strong> Neural Networks and Deep Learning &nbsp;|&nbsp; 
        <strong>Code:</strong> AM722T2A &nbsp;|&nbsp; 
        <strong>Team:</strong> Manikanta (4SF23CI076), Sai Sudarshan (4SF23CI128), Sampath (4SF23CI130), Dhruva Shetty (4SF23CI147)
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation & System Telemetry
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=180)
    st.title("🐄 MOotrack Navigation")
    st.caption("AI-Powered Cattle Acoustic Valence & Vision Behavior System")

    st.markdown("---")
    st.subheader("📌 Academic Details")
    st.markdown("""
    - **College:** Sahyadri College of Engg. & Mgmt.
    - **Dept:** CSE (AIML)
    - **Subject:** NNDL (`AM722T2A`)
    - **Models:** AST (Audio) + ResNet18 (Vision)
    """)

    st.markdown("---")
    st.subheader("👥 Project Team")
    st.markdown("""
    1. **Manikanta** (`4SF23CI076`)
    2. **Sai Sudarshan** (`4SF23CI128`)
    3. **Sampath** (`4SF23CI130`)
    4. **Dhruva Shetty** (`4SF23CI147`)
    """)

    st.markdown("---")
    st.caption("MooTrack v2.0 • Multimodal Livestock Assessment")


# Main Tabs
tab_multimodal, tab_vision, tab_audio, tab_dataset, tab_history, tab_about = st.tabs([
    "🌟 Multimodal Monitoring",
    "📸 Behavior Recognition (Vision)",
    "🔊 Sound Valence (Audio)",
    "📊 Dataset & Architecture",
    "📋 Observation History",
    "ℹ️ Project Poster & Info",
])


# ============================================================
# TAB 1: UNIFIED MULTIMODAL MONITORING
# ============================================================
with tab_multimodal:
    st.subheader("🌟 Dual-Stream Multimodal Cattle Assessment")
    st.write(
        "Upload or select **both Cattle Audio and Cow Image** simultaneously for a comprehensive, "
        "holistic assessment of the animal's physical posture and acoustic emotional valence."
    )

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### 1. Cow Image Input")
        mm_img_file = st.file_uploader("Upload Cattle Image", type=["jpg", "jpeg", "png", "webp"], key="mm_img")
        mm_img_sample = st.selectbox(
            "Or choose a sample cow image:",
            ["(None)", "sample_feeding.jpg", "sample_lying.jpg", "sample_standing.jpg", "sample_drinking.jpg", "sample_rumination.jpg"],
            key="mm_img_sample"
        )

        selected_image = None
        if mm_img_file is not None:
            selected_image = Image.open(mm_img_file).convert("RGB")
            st.image(selected_image, caption=f"Uploaded Image: {mm_img_file.name}", use_container_width=True)
        elif mm_img_sample != "(None)":
            sample_p = BEHAVIOR_SAMPLES_DIR / mm_img_sample
            if sample_p.exists():
                selected_image = Image.open(sample_p).convert("RGB")
                st.image(selected_image, caption=f"Sample Image: {mm_img_sample}", use_container_width=True)

    with col_m2:
        st.markdown("#### 2. Cattle Vocalization Audio Input")
        mm_aud_file = st.file_uploader("Upload Cattle Audio", type=["wav", "mp3", "ogg", "flac"], key="mm_aud")
        mm_aud_sample = st.selectbox(
            "Or choose a sample cattle vocalization:",
            ["(None)", "cow_moo_1.wav", "cow_moo_2.wav", "distress_call.wav"],
            key="mm_aud_sample"
        )

        selected_audio_path = None
        if mm_aud_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_a:
                tmp_a.write(mm_aud_file.read())
                selected_audio_path = tmp_a.name
            st.audio(selected_audio_path)
            st.caption(f"Uploaded Audio: {mm_aud_file.name}")
        elif mm_aud_sample != "(None)":
            sample_ap = AUDIO_SAMPLES_DIR / mm_aud_sample
            if sample_ap.exists():
                selected_audio_path = str(sample_ap)
                st.audio(selected_audio_path)
                st.caption(f"Sample Audio: {mm_aud_sample}")

    st.markdown("---")
    if st.button("🚀 Run Multimodal Dual-Analysis", type="primary", use_container_width=True):
        if selected_image is None and selected_audio_path is None:
            st.warning("Please provide at least an image or an audio file to run multimodal analysis.")
        else:
            with st.spinner("Processing Multimodal Deep Learning Pipelines (ResNet18 + AST)..."):
                start_time = time.time()
                vision_res = None
                audio_res = None

                # 1. Vision Forward Pass
                if selected_image is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_i:
                        selected_image.save(tmp_i.name)
                        tmp_img_path = tmp_i.name
                    try:
                        vision_res = predict_behavior(tmp_img_path)
                    finally:
                        if os.path.exists(tmp_img_path):
                            os.remove(tmp_img_path)

                # 2. Audio Forward Pass
                if selected_audio_path is not None:
                    try:
                        audio_res = predict_audio(selected_audio_path)
                    except Exception as e:
                        st.error(f"Audio processing error: {e}")

                elapsed = round((time.time() - start_time) * 1000, 1)

            st.success(f"Multimodal Inference Complete in {elapsed} ms")

            # Display Dual Results
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("### 👁️ Vision Behavior Prediction (ResNet18)")
                if vision_res:
                    b_class = vision_res.get("class", "Unknown").capitalize()
                    b_conf = vision_res.get("confidence", 0.0) * 100
                    st.metric("Observed Behavior", f"{b_class}", f"{b_conf:.1f}% Confidence")
                    st.info(f"**Ethological Context:** {vision_res.get('description', '')}")

                    # Bar Chart
                    probs = vision_res.get("probabilities", {})
                    if probs:
                        df_probs = pd.DataFrame({
                            "Behavior": [k.capitalize() for k in probs.keys()],
                            "Probability (%)": [v * 100 for v in probs.values()]
                        }).sort_values(by="Probability (%)", ascending=False)
                        st.bar_chart(df_probs.set_index("Behavior"))
                else:
                    st.write("No vision input provided.")

            with res_col2:
                st.markdown("### 🎙️ Acoustic Valence Prediction (AST)")
                if audio_res:
                    if audio_res.get("is_cattle_call", True):
                        v_class = audio_res.get("class", "Positive")
                        v_conf = audio_res.get("confidence", 0.0) * 100
                        badge_color = "green" if v_class == "Positive" else "red"
                        st.metric("Acoustic Valence", f"{v_class} Emotional State", f"{v_conf:.1f}% Confidence")
                        st.write(f"**Behavioral Context:** {audio_res.get('behavioral_context', '')}")

                        # Diagnostics
                        metrics = audio_res.get("audio_metrics", {})
                        if metrics:
                            m_col1, m_col2 = st.columns(2)
                            m_col1.metric("Call Type", metrics.get("call_type_estimate", "Bovine Call"))
                            m_col2.metric("Pitch (F0)", f"{metrics.get('f0_pitch_hz', 0)} Hz")
                    elif audio_res.get("is_human_speech"):
                        speech_pct = audio_res.get("speech_confidence", 95.0)
                        st.warning(f"🗣️ **Human Speaking Detected** ({speech_pct:.1f}% Speech Match)")
                        st.info("The AST AudioSet pre-screener recognized human vocalization. MooTrack evaluates cattle mooing for welfare scoring.")
                    else:
                        st.error(f"Audio Rejected: {audio_res.get('error', 'Non-cattle sound')}")
                else:
                    st.write("No audio input provided.")

            # Overall Holistic Welfare Index
            st.markdown("---")
            st.markdown("### 🩺 Holistic Cattle Welfare Summary")
            welfare_score = 90
            if vision_res and vision_res.get("class") in ["lying", "rumination"]:
                welfare_score = 95
            elif vision_res and vision_res.get("class") == "standing":
                welfare_score = 88

            if audio_res and audio_res.get("class") == "Negative":
                welfare_score -= 20

            st.progress(welfare_score / 100)
            st.write(f"**Calculated Welfare Score:** `{welfare_score}/100` — " + (
                "Animal shows positive valence and healthy rest/rumination patterns." if welfare_score >= 85 else
                "Animal exhibits distress vocalization or prolonged alert standing. Farm staff observation recommended."
            ))

            # Save to history
            save_history_record({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "multimodal",
                "filename": f"Image: {mm_img_file.name if mm_img_file else mm_img_sample} + Audio: {mm_aud_file.name if mm_aud_file else mm_aud_sample}",
                "predicted_class": f"Vision: {vision_res.get('class') if vision_res else 'N/A'} | Audio: {audio_res.get('class') if audio_res else 'N/A'}",
                "confidence": round((vision_res.get('confidence', 0.8) + (audio_res.get('confidence', 0.8) if audio_res else 0.8)) / 2, 4),
                "details": {"vision": vision_res, "audio": audio_res, "welfare_score": welfare_score}
            })


# ============================================================
# TAB 2: VISION BEHAVIOR RECOGNITION
# ============================================================
with tab_vision:
    st.subheader("📸 Cattle Behavior Recognition (ResNet18 CNN)")
    st.write(
        "Classifies 5 fundamental bovine behaviors from single photos, live webcam snapshots, "
        "or multi-second video clips based on the **CBVD-5 Cow Behavior Video Dataset**."
    )

    v_input_mode = st.radio("Select Vision Input Mode:", ["Upload Image/Video", "Live Camera Snapshot", "Quick Test Samples"], horizontal=True)

    target_vision_path = None
    is_video_input = False

    if v_input_mode == "Upload Image/Video":
        uploaded_v = st.file_uploader("Upload Cow Photo or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "webm"])
        if uploaded_v:
            suffix = Path(uploaded_v.name).suffix.lower()
            is_video_input = suffix in [".mp4", ".mov", ".avi", ".webm", ".mkv"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_f:
                tmp_f.write(uploaded_v.read())
                target_vision_path = tmp_f.name

            if is_video_input:
                st.video(target_vision_path)
            else:
                st.image(target_vision_path, caption=uploaded_v.name, use_container_width=True)

    elif v_input_mode == "Live Camera Snapshot":
        cam_snap = st.camera_input("Take a photo of cattle")
        if cam_snap:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_c:
                tmp_c.write(cam_snap.read())
                target_vision_path = tmp_c.name
            st.image(target_vision_path, caption="Live Camera Snapshot", use_container_width=True)

    elif v_input_mode == "Quick Test Samples":
        st.write("Click a sample to test the ResNet18 behavior model immediately:")
        sample_cols = st.columns(6)
        samples = [
            ("sample_standing.jpg", "Standing"),
            ("sample_feeding.jpg", "Feeding"),
            ("sample_drinking.jpg", "Drinking"),
            ("sample_lying.jpg", "Lying"),
            ("sample_rumination.jpg", "Rumination"),
            ("sample_cattle_video.mp4", "5s Video"),
        ]
        for idx, (fn, label) in enumerate(samples):
            with sample_cols[idx]:
                if st.button(f"🔍 {label}", key=f"btn_{fn}"):
                    p = BEHAVIOR_SAMPLES_DIR / fn
                    if p.exists():
                        target_vision_path = str(p)
                        is_video_input = fn.endswith(".mp4")

        if target_vision_path:
            if is_video_input:
                st.video(target_vision_path)
            else:
                st.image(target_vision_path, caption=Path(target_vision_path).name, use_container_width=True)

    if target_vision_path:
        st.markdown("---")
        with st.spinner("Analyzing with ResNet18 Behavior CNN..."):
            t0 = time.time()
            v_result = predict_behavior(target_vision_path)
            t_ms = round((time.time() - t0) * 1000, 1)

        v_class = v_result.get("class") or v_result.get("dominant_class", "standing")
        v_conf = v_result.get("confidence") or v_result.get("dominant_confidence", 0.0)

        st.success(f"Analysis Complete ({t_ms} ms)")

        # Result Display
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f"### Dominant Behavior: **{v_class.upper()}**")
            st.metric("Confidence Score", f"{v_conf*100:.1f}%")
            st.info(v_result.get("description") or v_result.get("summary", ""))

        with c2:
            st.markdown("### Class Probability Distribution")
            probs = v_result.get("probabilities") or v_result.get("activity_breakdown", {})
            if probs:
                df = pd.DataFrame({
                    "Behavior": [k.capitalize() for k in probs.keys()],
                    "Percentage (%)": [(v * 100 if v <= 1.0 else v) for v in probs.values()]
                }).sort_values(by="Percentage (%)", ascending=False)
                st.bar_chart(df.set_index("Behavior"))

        # If Video: Show Frame Timeline
        if is_video_input and "timeline" in v_result and v_result["timeline"]:
            st.markdown("---")
            st.subheader("⏱️ Video Keyframe Timeline & Behavior Transitions")
            st.write(f"Analyzed **{v_result.get('total_frames_analyzed')} frames** across video duration with **{v_result.get('transitions_count')} behavioral transitions**.")

            timeline_cols = st.columns(min(len(v_result["timeline"]), 5))
            for i, frame in enumerate(v_result["timeline"][:5]):
                with timeline_cols[i]:
                    if "thumbnail_base64" in frame:
                        st.image(frame["thumbnail_base64"], caption=f"{frame.get('timestamp_formatted')} — {frame.get('class').upper()}", use_container_width=True)
                    st.caption(f"Conf: {frame.get('confidence')*100:.1f}%")

        # Save history
        save_history_record({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "vision_video" if is_video_input else "vision",
            "filename": Path(target_vision_path).name,
            "predicted_class": v_class,
            "confidence": round(v_conf, 4),
            "details": v_result
        })


# ============================================================
# TAB 3: ACOUSTIC VALENCE ANALYSIS
# ============================================================
with tab_audio:
    st.subheader("🔊 Cattle Vocalization Valence Analysis (AST)")
    st.write(
        "Analyzes bovine vocalizations (mooing) using the **Audio Spectrogram Transformer (AST)** fine-tuned on the "
        "**OpenFarm Ungulate Valence Dataset** to classify emotional valence (*Positive* vs *Negative*) and reject non-cattle noises."
    )

    a_input_mode = st.radio("Select Audio Input Mode:", ["Upload Sound File", "Quick Test Vocal Samples", "Microphone Input Guide"], horizontal=True)

    target_audio_path = None

    if a_input_mode == "Upload Sound File":
        uploaded_a = st.file_uploader("Upload Cattle Vocalization (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg", "flac"])
        if uploaded_a:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_aud:
                tmp_aud.write(uploaded_a.read())
                target_audio_path = tmp_aud.name
            st.audio(target_audio_path)
            st.caption(uploaded_a.name)

    elif a_input_mode == "Quick Test Vocal Samples":
        st.write("Click a real cattle vocal sample to evaluate AST valence inference:")
        sample_a_cols = st.columns(3)
        audio_samples = [
            ("cow_moo_1.wav", "🔊 Calm Low Moo (Sample 1)"),
            ("cow_moo_2.wav", "🔊 Gentle Moo (Sample 2)"),
            ("distress_call.wav", "⚡ Agitated / Distress Call"),
        ]
        for idx, (afn, alabel) in enumerate(audio_samples):
            with sample_a_cols[idx]:
                if st.button(alabel, key=f"abtn_{afn}"):
                    p = AUDIO_SAMPLES_DIR / afn
                    if p.exists():
                        target_audio_path = str(p)

        if target_audio_path:
            st.audio(target_audio_path)
            st.caption(f"Selected: {Path(target_audio_path).name}")

    elif a_input_mode == "Microphone Input Guide":
        st.info("💡 To record live vocalizations directly in real-time with waveform animations, use the standalone web dashboard via `python -m app.web_dashboard` or upload your voice clip above.")

    if target_audio_path:
        st.markdown("---")
        with st.spinner("Processing Acoustic Spectrogram with AST Transformer..."):
            t0 = time.time()
            a_result = predict_audio(target_audio_path)
            t_ms = round((time.time() - t0) * 1000, 1)

        st.success(f"Acoustic Classification Complete ({t_ms} ms)")

        if a_result.get("is_cattle_call", True):
            a_class = a_result.get("class", "Positive")
            a_conf = a_result.get("confidence", 0.0)

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"### Detected Valence: **{a_class.upper()}**")
                st.metric("Valence Confidence", f"{a_conf*100:.1f}%")
                st.info(a_result.get("behavioral_context", ""))

            with col_res2:
                st.markdown("### Valence Class Probabilities")
                probs = a_result.get("probabilities", {})
                if probs:
                    df_a = pd.DataFrame({
                        "Valence State": list(probs.keys()),
                        "Probability (%)": [v * 100 for v in probs.values()]
                    })
                    st.bar_chart(df_a.set_index("Valence State"))

            # Bioacoustic Telemetry
            st.markdown("### 🔬 Bioacoustic Signal Diagnostics")
            metrics = a_result.get("audio_metrics", {})
            if metrics:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Fundamental Pitch (F0)", f"{metrics.get('f0_pitch_hz', 0)} Hz")
                m2.metric("Spectral Centroid", f"{metrics.get('spectral_centroid_hz', 0)} Hz")
                m3.metric("RMS Signal Energy", f"{metrics.get('rms_energy', 0)}")
                m4.metric("Arousal Level", metrics.get("arousal_level", "Normal"))

                st.write(f"**Call Type Classification:** `{metrics.get('call_type_estimate')}` — {metrics.get('call_type_description')}")

            # Save history
            save_history_record({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "audio",
                "filename": Path(target_audio_path).name,
                "predicted_class": a_class,
                "confidence": round(a_conf, 4),
                "details": a_result
            })
        elif a_result.get("is_human_speech"):
            speech_conf = a_result.get("speech_confidence", 95.0)
            st.warning("🗣️ **Human Speaking Detected!**")
            st.write(f"**Classification:** Human voice/speech recognized ({speech_conf:.1f}% probability) by the AudioSet event filter.")
            st.info("💡 **MooTrack Ethology Note:** The system is exclusively calibrated for bovine vocalizations (*Bos taurus* mooing) to calculate livestock emotional valence. Human speech is safely rejected to preserve farm data integrity.")

            # Diagnostics for Human Speech
            st.markdown("### 🔬 Audio Signal Diagnostics")
            metrics = a_result.get("audio_metrics", {})
            if metrics:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pitch (F0)", f"{metrics.get('f0_pitch_hz', 0)} Hz")
                m2.metric("Spectral Centroid", f"{metrics.get('spectral_centroid_hz', 0)} Hz")
                m3.metric("RMS Signal Energy", f"{metrics.get('rms_energy', 0)}")
                m4.metric("Source Type", "Human Vocal Tract")

            save_history_record({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "audio",
                "filename": Path(target_audio_path).name,
                "predicted_class": "Human Speaking",
                "confidence": round(speech_conf / 100.0, 4),
                "details": a_result
            })
        else:
            st.error(f"⚠️ Non-Cattle Sound Rejection: {a_result.get('error')}")


# ============================================================
# TAB 4: DATASET & ARCHITECTURE EXPLORER
# ============================================================
with tab_dataset:
    st.subheader("📊 Datasets & Deep Learning System Architecture")
    st.write("Detailed technical specifications of the datasets and deep learning architectures utilized in MOotrack as featured on the project poster.")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("### 🎙️ Audio Dataset: OpenFarm Ungulate Valence")
        st.markdown("""
        | Attribute | Specification |
        |---|---|
        | **Dataset Name** | OpenFarm Ungulate Valence Dataset |
        | **Source** | Zenodo DOI: `10.5281/zenodo.14636641` / Hugging Face |
        | **Total Samples** | **1,254 audio clips** |
        | **Individual Cattle** | **32 individual cattle** |
        | **Species** | *Bos taurus* (Domestic Cattle) |
        | **Target Classes** | 2 Classes (*Negative* vs *Positive*) |
        | **Negative Valence** | 1,179 samples (94.0%) — Social separation context |
        | **Positive Valence** | 75 samples (6.0%) — Social reunion context |
        """)

        st.caption("Distribution: 94% Negative (Separation Distress), 6% Positive (Reunion/Affiliation)")

    with col_d2:
        st.markdown("### 👁️ Vision Dataset: CBVD-5 Video Dataset")
        st.markdown("""
        | Attribute | Specification |
        |---|---|
        | **Dataset Name** | CBVD-5 (Cow Behavior Video Dataset) |
        | **Source** | Kaggle / Computer Vision Lab |
        | **Total Images** | **206,100 keyframes** |
        | **Video Segments** | **687 video segments** |
        | **Individual Cattle** | **107 individual cattle** |
        | **Target Classes** | 5 Behavioral Classes |
        | **Behaviors** | Standing, Lying, Feeding, Drinking, Rumination |
        """)

        st.caption("5 Behavioral Classes: Standing, Lying down, Feeding/Foraging, Drinking, Rumination")

    st.markdown("---")
    st.subheader("🧠 Deep Learning Architectures")

    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("#### Audio Model: AST (Audio Spectrogram Transformer)")
        st.markdown("""
        - **Pre-trained Backbone:** `MIT/ast-finetuned-audioset-10-10-0.4593`
        - **Fine-Tuning:** Custom 2-class classification head for cattle valence
        - **Input Processing:** 16 kHz Mono Audio $\\rightarrow$ Log-Mel Spectrogram (128 Mel bins, 1024 frames)
        - **Patch Embedding:** $16 \\times 16$ 2D spectrogram patches with positional embeddings
        - **Transformer Encoder:** 12 Multi-Head Self-Attention layers
        - **Output:** Emotional Valence (*Positive* / *Negative*) + Bioacoustic metrics
        """)

    with col_arch2:
        st.markdown("#### Vision Model: ResNet18 CNN")
        st.markdown("""
        - **Pre-trained Backbone:** ResNet18 (ImageNet-1k)
        - **Custom Head:** `nn.Linear(512, 5)` for 5 behavioral classes
        - **Input Processing:** $224 \\times 224$ RGB, Resized & Center-Cropped, ImageNet normalized
        - **Video Pipeline:** 1 fps temporal keyframe extraction + majority vote / transition analysis
        - **Output:** 5 Behavior Classes (Standing, Lying, Feeding, Drinking, Rumination) + Ethological diagnostics
        """)


# ============================================================
# TAB 5: OBSERVATION RECORDS & HISTORY
# ============================================================
with tab_history:
    st.subheader("📋 Observation Records & Audit Log")
    st.write("Review past acoustic, vision, and multimodal cattle monitoring checks.")

    records = load_history_records()

    if records:
        df_records = pd.DataFrame([
            {
                "Timestamp": r.get("timestamp"),
                "Type": r.get("type", "unknown").upper(),
                "Source/File": r.get("filename"),
                "Classification": r.get("predicted_class"),
                "Confidence": f"{r.get('confidence', 0.0)*100:.1f}%",
            }
            for r in records
        ])

        st.dataframe(df_records, use_container_width=True, hide_index=True)

        csv_data = df_records.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Records as CSV",
            data=csv_data,
            file_name="mootrack_observation_history.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("No monitoring checks recorded yet in this session.")


# ============================================================
# TAB 6: PROJECT POSTER & ACADEMIC DOCUMENTATION
# ============================================================
with tab_about:
    st.subheader("ℹ️ Project Poster & Academic Presentation Details")

    st.markdown("""
    ### 🏛️ Sahyadri College of Engineering & Management, Mangaluru
    *(Affiliated to Visvesvaraya Technological University, Belagavi)*  
    **Department of Computer Science and Engineering (Artificial Intelligence & Machine Learning)**  
    **Subject:** Neural Networks and Deep Learning | **Subject Code:** `AM722T2A`

    ---

    ### 👥 Project Team
    1. **Manikanta** (`4SF23CI076`)
    2. **Sai Sudarshan** (`4SF23CI128`)
    3. **Sampath** (`4SF23CI130`)
    4. **Dhruva Shetty** (`4SF23CI147`)

    ---

    ### 1. Problem Statement
    Farmers rely on manual observation to understand cattle behaviour, which is time-consuming and may lead to missed signs of health or behavioural changes. In large herds, it becomes difficult to monitor each animal continuously.  
    **MOotrack** analyses cattle audio and images to identify sound patterns and visible behaviours, and presents the results through a simple web dashboard.

    ### 2. Objectives
    - Develop a system to analyse cattle sounds and behaviour.
    - Identify positive and negative patterns in cattle vocalizations.
    - Recognize five common cattle behaviours from images.
    - Provide prediction and confidence information.
    - Create a simple web dashboard for easy monitoring.

    ### 8. Key Features
    - Cattle sound classification (Positive / Negative)
    - Five behaviour recognition from images (Standing, Lying, Feeding, Drinking, Rumination)
    - Prediction confidence scores
    - Audio and image analysis in one system (Multimodal)
    - Web-based monitoring dashboard
    - Prediction history logging & CSV export

    ### 9. Technology Stack
    `Python` • `PyTorch` • `Hugging Face Transformers` • `Librosa` • `NumPy` • `OpenCV` • `Scikit-learn` • `Streamlit` • `VS Code` • `Google Colab`

    ### 10. Applications
    - Cattle behaviour monitoring
    - Assist farmers and farm staff
    - Identify unusual sound patterns
    - Digital record of observations
    - Support precision livestock farming (PLF)

    ### 11. Limitations & 12. Future Enhancements
    - **Limitations:** Depends on dataset quality, audio/image noise conditions, prototype system (not a clinical veterinary diagnostic tool).
    - **Future Enhancements:** Continuous real-time edge monitoring, mobile app deployment, larger multi-breed datasets, unknown bioacoustic sound detection.

    ### 13. Conclusion
    MOotrack combines cattle sound analysis and image-based behaviour recognition in a single monitoring system. The system provides predictions and confidence information through a simple web interface, which can help in better cattle management.
    """)
