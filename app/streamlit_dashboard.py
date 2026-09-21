"""
MooTrack — Smart Cow Health & Mood Monitor
Simple, AI-powered cow health, mood, and behavior tracking for dairy farmers.
Sahyadri College of Engineering & Management, Mangaluru
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

# Dynamically resolve PROJECT_ROOT whether running from root or app/
_CURR = Path(__file__).resolve()
PROJECT_ROOT = _CURR.parent if (_CURR.parent / "audio_model").exists() else _CURR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_model.predict import predict_audio, compute_acoustic_features
from behavior_model.predict import predict_behavior, predict_behavior_image, predict_behavior_video
from behavior_model.config import BEHAVIOR_CLASSES, BEHAVIOR_DESCRIPTIONS

AUDIO_SAMPLES_DIR = PROJECT_ROOT / "audio_model" / "test_samples"
BEHAVIOR_SAMPLES_DIR = PROJECT_ROOT / "behavior_model" / "test_samples"
HISTORY_FILE = PROJECT_ROOT / "app" / "prediction_history.json"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
LOGO_PATH = STATIC_DIR / "logo_clean.png" if (STATIC_DIR / "logo_clean.png").exists() else STATIC_DIR / "logo.png"

# Streamlit Page Config
st.set_page_config(
    page_title="MooTrack — Smart Cow Health & Mood Monitor",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Farmer-Friendly Clean CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #166534 0%, #14532D 100%);
        padding: 24px 30px;
        border-radius: 20px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(20, 83, 45, 0.15);
    }
    .header-sub {
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #A3E635;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 4px;
        color: #FFFFFF;
    }
    .header-desc {
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.85);
    }
    .farmer-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #E2E8E0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 16px;
    }
    .status-box-pos {
        background: #F0FDF4;
        border: 2px solid #86EFAC;
        border-radius: 16px;
        padding: 20px;
        margin-top: 14px;
    }
    .status-box-neg {
        background: #FFF1F2;
        border: 2px solid #FDA4AF;
        border-radius: 16px;
        padding: 20px;
        margin-top: 14px;
    }
    .status-box-speech {
        background: #FEFCE8;
        border: 2px solid #FDE047;
        border-radius: 16px;
        padding: 20px;
        margin-top: 14px;
    }
    .status-box-behavior {
        background: #F8FAFC;
        border: 2px solid #CBD5E1;
        border-radius: 16px;
        padding: 20px;
        margin-top: 14px;
    }
    .advice-box {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 12px;
        border: 1px solid rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions for Prediction History
def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history[:100], f, indent=2)
    except Exception as e:
        print(f"[History Error]: {e}")

# Header Banner
st.markdown("""
<div class="main-header">
    <div class="header-sub">Sahyadri College of Engineering & Management • AIML</div>
    <div class="header-title">🐄 MooTrack — Smart Cow Health & Mood Monitor</div>
    <div class="header-desc">Simple, AI-powered cow mood and behavior assistant for dairy & livestock farmers</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=160)
    st.markdown("### 🌾 Barn Assistant Quick Menu")
    st.markdown("Use MooTrack to check your cow's sound and camera feed in seconds.")
    st.info("💡 **Farmer Tip:** Cows chew cud 7–9 hours a day. High-pitched moos usually mean empty water or distress.")
    
    st.markdown("---")
    st.markdown("👨‍🌾 **Quick Links:**")
    st.markdown("- [Standalone Web App (Port 8000)](http://localhost:8000)")
    st.caption("MooTrack Cattle Monitoring System v2.0")

# 4 Main Farmer Tabs
tab_audio, tab_vision, tab_welfare, tab_history = st.tabs([
    "🎙️ 1. Cow Moo & Voice Check",
    "📷 2. Cow Activity Camera",
    "📋 3. Daily Cow Care Guide",
    "📜 4. Past Records & History"
])

# =======================================================
# TAB 1: COW MOO & VOICE CHECK
# =======================================================
with tab_audio:
    st.markdown("### 🎙️ Listen to Your Cow's Moo")
    st.write("Record your cow's sound or upload an audio file to check if your cow is calm and happy or in distress.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Option A: Record Cow Sound")
        audio_record = st.audio_input("🎤 Record Cow Moo (Tap to record)")

        st.markdown("#### Option B: Upload Sound File")
        audio_file = st.file_uploader("Upload Cow Audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a", "aac", "ogg"])

    with col2:
        st.markdown("#### Option C: Try Quick Sample Moos")
        sample_choice = st.selectbox(
            "Select a pre-recorded barn sound to test:",
            ["None", "🟢 Happy / Calm Contact Moo", "🔴 High-Pitched Distress Moo"]
        )

    # Process Audio Input
    audio_to_process = None
    source_name = "Live Check"

    if audio_record is not None:
        audio_to_process = audio_record.read()
        source_name = "Live Microphone Recording"
    elif audio_file is not None:
        audio_to_process = audio_file.read()
        source_name = audio_file.name
    elif sample_choice == "🟢 Happy / Calm Contact Moo":
        p = AUDIO_SAMPLES_DIR / "cattle_positive_sample.wav"
        if p.exists():
            with open(p, "rb") as f:
                audio_to_process = f.read()
            source_name = "cattle_positive_sample.wav"
    elif sample_choice == "🔴 High-Pitched Distress Moo":
        p = AUDIO_SAMPLES_DIR / "cattle_negative_sample.wav"
        if p.exists():
            with open(p, "rb") as f:
                audio_to_process = f.read()
            source_name = "cattle_negative_sample.wav"

    if audio_to_process is not None:
        st.markdown("---")
        with st.spinner("Analyzing cow vocalization..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_to_process)
                tmp_path = tmp.name

            try:
                res = predict_audio(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        is_human = res.get("is_human_speech", False)
        is_cattle = res.get("is_cattle_call", True) and not is_human
        pred_class = res.get("class", "Unknown")
        conf = round(float(res.get("confidence", 0.9)) * 100)

        # Save to history
        save_history_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "audio",
            "filename": source_name,
            "predicted_class": pred_class,
            "confidence": res.get("confidence", 0.0),
            "details": res,
        })

        if is_human:
            st.markdown(f"""
            <div class="status-box-speech">
                <h3>🗣️ Human Voice Detected ({conf}% Speech Certainty)</h3>
                <p>Human speech was recognized instead of a bovine call. Please point your mic towards your cow and record her moo.</p>
            </div>
            """, unsafe_allow_html=True)
        elif is_cattle and pred_class == "Positive":
            st.markdown(f"""
            <div class="status-box-pos">
                <h3>🟢 Cow is Calm & Happy (Positive Mood) — {conf}% Certainty</h3>
                <div class="advice-box">
                    <strong>💡 Farmer Guidance:</strong>
                    <p>Calm, low contact moo detected. Your cow is feeling comfortable, content with her herdmates, or is giving a gentle contact call. Good herd comfort!</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif is_cattle and pred_class == "Negative":
            st.markdown(f"""
            <div class="status-box-neg">
                <h3>🔴 Cow is Distressed / Needs Attention — {conf}% Certainty</h3>
                <div class="advice-box">
                    <strong>💡 Action Needed for Farmer:</strong>
                    <p>High-pitched distress call detected! Please check: 1) Is the water trough empty? 2) Is feed bunk low? 3) Is the cow isolated from the herd? 4) Is she in heat (estrus) or in pain?</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ {res.get('error', 'Audio was too quiet or background noise. Please record closer to the cow.')}")

        # Optional Technical Bioacoustics
        if res.get("audio_metrics"):
            with st.expander("🔬 View Technical Sound Numbers (Pitch & Frequency)"):
                m = res["audio_metrics"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Voice Pitch (f₀)", f"{m.get('f0_pitch_hz', 0)} Hz")
                c2.metric("Sound Frequency", f"{m.get('spectral_centroid_hz', 0)} Hz")
                c3.metric("Loudness (RMS)", f"{m.get('rms_energy', 0)}")
                c4.metric("Call Type", m.get('call_type_estimate', 'Bovine').replace(' Call', ''))

# =======================================================
# TAB 2: COW ACTIVITY CAMERA
# =======================================================
with tab_vision:
    st.markdown("### 📷 Check What Your Cow is Doing")
    st.write("Take a live photo or upload a picture/video to check if your cow is drinking, feeding, resting, chewing cud, or standing.")

    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        st.markdown("#### Option A: Take a Live Photo")
        camera_photo = st.camera_input("📸 Take Cow Photo with Camera")

        st.markdown("#### Option B: Upload Photo or Video")
        vision_file = st.file_uploader("Upload Picture or Video (.jpg, .png, .mp4)", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

    with v_col2:
        st.markdown("#### Option C: Try Barn Sample Pictures")
        barn_sample = st.selectbox(
            "Select a sample barn picture to test:",
            ["None", "💧 Drinking Water", "🌿 Eating / Feeding", "🛌 Resting / Lying Down", "🌾 Chewing Cud (Rumination)", "🚶 Standing Up"]
        )

    vision_to_process = None
    v_source_name = "Live Photo"
    is_vid = False

    if camera_photo is not None:
        vision_to_process = camera_photo.read()
        v_source_name = "Live Camera Photo"
    elif vision_file is not None:
        vision_to_process = vision_file.read()
        v_source_name = vision_file.name
        is_vid = Path(vision_file.name).suffix.lower() in [".mp4", ".mov", ".avi", ".webm"]
    elif barn_sample != "None":
        sample_map = {
            "💧 Drinking Water": "sample_drinking.jpg",
            "🌿 Eating / Feeding": "sample_feeding.jpg",
            "🛌 Resting / Lying Down": "sample_lying.jpg",
            "🌾 Chewing Cud (Rumination)": "sample_rumination.jpg",
            "🚶 Standing Up": "sample_standing.jpg",
        }
        fn = sample_map.get(barn_sample)
        p = BEHAVIOR_SAMPLES_DIR / fn
        if p.exists():
            with open(p, "rb") as f:
                vision_to_process = f.read()
            v_source_name = fn

    if vision_to_process is not None:
        st.markdown("---")
        with st.spinner("Analyzing cow behavior with ResNet18..."):
            ext = ".mp4" if is_vid else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(vision_to_process)
                tmp_path = tmp.name

            try:
                v_res = predict_behavior(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        b_cls = (v_res.get("class") or v_res.get("dominant_class") or "standing").lower()
        b_conf = round(float(v_res.get("confidence") or v_res.get("dominant_confidence") or 0.9) * 100)

        # Save to history
        save_history_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "vision_video" if is_vid else "vision",
            "filename": v_source_name,
            "predicted_class": b_cls,
            "confidence": v_res.get("confidence", 0.0),
            "details": v_res,
        })

        behavior_map = {
            "drinking": ("💧 Drinking Water", "Cow is drinking at the water trough. Good hydration! Ensure water is clean, cool, and plentiful (cows need 60-120L daily for high milk yield)."),
            "feeding": ("🌿 Feeding / Eating", "Cow is eating forage or silage from the feed bunk. Active eating is a vital indicator of healthy appetite and dry matter intake."),
            "lying": ("🛌 Resting / Lying Down", "Cow is resting comfortably in the stall. (Dairy cows require 10-14 hours of rest daily. Each additional hour of rest increases milk yield by ~1-1.5 kg)."),
            "rumination": ("🌾 Chewing Cud (Rumination)", "Cow is actively chewing cud. Excellent sign! Rumination confirms healthy rumen bacteria, good digestion, and high cow comfort."),
            "standing": ("🚶 Standing Alert", "Cow is standing upright in an alert posture. Standard baseline posture observed throughout daylight hours."),
        }

        b_title, b_advice = behavior_map.get(b_cls, (b_cls.capitalize(), v_res.get("description", "Observed cow behavior.")))

        st.markdown(f"""
        <div class="status-box-behavior">
            <h3>{b_title} — {b_conf}% Certainty</h3>
            <div class="advice-box">
                <strong>💡 What this means for your cow:</strong>
                <p>{b_advice}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Video time budget breakdown if video
        if v_res.get("activity_breakdown"):
            st.markdown("#### ⏱️ Video Activity Breakdown")
            st.write(v_res["activity_breakdown"])

# =======================================================
# TAB 3: DAILY COW CARE GUIDE
# =======================================================
with tab_welfare:
    st.markdown("### 📋 Practical Cow Health & Care Guide for Farmers")
    st.write("Essential daily benchmarks every dairy farmer should check in the barn.")

    st.markdown("""
    <div class="farmer-card">
        <h4>🌾 1. Rumination & Cud Chewing (Target: 7 - 9 Hours Daily)</h4>
        <p>When resting, at least 50–60% of cows lying down should be actively chewing their cud (40–70 chews per bolus). If cud chewing drops, check fiber length and silage quality.</p>
    </div>

    <div class="farmer-card">
        <h4>🛌 2. Resting & Stall Comfort (Target: 10 - 14 Hours Daily)</h4>
        <p>Dairy cows produce peak milk when lying down due to increased blood flow to the udder (+30%). Ensure dry, clean bedding (sand, sawdust, or rubber mats) to prevent mastitis and lameness.</p>
    </div>

    <div class="farmer-card">
        <h4>💧 3. Fresh Clean Water (Target: 60 - 120 Liters Daily)</h4>
        <p>Water drives milk production (milk is 87% water). Ensure troughs are clean, odor-free, accessible, and positioned close to the feed alley.</p>
    </div>

    <div class="farmer-card">
        <h4>🎙️ 4. High-Pitched Distress Calls (Mooing Alerts)</h4>
        <p>Frequent high-frequency moos usually mean: empty water trough, hunger, estrus (heat cycle), pain/illness, or separation from the herd. Check the pen immediately.</p>
    </div>
    """, unsafe_allow_html=True)

# =======================================================
# TAB 4: RECENT RECORDS & HISTORY
# =======================================================
with tab_history:
    st.markdown("### 📜 Past Cow Checks & History")
    history_data = load_history()

    if not history_data:
        st.info("No checks recorded yet. Record a cow moo or photo to start building your records!")
    else:
        df_history = pd.DataFrame([
            {
                "Time": h.get("timestamp"),
                "Type": "🎙️ Sound / Moo" if h.get("type") == "audio" else "📷 Camera / Photo",
                "Identified Status": h.get("predicted_class"),
                "Certainty": f"{round(float(h.get('confidence', 0))*100)}%",
                "File / Source": h.get("filename", "Live Check")
            }
            for h in history_data
        ])
        st.dataframe(df_history, use_container_width=True)

        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV History Report",
            data=csv_data,
            file_name="mootrack_cow_records.csv",
            mime="text/csv",
        )
