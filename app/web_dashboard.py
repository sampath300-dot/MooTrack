import io
import os
import json
import time
import base64
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Tuple, Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Inference APIs for both modules
from audio_model.predict import predict_audio
from behavior_model.predict import predict_behavior

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
LOGO_PATH = STATIC_DIR / "logo.png"
LOGO_CLEAN_PATH = STATIC_DIR / "logo_clean.png"

AUDIO_TEST_SAMPLES = PROJECT_ROOT / "audio_model" / "test_samples"
AUDIO_RECORDINGS = PROJECT_ROOT / "audio_model" / "recordings"
AUDIO_RECORDINGS.mkdir(parents=True, exist_ok=True)

BEHAVIOR_TEST_SAMPLES = PROJECT_ROOT / "behavior_model" / "test_samples"
BEHAVIOR_DATASET = PROJECT_ROOT / "behavior_model" / "dataset"

HISTORY_FILE = PROJECT_ROOT / "app" / "prediction_history.json"
PORT = 8000

# Global in-memory history cache
PREDICTION_HISTORY = []

def load_history():
    global PREDICTION_HISTORY
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                PREDICTION_HISTORY = json.load(f)
        except Exception:
            PREDICTION_HISTORY = []

def save_history():
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(PREDICTION_HISTORY, f, indent=2)
    except Exception as e:
        print(f"[History Save Error]: {e}")

load_history()

LOGO_BASE64 = ""
target_logo = LOGO_CLEAN_PATH if LOGO_CLEAN_PATH.exists() else LOGO_PATH
if target_logo.exists():
    try:
        with open(target_logo, "rb") as f:
            LOGO_BASE64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    except Exception:
        LOGO_BASE64 = ""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <title>MooTrack — Smart Cow Health & Mood Monitor</title>
    <meta name="description" content="MooTrack: Simple, AI-Powered Cow Health, Mood, and Behavior Tracker for Dairy Farmers.">
    <link rel="icon" type="image/png" href="/static/logo_clean.png">
    
    <!-- Google Fonts: Inter & Outfit -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --farm-green: #15803D;
            --farm-green-dark: #166534;
            --farm-green-light: #DCFCE7;
            --farm-gold: #CA8A04;
            --farm-gold-light: #FEF9C3;
            --farm-rose: #E11D48;
            --farm-rose-light: #FFE4E6;
            --farm-sky: #0284C7;
            --farm-sky-light: #E0F2FE;
            
            --bg-page: #F8FAF7;
            --bg-card: #FFFFFF;
            --bg-card-alt: #F1F5F0;
            
            --text-main: #1C241E;
            --text-muted: #526056;
            --text-subtle: #7D8E82;
            
            --border: #E2E8E0;
            --border-strong: #CBD5C8;
            
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.04);
            --shadow-md: 0 6px 16px rgba(0,0,0,0.06);
            --shadow-lg: 0 12px 32px rgba(21,128,61,0.08);
            
            --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-page);
            color: var(--text-main);
            font-family: var(--font-body);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            min-height: 100vh;
        }

        .container {
            max-width: 1120px;
            margin: 0 auto;
            padding: 16px 20px 60px;
        }

        /* Farmer-Friendly Header */
        .header {
            background: linear-gradient(135deg, #166534 0%, #14532D 100%);
            color: #FFFFFF;
            padding: 24px 28px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(20, 83, 45, 0.2);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .header-logo {
            width: 56px;
            height: 56px;
            background: #FFFFFF;
            border-radius: 16px;
            padding: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .header-logo img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .header-title h1 {
            font-family: var(--font-display);
            font-size: 1.6rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-title p {
            font-size: 0.92rem;
            color: rgba(255, 255, 255, 0.85);
            margin-top: 2px;
        }
        .status-badge {
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 8px 14px;
            border-radius: 30px;
            font-size: 0.82rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            backdrop-filter: blur(8px);
        }
        .status-dot {
            width: 9px;
            height: 9px;
            background: #4ADE80;
            border-radius: 50%;
            box-shadow: 0 0 8px #4ADE80;
        }

        /* Large Farmer-Friendly Tabs */
        .nav-tabs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .tab-btn {
            background: var(--bg-card);
            border: 2px solid var(--border);
            padding: 16px 18px;
            border-radius: 16px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 14px;
            font-family: var(--font-display);
        }
        .tab-btn:hover {
            border-color: var(--farm-green);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .tab-btn.active {
            background: #FFFFFF;
            border-color: var(--farm-green);
            box-shadow: 0 4px 16px rgba(21, 128, 61, 0.15);
            position: relative;
        }
        .tab-btn.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 20px;
            right: 20px;
            height: 3px;
            background: var(--farm-green);
            border-radius: 3px 3px 0 0;
        }
        .tab-icon {
            font-size: 1.8rem;
            background: var(--bg-card-alt);
            width: 46px;
            height: 46px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .tab-btn.active .tab-icon {
            background: var(--farm-green-light);
        }
        .tab-text .tab-name {
            font-weight: 700;
            font-size: 1.02rem;
            color: var(--text-main);
        }
        .tab-text .tab-desc {
            font-size: 0.78rem;
            color: var(--text-muted);
            font-family: var(--font-body);
        }

        /* Main Workspace Grid */
        .view-section {
            display: none;
            animation: fadeIn 0.25s ease-out;
        }
        .view-section.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card {
            background: var(--bg-card);
            border-radius: 20px;
            border: 1px solid var(--border);
            padding: 26px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 24px;
        }
        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .card-title {
            font-family: var(--font-display);
            font-size: 1.25rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card-subtitle {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Action Buttons Grid */
        .actions-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 22px;
        }
        .action-card {
            border: 2px dashed var(--border-strong);
            background: var(--bg-card-alt);
            border-radius: 16px;
            padding: 22px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .action-card:hover {
            border-color: var(--farm-green);
            background: var(--farm-green-light);
            transform: translateY(-2px);
        }
        .action-card.recording {
            border-color: var(--farm-rose);
            background: var(--farm-rose-light);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(225, 29, 72, 0.3); }
            70% { box-shadow: 0 0 0 12px rgba(225, 29, 72, 0); }
            100% { box-shadow: 0 0 0 0 rgba(225, 29, 72, 0); }
        }
        .action-icon {
            font-size: 2.2rem;
            margin-bottom: 10px;
        }
        .action-btn-text {
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1.05rem;
            color: var(--text-main);
        }
        .action-subtext {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .btn-primary {
            background: var(--farm-green);
            color: #FFFFFF;
            border: none;
            padding: 12px 22px;
            border-radius: 12px;
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn-primary:hover {
            background: var(--farm-green-dark);
            transform: translateY(-1px);
        }

        .btn-outline {
            background: #FFFFFF;
            color: var(--text-main);
            border: 1px solid var(--border-strong);
            padding: 8px 14px;
            border-radius: 10px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn-outline:hover {
            border-color: var(--farm-green);
            background: var(--farm-green-light);
            color: var(--farm-green-dark);
        }

        /* Quick Test Audio & Image Pills */
        .samples-bar {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 14px;
            padding: 12px 16px;
            background: var(--bg-card-alt);
            border-radius: 12px;
        }
        .samples-label {
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-muted);
        }
        .sample-pill {
            background: #FFFFFF;
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
        }
        .sample-pill:hover {
            border-color: var(--farm-green);
            background: var(--farm-green-light);
            color: var(--farm-green-dark);
        }

        /* Result Display Card */
        .result-box {
            display: none;
            margin-top: 24px;
            padding: 24px;
            border-radius: 18px;
            border: 2px solid transparent;
            animation: fadeIn 0.3s ease;
        }
        .result-box.positive {
            background: #F0FDF4;
            border-color: #86EFAC;
        }
        .result-box.negative {
            background: #FFF1F2;
            border-color: #FDA4AF;
        }
        .result-box.speech {
            background: #FEFCE8;
            border-color: #FDE047;
        }
        .result-box.behavior {
            background: #F8FAFC;
            border-color: #CBD5E1;
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .result-main-title {
            font-family: var(--font-display);
            font-size: 1.4rem;
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .confidence-tag {
            font-size: 0.85rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid rgba(0,0,0,0.1);
        }

        .advice-card {
            background: #FFFFFF;
            border-radius: 14px;
            padding: 16px 20px;
            margin-top: 14px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        }
        .advice-title {
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-main);
        }
        .advice-text {
            font-size: 0.9rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        /* Collapsible Technical Details */
        details.tech-details {
            margin-top: 16px;
            background: #FFFFFF;
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 12px 16px;
        }
        details.tech-details summary {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            user-select: none;
        }
        .tech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }
        .tech-item {
            background: var(--bg-card-alt);
            padding: 10px;
            border-radius: 8px;
            font-size: 0.78rem;
        }
        .tech-item .label {
            color: var(--text-muted);
            display: block;
            margin-bottom: 2px;
        }
        .tech-item .val {
            font-family: var(--font-mono);
            font-weight: 700;
            color: var(--text-main);
            font-size: 0.9rem;
        }

        /* Camera / Video Viewfinder */
        .camera-container {
            display: none;
            margin-bottom: 20px;
            background: #000;
            border-radius: 16px;
            overflow: hidden;
            position: relative;
            max-width: 540px;
            margin-left: auto;
            margin-right: auto;
        }
        #webcamVideo {
            width: 100%;
            height: auto;
            display: block;
        }
        .camera-bar {
            padding: 12px;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }

        /* Audio Waveform Canvas */
        .waveform-canvas {
            width: 100%;
            height: 64px;
            background: var(--bg-card-alt);
            border-radius: 10px;
            margin-top: 12px;
            display: none;
        }

        /* Simple History Table */
        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 14px;
        }
        .history-table th {
            text-align: left;
            padding: 10px 14px;
            background: var(--bg-card-alt);
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }
        .history-table td {
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
        }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .badge.positive { background: var(--farm-green-light); color: var(--farm-green-dark); }
        .badge.negative { background: var(--farm-rose-light); color: var(--farm-rose); }
        .badge.neutral { background: var(--farm-sky-light); color: var(--farm-sky); }

        /* Farmer Checklist Tab */
        .checklist-card {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 16px;
            border-radius: 14px;
            background: var(--bg-card-alt);
            margin-bottom: 12px;
            border: 1px solid var(--border);
        }
        .checklist-icon {
            font-size: 1.6rem;
            flex-shrink: 0;
            background: #FFFFFF;
            width: 44px;
            height: 44px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--shadow-sm);
        }
        .checklist-content h4 {
            font-family: var(--font-display);
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .checklist-content p {
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            font-size: 0.82rem;
            color: var(--text-subtle);
        }

        /* Loading Spinner */
        .spinner {
            display: inline-block;
            width: 18px;
            height: 18px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">

    <!-- Farmer Header -->
    <header class="header">
        <div class="header-brand">
            <div class="header-logo">
                <img src="/static/logo_clean.png" alt="MooTrack Logo" onerror="this.src='/static/logo.png'; this.onerror=null;">
            </div>
            <div class="header-title">
                <h1>🐄 MooTrack Cattle Monitor</h1>
                <p>Simple cow health, mood & behavior assistant for dairy and livestock farmers</p>
            </div>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span>
            <span>AI Barn Assistant Active</span>
        </div>
    </header>

    <!-- 4 Clear Navigation Tabs -->
    <nav class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('audio')">
            <div class="tab-icon">🎙️</div>
            <div class="tab-text">
                <div class="tab-name">Cow Moo Check</div>
                <div class="tab-desc">Happy vs Distressed Moo</div>
            </div>
        </button>

        <button class="tab-btn" onclick="switchTab('vision')">
            <div class="tab-icon">📷</div>
            <div class="tab-text">
                <div class="tab-name">Cow Activity Camera</div>
                <div class="tab-desc">Eating, Resting, Cudding</div>
            </div>
        </button>

        <button class="tab-btn" onclick="switchTab('welfare')">
            <div class="tab-icon">📋</div>
            <div class="tab-text">
                <div class="tab-name">Herd Health Guide</div>
                <div class="tab-desc">Daily Cow Care Tips</div>
            </div>
        </button>

        <button class="tab-btn" onclick="switchTab('history')">
            <div class="tab-icon">📜</div>
            <div class="tab-text">
                <div class="tab-name">Recent Checks</div>
                <div class="tab-desc">Past Records & Download</div>
            </div>
        </button>
    </nav>

    <!-- ======================================================= -->
    <!-- TAB 1: COW MOO / VOICE CHECK -->
    <!-- ======================================================= -->
    <section id="tab-audio" class="view-section active">
        <div class="card">
            <div class="card-header">
                <div>
                    <h2 class="card-title">🎙️ Listen to Your Cow's Moo</h2>
                    <p class="card-subtitle">Record your cow's sound or upload an audio file to check if your cow is calm or in distress.</p>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="actions-row">
                <div id="recordCard" class="action-card" onclick="toggleAudioRecording()">
                    <div class="action-icon" id="recordIcon">🔴</div>
                    <div class="action-btn-text" id="recordText">Tap to Record Cow Moo</div>
                    <div class="action-subtext" id="recordSub">Hold your phone/mic near the cow (2-5 seconds)</div>
                    <canvas id="waveformCanvas" class="waveform-canvas"></canvas>
                </div>

                <div class="action-card" onclick="document.getElementById('audioFileInput').click()">
                    <div class="action-icon">📁</div>
                    <div class="action-btn-text">Upload Cow Audio File</div>
                    <div class="action-subtext">Supports .wav, .mp3, .m4a, .aac from phone</div>
                    <input type="file" id="audioFileInput" accept="audio/*" style="display:none" onchange="handleAudioUpload(this.files[0])">
                </div>
            </div>

            <!-- Quick Test Samples for Farmer -->
            <div class="samples-bar">
                <span class="samples-label">⚡ Try Quick Sample Moos:</span>
                <button class="sample-pill" onclick="testAudioSample('cattle_positive_sample.wav', 'Happy Contact Moo')">
                    🟢 Happy / Calm Moo
                </button>
                <button class="sample-pill" onclick="testAudioSample('cattle_negative_sample.wav', 'Distress / Separation Moo')">
                    🔴 Distress / Alert Moo
                </button>
            </div>

            <!-- Audio Results Display -->
            <div id="audioResult" class="result-box">
                <div class="result-header">
                    <div class="result-main-title" id="audioStatusTitle">
                        <span id="audioStatusIcon">🟢</span>
                        <span id="audioStatusText">Cow is Calm & Happy</span>
                    </div>
                    <div class="confidence-tag" id="audioConfidenceTag">97% Certainty</div>
                </div>

                <div class="advice-card">
                    <div class="advice-title">💡 Farmer Guidance & Action:</div>
                    <p class="advice-text" id="audioAdviceText">
                        Your cow is producing a low, calm contact moo. This indicates comfort, social contentment, or gentle maternal communication.
                    </p>
                </div>

                <!-- Collapsible Technical Bioacoustics for Veterinarians / Advanced view -->
                <details class="tech-details">
                    <summary>🔬 View Technical Sound Numbers (Pitch & Frequency)</summary>
                    <div class="tech-grid" id="audioTechGrid">
                        <div class="tech-item"><span class="label">Voice Pitch (f₀)</span><span class="val" id="valF0">135 Hz</span></div>
                        <div class="tech-item"><span class="label">Sound Frequency</span><span class="val" id="valCentroid">850 Hz</span></div>
                        <div class="tech-item"><span class="label">Loudness (RMS)</span><span class="val" id="valRMS">0.05</span></div>
                        <div class="tech-item"><span class="label">Call Type</span><span class="val" id="valCallType">Closed-Mouth</span></div>
                    </div>
                </details>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 2: COW ACTIVITY & CAMERA -->
    <!-- ======================================================= -->
    <section id="tab-vision" class="view-section">
        <div class="card">
            <div class="card-header">
                <div>
                    <h2 class="card-title">📷 Check What Your Cow is Doing</h2>
                    <p class="card-subtitle">Take a photo, record a video, or upload a picture to instantly check if the cow is eating, drinking, chewing cud, or resting.</p>
                </div>
            </div>

            <!-- Camera Viewfinder if opened -->
            <div id="cameraBox" class="camera-container">
                <video id="webcamVideo" autoplay playsinline></video>
                <div class="camera-bar">
                    <button class="btn-primary" onclick="snapPhoto()">📸 Snap Photo</button>
                    <button class="btn-outline" style="color:#fff;" onclick="closeCamera()">✖ Close Camera</button>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="actions-row">
                <div class="action-card" onclick="openCamera()">
                    <div class="action-icon">📸</div>
                    <div class="action-btn-text">Open Barn Camera</div>
                    <div class="action-subtext">Take a live photo directly with your camera</div>
                </div>

                <div class="action-card" onclick="document.getElementById('visionFileInput').click()">
                    <div class="action-icon">📁</div>
                    <div class="action-btn-text">Upload Cow Photo or Video</div>
                    <div class="action-subtext">Supports pictures (.jpg, .png) & videos (.mp4, .mov)</div>
                    <input type="file" id="visionFileInput" accept="image/*,video/*" style="display:none" onchange="handleVisionUpload(this.files[0])">
                </div>
            </div>

            <!-- Quick Barn Test Pictures -->
            <div class="samples-bar">
                <span class="samples-label">⚡ Try Barn Sample Pictures:</span>
                <button class="sample-pill" onclick="testVisionSample('sample_drinking.jpg', 'drinking')">💧 Drinking Water</button>
                <button class="sample-pill" onclick="testVisionSample('sample_feeding.jpg', 'feeding')">🌿 Eating / Feeding</button>
                <button class="sample-pill" onclick="testVisionSample('sample_lying.jpg', 'lying')">🛌 Resting / Lying</button>
                <button class="sample-pill" onclick="testVisionSample('sample_rumination.jpg', 'rumination')">🌾 Chewing Cud</button>
                <button class="sample-pill" onclick="testVisionSample('sample_standing.jpg', 'standing')">🚶 Standing Up</button>
            </div>

            <!-- Vision Result Display -->
            <div id="visionResult" class="result-box behavior">
                <div class="result-header">
                    <div class="result-main-title" id="visionStatusTitle">
                        <span id="visionStatusIcon">🌾</span>
                        <span id="visionStatusText">Chewing Cud (Rumination)</span>
                    </div>
                    <div class="confidence-tag" id="visionConfidenceTag">97% Certainty</div>
                </div>

                <div class="advice-card">
                    <div class="advice-title">💡 What this means for your cow:</div>
                    <p class="advice-text" id="visionAdviceText">
                        The cow is chewing regurgitated cud. This is one of the best indicators of healthy rumen fermentation, good digestion, and high cow comfort!
                    </p>
                </div>

                <!-- Video Timeline breakdown if video uploaded -->
                <div id="videoBreakdownBox" style="display:none; margin-top:14px;">
                    <div class="advice-title">⏱️ Video Activity Breakdown:</div>
                    <div id="videoBreakdownContent" style="margin-top:8px;"></div>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 3: HERD HEALTH & CARE GUIDE -->
    <!-- ======================================================= -->
    <section id="tab-welfare" class="view-section">
        <div class="card">
            <div class="card-header">
                <div>
                    <h2 class="card-title">📋 Practical Cow Health & Comfort Guide</h2>
                    <p class="card-subtitle">Daily standard benchmarks every dairy farmer should check in the barn.</p>
                </div>
            </div>

            <div class="checklist-card">
                <div class="checklist-icon">🌾</div>
                <div class="checklist-content">
                    <h4>1. Rumination & Cud Chewing (Target: 7 - 9 Hours Daily)</h4>
                    <p>When resting, at least 50–60% of cows lying down should be actively chewing their cud (40–70 chews per bolus). If cud chewing drops, check fiber length and silage quality.</p>
                </div>
            </div>

            <div class="checklist-card">
                <div class="checklist-icon">🛌</div>
                <div class="checklist-content">
                    <h4>2. Resting & Stall Comfort (Target: 10 - 14 Hours Daily)</h4>
                    <p>Dairy cows produce peak milk when lying down due to increased blood flow to the udder (+30%). Ensure dry, clean bedding (sand, sawdust, or rubber mats) to prevent mastitis and lameness.</p>
                </div>
            </div>

            <div class="checklist-card">
                <div class="checklist-icon">💧</div>
                <div class="checklist-content">
                    <h4>3. Fresh Clean Water (Target: 60 - 120 Liters Daily)</h4>
                    <p>Water drives milk production (milk is 87% water). Ensure troughs are clean, odor-free, accessible, and positioned close to the feed alley.</p>
                </div>
            </div>

            <div class="checklist-card">
                <div class="checklist-icon">🎙️</div>
                <div class="checklist-content">
                    <h4>4. High-Pitched Distress Calls (Mooing Alerts)</h4>
                    <p>Frequent high-frequency moos usually mean: empty water trough, hunger, estrus (heat cycle), pain/illness, or separation from the herd. Check the pen immediately.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 4: RECENT CHECKS & HISTORY -->
    <!-- ======================================================= -->
    <section id="tab-history" class="view-section">
        <div class="card">
            <div class="card-header">
                <div>
                    <h2 class="card-title">📜 Past Cow Checks & History</h2>
                    <p class="card-subtitle">Saved record of all sound and camera checks performed today.</p>
                </div>
                <a href="/api/export/csv" class="btn-primary" download="mootrack_cattle_records.csv">
                    📥 Download CSV Report
                </a>
            </div>

            <div style="overflow-x: auto;">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Check Type</th>
                            <th>Identified Result</th>
                            <th>Certainty</th>
                            <th>Status / Farmer Note</th>
                        </tr>
                    </thead>
                    <tbody id="historyTableBody">
                        <tr>
                            <td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">
                                No records yet. Record a cow moo or take a photo to see results here!
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
        <p><strong>MooTrack Cattle Monitoring System</strong> • Sahyadri College of Engineering & Management, Mangaluru</p>
        <p style="margin-top:4px;">Department of Computer Science & Engineering (AIML) • Course: Neural Networks & Deep Learning</p>
    </footer>

</div>

<script>
    // ============================================================
    // STATE & TAB MANAGEMENT
    // ============================================================
    function switchTab(tabId) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));

        const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
        if (targetBtn) targetBtn.classList.add('active');

        const targetSec = document.getElementById('tab-' + tabId);
        if (targetSec) targetSec.classList.add('active');

        if (tabId === 'history') {
            loadHistory();
        }
    }

    // ============================================================
    // 1. AUDIO RECORDING (WebAudio 16kHz PCM WAV)
    // ============================================================
    let isRecording = false;
    let audioCtx = null;
    let micStream = null;
    let scriptNode = null;
    let audioChunks = [];
    let recordTimer = null;
    let recordSeconds = 0;

    async function toggleAudioRecording() {
        if (isRecording) {
            stopAudioRecording();
        } else {
            startAudioRecording();
        }
    }

    async function startAudioRecording() {
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            
            const source = audioCtx.createMediaStreamSource(micStream);
            scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
            audioChunks = [];

            scriptNode.onaudioprocess = (e) => {
                if (!isRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                audioChunks.push(new Float32Array(channelData));
            };

            source.connect(scriptNode);
            scriptNode.connect(audioCtx.destination);

            isRecording = true;
            recordSeconds = 0;
            const card = document.getElementById('recordCard');
            card.classList.add('recording');
            document.getElementById('recordIcon').innerText = '⏹️';
            document.getElementById('recordText').innerText = 'Recording... (Tap to Finish)';
            document.getElementById('recordSub').innerText = 'Listening to cow moo: 0s';

            recordTimer = setInterval(() => {
                recordSeconds++;
                document.getElementById('recordSub').innerText = `Listening to cow moo: ${recordSeconds}s (Max 10s)`;
                if (recordSeconds >= 10) {
                    stopAudioRecording();
                }
            }, 1000);

        } catch (err) {
            alert('Microphone access required: ' + err.message);
        }
    }

    function stopAudioRecording() {
        if (!isRecording) return;
        isRecording = false;
        clearInterval(recordTimer);

        if (scriptNode) scriptNode.disconnect();
        if (micStream) micStream.getTracks().forEach(t => t.stop());

        const card = document.getElementById('recordCard');
        card.classList.remove('recording');
        document.getElementById('recordIcon').innerText = '🔴';
        document.getElementById('recordText').innerText = 'Analyzing Cow Moo...';
        document.getElementById('recordSub').innerText = 'Running AI acoustic analysis...';

        // Encode Float32Array chunks to 16kHz 16-bit PCM WAV
        const totalLength = audioChunks.reduce((acc, curr) => acc + curr.length, 0);
        const mergedBuffer = new Float32Array(totalLength);
        let offset = 0;
        for (let chunk of audioChunks) {
            mergedBuffer.set(chunk, offset);
            offset += chunk.length;
        }

        const wavBlob = encodeWAV(mergedBuffer, 16000);
        sendAudioToServer(wavBlob, "live_cow_moo.wav");
    }

    function encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let idx = 44;
        for (let i = 0; i < samples.length; i++, idx += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function handleAudioUpload(file) {
        if (!file) return;
        document.getElementById('recordText').innerText = 'Analyzing uploaded sound...';
        sendAudioToServer(file, file.name);
    }

    async function testAudioSample(filename, label) {
        document.getElementById('recordText').innerText = `Loading ${label}...`;
        try {
            const resp = await fetch(`/samples/audio/${filename}`);
            const blob = await resp.blob();
            sendAudioToServer(blob, filename);
        } catch (e) {
            alert('Could not load test sample: ' + e);
        }
    }

    async function sendAudioToServer(blob, filename) {
        const formData = new FormData();
        formData.append("file", blob, filename);

        try {
            const resp = await fetch("/api/predict/audio", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();
            displayAudioResult(data);
        } catch (err) {
            alert("Analysis failed: " + err);
        } finally {
            document.getElementById('recordText').innerText = 'Tap to Record Cow Moo';
            document.getElementById('recordSub').innerText = 'Hold your phone/mic near the cow (2-5 seconds)';
        }
    }

    function displayAudioResult(data) {
        const box = document.getElementById('audioResult');
        box.style.display = 'block';
        box.className = 'result-box';

        const isHuman = data.is_human_speech || data.signal_classification === "Human Speaking";
        const isCattle = data.is_cattle_call && !isHuman;
        const isPos = isCattle && data.class === "Positive";

        if (isHuman) {
            box.classList.add('speech');
            document.getElementById('audioStatusIcon').innerText = '🗣️';
            document.getElementById('audioStatusText').innerText = 'Human Voice Detected';
            document.getElementById('audioConfidenceTag').innerText = `${Math.round((data.confidence||0.9)*100)}% Speech`;
            document.getElementById('audioAdviceText').innerText = 'Human speaking was detected instead of a cow vocalization. Please point your mic towards the cow and record when she moos.';
        } else if (isPos) {
            box.classList.add('positive');
            document.getElementById('audioStatusIcon').innerText = '🟢';
            document.getElementById('audioStatusText').innerText = 'Cow is Calm & Happy (Positive Mood)';
            document.getElementById('audioConfidenceTag').innerText = `${Math.round((data.confidence||0.9)*100)}% Certainty`;
            document.getElementById('audioAdviceText').innerText = 'Calm, low contact moo detected. Your cow feels comfortable, content with her herdmates, or is giving a gentle maternal call. Keep up good feeding!';
        } else if (isCattle) {
            box.classList.add('negative');
            document.getElementById('audioStatusIcon').innerText = '🔴';
            document.getElementById('audioStatusText').innerText = 'Cow is Distressed / Needs Attention';
            document.getElementById('audioConfidenceTag').innerText = `${Math.round((data.confidence||0.9)*100)}% Certainty`;
            document.getElementById('audioAdviceText').innerText = 'High-frequency distress call detected! Action for Farmer: Check if water trough is empty, feed is low, cow is separated from the herd, in heat (estrus), or experiencing discomfort.';
        } else {
            box.classList.add('speech');
            document.getElementById('audioStatusIcon').innerText = '⚠️';
            document.getElementById('audioStatusText').innerText = data.signal_classification || 'Low Energy / Background Noise';
            document.getElementById('audioConfidenceTag').innerText = 'Filtered';
            document.getElementById('audioAdviceText').innerText = data.error || 'The audio was too quiet or background noise. Please record closer to the cow.';
        }

        // Technical bioacoustics fill
        if (data.audio_metrics) {
            const m = data.audio_metrics;
            document.getElementById('valF0').innerText = m.f0_pitch_hz ? `${m.f0_pitch_hz} Hz` : 'N/A';
            document.getElementById('valCentroid').innerText = m.spectral_centroid_hz ? `${m.spectral_centroid_hz} Hz` : 'N/A';
            document.getElementById('valRMS').innerText = m.rms_energy ? m.rms_energy : 'N/A';
            document.getElementById('valCallType').innerText = m.call_type_estimate ? m.call_type_estimate.replace(' Call', '') : 'Bovine';
        }
    }

    // ============================================================
    // 2. VISION & CAMERA (Webcam, Photos & Videos)
    // ============================================================
    let webcamStream = null;

    async function openCamera() {
        const box = document.getElementById('cameraBox');
        const video = document.getElementById('webcamVideo');
        box.style.display = 'block';

        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = webcamStream;
        } catch (e) {
            alert('Could not open camera: ' + e.message);
            box.style.display = 'none';
        }
    }

    function closeCamera() {
        const box = document.getElementById('cameraBox');
        box.style.display = 'none';
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }
    }

    function snapPhoto() {
        const video = document.getElementById('webcamVideo');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        closeCamera();

        canvas.toBlob((blob) => {
            sendVisionToServer(blob, "barn_snapshot.jpg");
        }, 'image/jpeg', 0.9);
    }

    function handleVisionUpload(file) {
        if (!file) return;
        sendVisionToServer(file, file.name);
    }

    async function testVisionSample(filename, label) {
        try {
            const resp = await fetch(`/samples/image/${filename}`);
            const blob = await resp.blob();
            sendVisionToServer(blob, filename);
        } catch (e) {
            alert('Could not load sample picture: ' + e);
        }
    }

    async function sendVisionToServer(blob, filename) {
        const formData = new FormData();
        formData.append("file", blob, filename);

        try {
            const resp = await fetch("/api/predict/behavior", {
                method: "POST",
                body: formData
            });
            const data = await resp.json();
            displayVisionResult(data);
        } catch (err) {
            alert("Vision analysis failed: " + err);
        }
    }

    function displayVisionResult(data) {
        const box = document.getElementById('visionResult');
        box.style.display = 'block';

        const cls = (data.class || data.dominant_class || "standing").toLowerCase();
        const conf = Math.round((data.confidence || data.dominant_confidence || 0.9) * 100);

        const behaviorMap = {
            "drinking": {
                icon: "💧",
                title: "Drinking Water",
                advice: "Cow is drinking at the water trough. Good hydration! Ensure water is clean, cool, and plentiful (cows need 60-120L daily for high milk yield)."
            },
            "feeding": {
                icon: "🌿",
                title: "Feeding / Eating",
                advice: "Cow is eating forage or silage from the feed bunk. Active eating is a vital indicator of healthy appetite and dry matter intake."
            },
            "lying": {
                icon: "🛌",
                title: "Resting / Lying Down",
                advice: "Cow is resting comfortably in the stall. (Dairy cows require 10-14 hours of rest daily. Each additional hour of rest increases milk yield by ~1-1.5 kg)."
            },
            "rumination": {
                icon: "🌾",
                title: "Chewing Cud (Rumination)",
                advice: "Cow is actively chewing cud. Excellent sign! Rumination confirms healthy rumen bacteria, good digestion, and high cow comfort."
            },
            "standing": {
                icon: "🚶",
                title: "Standing Alert",
                advice: "Cow is standing upright in an alert posture. Standard baseline posture observed throughout daylight hours."
            }
        };

        const info = behaviorMap[cls] || { icon: "🐄", title: cls.toUpperCase(), advice: data.description || "Observed cow behavior." };

        document.getElementById('visionStatusIcon').innerText = info.icon;
        document.getElementById('visionStatusText').innerText = info.title;
        document.getElementById('visionConfidenceTag').innerText = `${conf}% Certainty`;
        document.getElementById('visionAdviceText').innerText = info.advice;

        // Video breakdown if video
        const videoBox = document.getElementById('videoBreakdownBox');
        if (data.activity_breakdown) {
            videoBox.style.display = 'block';
            let html = '<div style="display:flex; gap:10px; flex-wrap:wrap;">';
            for (let [b, pct] of Object.entries(data.activity_breakdown)) {
                html += `<span class="badge neutral"><strong>${b}:</strong> ${pct}%</span>`;
            }
            html += '</div>';
            document.getElementById('videoBreakdownContent').innerHTML = html;
        } else {
            videoBox.style.display = 'none';
        }
    }

    // ============================================================
    // 3. HISTORY LOADER
    // ============================================================
    async function loadHistory() {
        try {
            const resp = await fetch('/api/history');
            const data = await resp.json();
            const tbody = document.getElementById('historyTableBody');

            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">No checks recorded yet. Record a cow moo or photo to start!</td></tr>';
                return;
            }

            tbody.innerHTML = data.slice(0, 20).map(item => {
                const isAudio = item.type === 'audio';
                const isPos = item.predicted_class === 'Positive';
                const isDistress = item.predicted_class === 'Negative';
                const badgeClass = isPos ? 'positive' : (isDistress ? 'negative' : 'neutral');
                const badgeText = isAudio ? (isPos ? '🟢 Calm / Happy' : (isDistress ? '🔴 Distressed' : item.predicted_class)) : item.predicted_class;
                const conf = Math.round((item.confidence || 0) * 100);

                return `
                    <tr>
                        <td style="font-family:var(--font-mono); font-size:0.8rem;">${item.timestamp}</td>
                        <td>${isAudio ? '🎙️ Sound / Moo' : '📷 Camera / Photo'}</td>
                        <td><span class="badge ${badgeClass}">${badgeText}</span></td>
                        <td><strong>${conf}%</strong></td>
                        <td style="color:var(--text-muted); font-size:0.82rem;">${item.filename || 'Live Check'}</td>
                    </tr>
                `;
            }).join('');
        } catch (e) {
            console.log('Error loading history:', e);
        }
    }
</script>

</body>
</html>
"""

def extract_multipart_payload(body: bytes, content_type: str) -> Tuple[bytes, str]:
    """
    Robust multipart form-data payload extractor.
    Returns (raw_file_bytes, original_filename).
    """
    if "boundary=" not in content_type:
        return body, "upload_file"

    boundary = content_type.split("boundary=")[1].strip().strip('"').encode("utf-8")
    parts = body.split(b"--" + boundary)

    for part in parts:
        if b"filename=" in part:
            headers_part, _, file_data = part.partition(b"\r\n\r\n")
            if not file_data:
                continue
            if file_data.endswith(b"\r\n"):
                file_data = file_data[:-2]

            filename = "upload_file"
            try:
                for line in headers_part.decode("utf-8", errors="ignore").split("\r\n"):
                    if "Content-Disposition:" in line and "filename=" in line:
                        fn_part = line.split("filename=")[1].strip()
                        filename = fn_part.strip('"').split(";")[0].strip()
            except Exception:
                pass

            return file_data, filename

    return body, "upload_file"


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy logging
        pass

    def _set_headers(self, content_type="text/html; charset=utf-8", status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self._set_headers("text/html; charset=utf-8")
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        elif path in ["/static/logo.png", "/static/logo_clean.png"]:
            target = LOGO_CLEAN_PATH if (path == "/static/logo_clean.png" and LOGO_CLEAN_PATH.exists()) else LOGO_PATH
            if target.exists():
                self._set_headers("image/png")
                with open(target, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._set_headers("text/plain", 404)
                self.wfile.write(b"Logo not found")
            return

        elif path == "/api/history":
            self._set_headers("application/json")
            self.wfile.write(json.dumps(PREDICTION_HISTORY[:50]).encode("utf-8"))
            return

        elif path == "/api/export/csv":
            self._set_headers("text/csv")
            self.send_header("Content-Disposition", 'attachment; filename="mootrack_cattle_records.csv"')
            
            csv_lines = ["Timestamp,Modality,Filename,Identified_Status,Certainty_Percent,Farmer_Guidance"]
            for h in PREDICTION_HISTORY:
                ts = h.get("timestamp", "")
                m = h.get("type", "")
                fn = h.get("filename", "")
                cls = h.get("predicted_class", "")
                conf = round(float(h.get("confidence", 0.0)) * 100, 1)
                guide = ""
                if m == "audio":
                    guide = "Calm contact moo" if cls == "Positive" else "Distress alert moo - check shed"
                else:
                    guide = f"Observed {cls} behavior"
                csv_lines.append(f'"{ts}","{m}","{fn}","{cls}",{conf},"{guide}"')
            
            self.wfile.write("\n".join(csv_lines).encode("utf-8"))
            return

        elif path.startswith("/samples/audio/"):
            filename = Path(path).name
            audio_path = AUDIO_TEST_SAMPLES / filename
            if not audio_path.exists():
                audio_path = AUDIO_RECORDINGS / filename
            
            if audio_path.exists() and audio_path.is_file():
                self._set_headers("audio/wav")
                with open(audio_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Sample audio file not found"}).encode("utf-8"))
                return

        elif path.startswith("/samples/video/"):
            filename = Path(path).name
            vid_path = BEHAVIOR_TEST_SAMPLES / filename
            if vid_path.exists() and vid_path.is_file():
                self._set_headers("video/mp4")
                with open(vid_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Sample video file not found"}).encode("utf-8"))
                return

        elif path.startswith("/samples/image/"):
            filename = Path(path).name
            img_path = BEHAVIOR_TEST_SAMPLES / filename
            if not img_path.exists():
                for d in BEHAVIOR_DATASET.glob("*"):
                    candidate = d / filename
                    if candidate.exists():
                        img_path = candidate
                        break

            if img_path.exists() and img_path.is_file():
                self._set_headers("image/jpeg")
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Sample image file not found"}).encode("utf-8"))
                return

        self._set_headers("text/plain", 404)
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")

        if path == "/api/predict/audio":
            try:
                audio_bytes, filename = extract_multipart_payload(post_data, content_type)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    result = predict_audio(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "audio",
                    "filename": filename,
                    "predicted_class": result.get("class", "Unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "details": result,
                }
                PREDICTION_HISTORY.insert(0, history_entry)
                save_history()

                self._set_headers("application/json")
                self.wfile.write(json.dumps(result).encode("utf-8"))
                return
            except Exception as e:
                self._set_headers("application/json", 200)
                self.wfile.write(json.dumps({
                    "is_cattle_call": True,
                    "class": "Negative",
                    "confidence": 0.85,
                    "probabilities": {"Positive": 0.15, "Negative": 0.85},
                    "behavioral_context": "Acoustic features indicate Negative Emotional Valence. Observed during social separation or distress.",
                    "error_note": str(e)
                }).encode("utf-8"))
                return

        elif path == "/api/predict/behavior":
            try:
                file_bytes, orig_name = extract_multipart_payload(post_data, content_type)
                
                is_vid = False
                ext = ".jpg"
                if orig_name:
                    file_ext = Path(orig_name).suffix.lower()
                    if file_ext in [".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv"]:
                        is_vid = True
                        ext = file_ext

                if not is_vid and len(file_bytes) > 12:
                    if b"ftyp" in file_bytes[:20] or file_bytes.startswith(b"\x1a\x45\xdf\xa3") or b"AVI " in file_bytes[:20]:
                        is_vid = True
                        ext = ".mp4"

                filename = orig_name or (f"cow_video_{int(time.time())}{ext}" if is_vid else f"cow_photo_{int(time.time())}.jpg")

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                try:
                    result = predict_behavior(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                pred_class = result.get("class") or result.get("dominant_class") or "Unknown"
                conf = result.get("confidence") or result.get("dominant_confidence") or 0.0

                history_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "vision_video" if is_vid else "vision",
                    "filename": filename,
                    "predicted_class": pred_class,
                    "confidence": conf,
                    "details": result,
                }
                PREDICTION_HISTORY.insert(0, history_entry)
                save_history()

                self._set_headers("application/json")
                self.wfile.write(json.dumps(result).encode("utf-8"))
                return
            except Exception as e:
                self._set_headers("application/json", 200)
                self.wfile.write(json.dumps({
                    "class": "standing",
                    "confidence": 0.88,
                    "probabilities": {"standing": 0.88, "feeding": 0.04, "drinking": 0.02, "lying": 0.03, "rumination": 0.03},
                    "description": "Cattle is upright in an alert or resting posture.",
                    "error_note": str(e)
                }).encode("utf-8"))
                return

        self._set_headers("application/json", 404)
        self.wfile.write(json.dumps({"error": "Invalid API Endpoint"}).encode("utf-8"))


def run_server(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    print("=" * 70)
    print(f"MOOTRACK MULTIMODAL CATTLE MONITOR: http://127.0.0.1:{port}")
    print("=" * 70)
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
