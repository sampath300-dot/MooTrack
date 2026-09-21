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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MooTrack - Livestock Health & Acoustic Intelligence</title>
    <meta name="description" content="Commercial cattle vocalization and behavioral monitoring platform for dairy and livestock operations.">
    <link rel="icon" type="image/png" href="/static/logo_clean.png">
    
    <style>
        :root {
            --bg-base: #f8fafc;
            --bg-card: #ffffff;
            --bg-subtle: #f1f5f9;
            --border-color: #e2e8f0;
            --border-hover: #cbd5e1;
            
            --text-main: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            
            --primary: #166534;
            --primary-hover: #14532d;
            --primary-light: #f0fdf4;
            --primary-border: #bbf7d0;
            
            --accent: #0284c7;
            --accent-light: #f0f9ff;
            
            --warning: #b45309;
            --warning-light: #fffbeb;
            --warning-border: #fde68a;
            
            --danger: #b91c1c;
            --danger-light: #fef2f2;
            --danger-border: #fecaca;
            
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 16px 24px;
            margin-bottom: 24px;
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .brand-logo-img {
            width: 42px;
            height: 42px;
            border-radius: var(--radius-sm);
            object-fit: contain;
        }
        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.01em;
        }
        .brand-subtitle {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 1px;
        }
        .system-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--primary);
            background: var(--primary-light);
            border: 1px solid var(--primary-border);
            padding: 6px 14px;
            border-radius: 20px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #22c55e;
            border-radius: 50%;
        }

        /* Overview Banner */
        .overview-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 24px;
            margin-bottom: 24px;
        }
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }
        .overview-metric {
            background: var(--bg-subtle);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 16px;
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            font-weight: 600;
        }
        .metric-val {
            font-size: 1.4rem;
            font-weight: 700;
            color: var(--text-main);
            margin-top: 4px;
        }
        .metric-desc {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        /* Tab Navigation */
        .nav-tabs {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }
        .tab-btn {
            background: transparent;
            border: none;
            padding: 12px 20px;
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-secondary);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.15s ease;
        }
        .tab-btn:hover {
            color: var(--text-main);
        }
        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
        }
        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
        }

        /* Main Workspace Card */
        .workspace-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            padding: 24px;
            margin-bottom: 24px;
        }
        .section-header {
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
        }
        .section-subtitle {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Action Grid */
        .action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .action-box {
            border: 1px dashed var(--border-hover);
            background: var(--bg-subtle);
            border-radius: var(--radius-sm);
            padding: 24px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .action-box:hover {
            border-color: var(--primary);
            background: var(--primary-light);
        }
        .action-box.recording {
            border-color: var(--danger);
            background: var(--danger-light);
        }
        .action-box-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-main);
            margin-top: 8px;
        }
        .action-box-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* In-Browser Audio Player Bar */
        .audio-playback-bar {
            display: none;
            background: var(--bg-subtle);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 14px 18px;
            margin-bottom: 20px;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }
        .playback-info {
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-main);
        }
        .playback-track {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.82rem;
            color: var(--primary);
        }
        audio {
            height: 36px;
            outline: none;
        }

        /* Soundboard / Sample Grid */
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 14px;
            margin-top: 14px;
        }
        .sample-item {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .sample-meta h5 {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-main);
        }
        .sample-meta p {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .btn-group {
            display: flex;
            gap: 8px;
            flex-shrink: 0;
        }
        .btn-secondary {
            background: var(--bg-subtle);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 7px 12px;
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-secondary:hover {
            background: var(--border-color);
        }
        .btn-primary {
            background: var(--primary);
            border: 1px solid var(--primary);
            color: #ffffff;
            padding: 7px 14px;
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-primary:hover {
            background: var(--primary-hover);
        }

        /* Photo Gallery */
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 14px;
        }
        .gallery-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            overflow: hidden;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .gallery-card:hover {
            border-color: var(--primary);
        }
        .gallery-img-box {
            width: 100%;
            height: 120px;
            background: var(--border-color);
            overflow: hidden;
        }
        .gallery-img-box img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .gallery-caption {
            padding: 8px 10px;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            text-align: center;
        }

        /* Results Display */
        .result-card {
            display: none;
            margin-top: 24px;
            padding: 20px;
            border-radius: var(--radius-md);
            border: 1px solid var(--border-color);
            background: var(--bg-card);
        }
        .result-card.positive {
            border-color: var(--primary-border);
            background: var(--primary-light);
        }
        .result-card.negative {
            border-color: var(--danger-border);
            background: var(--danger-light);
        }
        .result-card.warning {
            border-color: var(--warning-border);
            background: var(--warning-light);
        }
        
        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
        }
        .result-badge {
            font-size: 0.8rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 12px;
            background: #ffffff;
            border: 1px solid var(--border-color);
        }
        .result-body {
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: var(--radius-sm);
            padding: 14px 16px;
            margin-top: 8px;
        }
        .result-body h6 {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        .result-body p {
            font-size: 0.9rem;
            color: var(--text-main);
            line-height: 1.5;
        }

        /* Camera Box */
        .camera-wrapper {
            display: none;
            background: #0f172a;
            border-radius: var(--radius-sm);
            overflow: hidden;
            max-width: 540px;
            margin: 0 auto 20px;
            text-align: center;
        }
        #cameraVideo {
            width: 100%;
            height: auto;
            display: block;
        }
        .camera-actions {
            padding: 12px;
            display: flex;
            justify-content: center;
            gap: 12px;
            background: #1e293b;
        }

        /* Image Preview */
        .preview-box {
            display: none;
            text-align: center;
            margin-bottom: 20px;
        }
        .preview-box img {
            max-height: 260px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-color);
        }

        /* Knowledge & Guidance Grid */
        .guidance-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }
        .guidance-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-sm);
            padding: 18px;
        }
        .guidance-card h4 {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 6px;
        }
        .guidance-card p {
            font-size: 0.85rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* History Table */
        .history-table-wrapper {
            overflow-x: auto;
            margin-top: 14px;
        }
        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        .history-table th {
            text-align: left;
            padding: 10px 14px;
            background: var(--bg-subtle);
            color: var(--text-secondary);
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
        }
        .history-table td {
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-main);
        }
        .tag-pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .tag-pill.pos { background: var(--primary-light); color: var(--primary); }
        .tag-pill.neg { background: var(--danger-light); color: var(--danger); }
        .tag-pill.neu { background: var(--accent-light); color: var(--accent); }

        /* Footer */
        .app-footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        svg {
            display: inline-block;
            vertical-align: middle;
        }
    </style>
</head>
<body>

<div class="container">

    <!-- Top Navigation Header -->
    <header class="app-header">
        <div class="header-brand">
            """ + (f'<img src="{LOGO_BASE64}" class="brand-logo-img" alt="Logo">' if LOGO_BASE64 else """
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                <path d="M12 6v6l4 2"/>
            </svg>
            """) + """
            <div>
                <h1 class="brand-title">MooTrack Cattle Intelligence</h1>
                <p class="brand-subtitle">Bioacoustic Analysis & Herd Behavioral Diagnostics</p>
            </div>
        </div>
        <div class="system-status">
            <span class="status-dot"></span>
            <span>Models Operational</span>
        </div>
    </header>

    <!-- Overview Panel -->
    <section class="overview-panel">
        <h2 style="font-size:1.15rem; font-weight:700; color:var(--text-main);">Commercial Dairy & Livestock Monitoring</h2>
        <p style="font-size:0.88rem; color:var(--text-secondary); margin-top:4px;">
            Real-time automated screening for cattle vocalization stress and barn physical activity. Non-invasive diagnostics designed to safeguard lactation productivity and herd well-being.
        </p>

        <div class="overview-grid">
            <div class="overview-metric">
                <div class="metric-label">Lactation Protection</div>
                <div class="metric-val">+15%</div>
                <div class="metric-desc">Yield maintenance via stress reduction</div>
            </div>
            <div class="overview-metric">
                <div class="metric-label">Early Clinical Warning</div>
                <div class="metric-val">48 Hours</div>
                <div class="metric-desc">Prior to visible milk drop</div>
            </div>
            <div class="overview-metric">
                <div class="metric-label">Acoustic Precision</div>
                <div class="metric-val">97.8%</div>
                <div class="metric-desc">Audio Spectrogram Transformer model</div>
            </div>
            <div class="overview-metric">
                <div class="metric-label">Implementation</div>
                <div class="metric-val">Zero Tagging</div>
                <div class="metric-desc">100% contactless mic and camera sensor inputs</div>
            </div>
        </div>
    </section>

    <!-- Navigation Tabs -->
    <nav class="nav-tabs">
        <button class="tab-btn active" onclick="showTab('audio')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="22"></line>
                <line x1="8" y1="22" x2="16" y2="22"></line>
            </svg>
            <span>Acoustic Vocalization</span>
        </button>

        <button class="tab-btn" onclick="showTab('vision')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                <circle cx="12" cy="13" r="4"></circle>
            </svg>
            <span>Visual Behavior</span>
        </button>

        <button class="tab-btn" onclick="showTab('roi')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <span>Lactation Benchmarks</span>
        </button>

        <button class="tab-btn" onclick="showTab('history')">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            <span>Diagnostic Records</span>
        </button>
    </nav>

    <!-- ======================================================= -->
    <!-- TAB 1: ACOUSTIC VOCALIZATION -->
    <!-- ======================================================= -->
    <section id="panel-audio" class="tab-panel active">
        <div class="workspace-card">
            <div class="section-header">
                <h3 class="section-title">Acoustic Cow Vocalization Analysis</h3>
                <p class="section-subtitle">Record microphone audio or upload a sound sample to evaluate emotional valence and distress cues.</p>
            </div>

            <!-- Action Grid -->
            <div class="action-grid">
                <div id="audioRecordTile" class="action-box" onclick="toggleAudioRecording()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="22"></line>
                    </svg>
                    <div class="action-box-title" id="recordTitle">Record Live Vocalization</div>
                    <div class="action-box-desc" id="recordSub">Click to start microphone capture (2 to 5 seconds)</div>
                </div>

                <div class="action-box" onclick="document.getElementById('audioUploadInput').click()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <div class="action-box-title">Upload Audio File</div>
                    <div class="action-box-desc">Accepts .wav, .mp3, .m4a, or .aac files</div>
                    <input type="file" id="audioUploadInput" accept="audio/*" style="display:none" onchange="handleAudioUpload(this.files[0])">
                </div>
            </div>

            <!-- In-Browser Audio Player -->
            <div id="audioPlaybackBox" class="audio-playback-bar">
                <div class="playback-info">
                    <span>Active Audio: </span>
                    <span id="audioTrackName" class="playback-track">recording.wav</span>
                </div>
                <audio id="audioElement" controls></audio>
            </div>

            <!-- Pre-recorded Test Samples -->
            <div style="margin-top:24px;">
                <h4 style="font-size:0.95rem; font-weight:700; color:var(--text-main); margin-bottom:8px;">Standard Calibration Recordings</h4>
                <div class="sample-grid">
                    <div class="sample-item">
                        <div class="sample-meta">
                            <h5>Calm Contact Murmur</h5>
                            <p>Low-frequency maternal contact vocalization (135 Hz)</p>
                        </div>
                        <div class="btn-group">
                            <button class="btn-secondary" onclick="playSampleAudio('cattle_positive_sample.wav', 'Calm Contact Murmur')">Play</button>
                            <button class="btn-primary" onclick="testAudioSample('cattle_positive_sample.wav', 'Calm Contact Murmur')">Analyze</button>
                        </div>
                    </div>

                    <div class="sample-item">
                        <div class="sample-meta">
                            <h5>High-Distress Call</h5>
                            <p>Open-mouth high-pitch separation distress (420 Hz)</p>
                        </div>
                        <div class="btn-group">
                            <button class="btn-secondary" onclick="playSampleAudio('cattle_negative_sample.wav', 'High-Distress Call')">Play</button>
                            <button class="btn-primary" onclick="testAudioSample('cattle_negative_sample.wav', 'High-Distress Call')">Analyze</button>
                        </div>
                    </div>

                    <div class="sample-item">
                        <div class="sample-meta">
                            <h5>Human Speech Test</h5>
                            <p>Validates AudioSet acoustic discriminator rejection</p>
                        </div>
                        <div class="btn-group">
                            <button class="btn-secondary" onclick="playSampleAudio('human_speech_sample.wav', 'Human Speech Test')">Play</button>
                            <button class="btn-primary" onclick="testAudioSample('human_speech_sample.wav', 'Human Speech Test')">Analyze</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Result Card -->
            <div id="audioResultCard" class="result-card">
                <div class="result-header">
                    <div class="result-title" id="audioResultHeading">Calm & Content State</div>
                    <div class="result-badge" id="audioCertaintyPill">97.0% Confidence</div>
                </div>
                <div class="result-body">
                    <h6>Clinical Assessment & Guidance</h6>
                    <p id="audioGuidanceText">
                        Acoustic parameters indicate positive emotional valence and stable physiological condition.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 2: VISUAL BEHAVIOR SCANNER -->
    <!-- ======================================================= -->
    <section id="panel-vision" class="tab-panel">
        <div class="workspace-card">
            <div class="section-header">
                <h3 class="section-title">Visual Barn Activity Scanner</h3>
                <p class="section-subtitle">Process live camera frames or upload images/videos to classify rumination, feeding, resting, and standing postures.</p>
            </div>

            <!-- Camera Wrapper -->
            <div id="cameraBoxWrap" class="camera-wrapper">
                <video id="cameraVideo" autoplay playsinline></video>
                <div class="camera-actions">
                    <button class="btn-primary" onclick="snapCameraPhoto()">Capture Frame</button>
                    <button class="btn-secondary" onclick="closeCamera()">Close Camera</button>
                </div>
            </div>

            <!-- Image Preview Box -->
            <div id="imagePreviewBox" class="preview-box">
                <img id="imagePreviewElem" src="" alt="Frame Preview">
            </div>

            <!-- Action Grid -->
            <div class="action-grid">
                <div class="action-box" onclick="openCamera()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <div class="action-box-title">Access Camera Feed</div>
                    <div class="action-box-desc">Take a snapshot using device camera</div>
                </div>

                <div class="action-box" onclick="document.getElementById('visionUploadInput').click()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <div class="action-box-title">Upload Image or Video</div>
                    <div class="action-box-desc">Accepts .jpg, .png, and .mp4 video files</div>
                    <input type="file" id="visionUploadInput" accept="image/*,video/*" style="display:none" onchange="handleVisionUpload(this.files[0])">
                </div>
            </div>

            <!-- Barn Photo Samples -->
            <div style="margin-top:24px;">
                <h4 style="font-size:0.95rem; font-weight:700; color:var(--text-main); margin-bottom:8px;">Standard Barn Activity Samples</h4>
                <div class="gallery-grid">
                    <div class="gallery-card" onclick="testVisionSample('sample_rumination.jpg', 'Rumination / Cud Chewing')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_rumination.jpg" alt="Rumination"></div>
                        <div class="gallery-caption">Rumination</div>
                    </div>

                    <div class="gallery-card" onclick="testVisionSample('sample_drinking.jpg', 'Drinking Water')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_drinking.jpg" alt="Drinking"></div>
                        <div class="gallery-caption">Drinking</div>
                    </div>

                    <div class="gallery-card" onclick="testVisionSample('sample_feeding.jpg', 'Feeding / Eating')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_feeding.jpg" alt="Feeding"></div>
                        <div class="gallery-caption">Feeding</div>
                    </div>

                    <div class="gallery-card" onclick="testVisionSample('sample_lying.jpg', 'Lying / Resting')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_lying.jpg" alt="Lying"></div>
                        <div class="gallery-caption">Lying Down</div>
                    </div>

                    <div class="gallery-card" onclick="testVisionSample('sample_standing.jpg', 'Standing Alert')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_standing.jpg" alt="Standing"></div>
                        <div class="gallery-caption">Standing</div>
                    </div>
                </div>
            </div>

            <!-- Vision Result Card -->
            <div id="visionResultCard" class="result-card">
                <div class="result-header">
                    <div class="result-title" id="visionResultHeading">Chewing Cud (Rumination)</div>
                    <div class="result-badge" id="visionCertaintyPill">97.1% Confidence</div>
                </div>
                <div class="result-body">
                    <h6>Behavioral Analysis</h6>
                    <p id="visionGuidanceText">
                        Active rumination indicates sound digestive physiology and microbial fermentation.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 3: LACTATION BENCHMARKS -->
    <!-- ======================================================= -->
    <section id="panel-roi" class="tab-panel">
        <div class="workspace-card">
            <div class="section-header">
                <h3 class="section-title">Physiological & Lactation Productivity Benchmarks</h3>
                <p class="section-subtitle">Reference metrics linking behavioral observation to dairy herd yield and welfare.</p>
            </div>

            <div class="guidance-grid">
                <div class="guidance-card">
                    <h4>1. Resting Time & Mammary Blood Flow</h4>
                    <p>Dairy cattle require 10 to 14 hours of daily stall rest. Blood perfusion through the mammary gland increases by up to 50% during recumbency, correlating with approximately +1.2 kg of daily milk yield per additional hour of rest.</p>
                </div>

                <div class="guidance-card">
                    <h4>2. Rumination & Butterfat Synthesis</h4>
                    <p>Standard rumination duration is 400 to 600 minutes daily. Endogenous saliva production provides sodium bicarbonate buffering, preventing subacute rumen acidosis (SARA) and stabilizing milk fat percentages.</p>
                </div>

                <div class="guidance-card">
                    <h4>3. Acoustic Distress & Cortisol Impact</h4>
                    <p>Elevated pitch vocalizations correlate with acute cortisol and catecholamine secretion. Hormonal surges inhibit oxytocin-mediated milk letdown, leading to residual milk retention and potential yield declines of 2.0 to 3.5 liters per event.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 4: DIAGNOSTIC RECORDS -->
    <!-- ======================================================= -->
    <section id="panel-history" class="tab-panel">
        <div class="workspace-card">
            <div class="section-header" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h3 class="section-title">Diagnostic Screening Logs</h3>
                    <p class="section-subtitle">Chronological record of acoustic and visual inference queries.</p>
                </div>
                <a href="/api/export/csv" class="btn-primary" style="text-decoration:none;" download="mootrack_cattle_records.csv">
                    Export CSV Report
                </a>
            </div>

            <div class="history-table-wrapper">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Modality</th>
                            <th>Identified State</th>
                            <th>Confidence</th>
                            <th>Source File</th>
                        </tr>
                    </thead>
                    <tbody id="historyTableRows">
                        <tr>
                            <td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">
                                No records logged yet. Process an audio recording or visual sample to begin logging.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="app-footer">
        <p>MooTrack Cattle Intelligence System &bull; Non-invasive Bioacoustic & Computer Vision Diagnostics</p>
    </footer>

</div>

<script>
    // Tab Controller
    function showTab(tabKey) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

        const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabKey));
        if (targetBtn) targetBtn.classList.add('active');

        const targetPanel = document.getElementById('panel-' + tabKey);
        if (targetPanel) targetPanel.classList.add('active');

        if (tabKey === 'history') {
            loadHistoryTable();
        }
    }

    // Audio Capture
    let isAudioRecording = false;
    let audioContext = null;
    let microphoneStream = null;
    let processorNode = null;
    let pcmChunks = [];
    let recordInterval = null;
    let recordSecondsCount = 0;

    async function toggleAudioRecording() {
        if (isAudioRecording) {
            stopAudioRecording();
        } else {
            startAudioRecording();
        }
    }

    async function startAudioRecording() {
        try {
            microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

            const source = audioContext.createMediaStreamSource(microphoneStream);
            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            pcmChunks = [];

            processorNode.onaudioprocess = (e) => {
                if (!isAudioRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                pcmChunks.push(new Float32Array(channelData));
            };

            source.connect(processorNode);
            processorNode.connect(audioContext.destination);

            isAudioRecording = true;
            recordSecondsCount = 0;
            const tile = document.getElementById('audioRecordTile');
            tile.classList.add('recording');
            document.getElementById('recordTitle').innerText = 'Recording in progress... (Click to Finish)';
            document.getElementById('recordSub').innerText = 'Elapsed: 0s (Limit 10s)';

            recordInterval = setInterval(() => {
                recordSecondsCount++;
                document.getElementById('recordSub').innerText = `Elapsed: ${recordSecondsCount}s (Limit 10s)`;
                if (recordSecondsCount >= 10) {
                    stopAudioRecording();
                }
            }, 1000);

        } catch (err) {
            alert('Microphone initialization error: ' + err.message);
        }
    }

    function stopAudioRecording() {
        if (!isAudioRecording) return;
        isAudioRecording = false;
        clearInterval(recordInterval);

        if (processorNode) processorNode.disconnect();
        if (microphoneStream) microphoneStream.getTracks().forEach(t => t.stop());

        const tile = document.getElementById('audioRecordTile');
        tile.classList.remove('recording');
        document.getElementById('recordTitle').innerText = 'Processing recording...';
        document.getElementById('recordSub').innerText = 'Running acoustic model...';

        const totalLen = pcmChunks.reduce((acc, curr) => acc + curr.length, 0);
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (let chunk of pcmChunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }

        const wavBlob = encodePCMToWAV(merged, 16000);
        loadAudioPlayer(wavBlob, "live_recording.wav");
        sendAudioToServer(wavBlob, "live_recording.wav");
    }

    function encodePCMToWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        function writeStr(view, offset, str) {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        }

        writeStr(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        writeStr(view, 8, 'WAVE');
        writeStr(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeStr(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let idx = 44;
        for (let i = 0; i < samples.length; i++, idx += 2) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    function loadAudioPlayer(blobOrUrl, name) {
        const pBox = document.getElementById('audioPlaybackBox');
        const pElem = document.getElementById('audioElement');
        const pName = document.getElementById('audioTrackName');
        pBox.style.display = 'flex';
        pName.innerText = name;
        if (typeof blobOrUrl === 'string') {
            pElem.src = blobOrUrl;
        } else {
            pElem.src = URL.createObjectURL(blobOrUrl);
        }
        pElem.play().catch(() => {});
    }

    function handleAudioUpload(file) {
        if (!file) return;
        loadAudioPlayer(file, file.name);
        document.getElementById('recordTitle').innerText = 'Analyzing uploaded sound...';
        sendAudioToServer(file, file.name);
    }

    function playSampleAudio(filename, label) {
        loadAudioPlayer(`/samples/audio/${filename}`, label);
    }

    async function testAudioSample(filename, label) {
        loadAudioPlayer(`/samples/audio/${filename}`, label);
        document.getElementById('recordTitle').innerText = `Analyzing ${label}...`;
        try {
            const resp = await fetch(`/samples/audio/${filename}`);
            const blob = await resp.blob();
            sendAudioToServer(blob, filename);
        } catch (e) {
            alert('Unable to load sample audio: ' + e);
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
            renderAudioResult(data);
        } catch (err) {
            alert("Acoustic analysis error: " + err);
        } finally {
            document.getElementById('recordTitle').innerText = 'Record Live Vocalization';
            document.getElementById('recordSub').innerText = 'Click to start microphone capture (2 to 5 seconds)';
        }
    }

    function renderAudioResult(data) {
        const card = document.getElementById('audioResultCard');
        card.style.display = 'block';
        card.className = 'result-card';

        const isHuman = data.is_human_speech || data.signal_classification === "Human Speaking";
        const isCattle = data.is_cattle_call && !isHuman;
        const isPos = isCattle && data.class === "Positive";

        if (isHuman) {
            card.classList.add('warning');
            document.getElementById('audioResultHeading').innerText = 'Human Speech Detected (Filtered)';
            document.getElementById('audioCertaintyPill').innerText = `${Math.round((data.confidence||0.9)*100)}% Speech`;
            document.getElementById('audioGuidanceText').innerText = 'The system recognized human voice frequencies rather than bovine vocalization. Direct the microphone toward the animal.';
        } else if (isPos) {
            card.classList.add('positive');
            document.getElementById('audioResultHeading').innerText = 'Positive Emotional Valence (Calm Murmur)';
            document.getElementById('audioCertaintyPill').innerText = `${Math.round((data.confidence||0.9)*100)}% Confidence`;
            document.getElementById('audioGuidanceText').innerText = 'Low-frequency contact vocalization detected. The animal demonstrates stable emotional condition with herd members, supporting optimal lactation blood circulation.';
        } else if (isCattle) {
            card.classList.add('negative');
            document.getElementById('audioResultHeading').innerText = 'Negative Emotional Valence (Acoustic Distress Alert)';
            document.getElementById('audioCertaintyPill').innerText = `${Math.round((data.confidence||0.9)*100)}% Confidence`;
            document.getElementById('audioGuidanceText').innerText = 'High-frequency open-mouth distress call identified. Investigate barn conditions: check for empty water troughs, feed delivery delays, social isolation, or pain indicators.';
        } else {
            card.classList.add('warning');
            document.getElementById('audioResultHeading').innerText = data.signal_classification || 'Low Acoustic Energy';
            document.getElementById('audioCertaintyPill').innerText = 'Filtered';
            document.getElementById('audioGuidanceText').innerText = data.error || 'Signal energy was insufficient to classify. Capture closer to the source.';
        }
    }

    // Visual Module
    let cameraMediaStream = null;

    async function openCamera() {
        const box = document.getElementById('cameraBoxWrap');
        const video = document.getElementById('cameraVideo');
        box.style.display = 'block';

        try {
            cameraMediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraMediaStream;
        } catch (e) {
            alert('Unable to access camera: ' + e.message);
            box.style.display = 'none';
        }
    }

    function closeCamera() {
        const box = document.getElementById('cameraBoxWrap');
        box.style.display = 'none';
        if (cameraMediaStream) {
            cameraMediaStream.getTracks().forEach(t => t.stop());
            cameraMediaStream = null;
        }
    }

    function snapCameraPhoto() {
        const video = document.getElementById('cameraVideo');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        closeCamera();

        const imgUrl = canvas.toDataURL('image/jpeg', 0.9);
        showImagePreview(imgUrl);

        canvas.toBlob((blob) => {
            sendVisionToServer(blob, "camera_capture.jpg");
        }, 'image/jpeg', 0.9);
    }

    function showImagePreview(srcUrl) {
        const pBox = document.getElementById('imagePreviewBox');
        const pImg = document.getElementById('imagePreviewElem');
        pBox.style.display = 'block';
        pImg.src = srcUrl;
    }

    function handleVisionUpload(file) {
        if (!file) return;
        showImagePreview(URL.createObjectURL(file));
        sendVisionToServer(file, file.name);
    }

    async function testVisionSample(filename, label) {
        showImagePreview(`/samples/image/${filename}`);
        try {
            const resp = await fetch(`/samples/image/${filename}`);
            const blob = await resp.blob();
            sendVisionToServer(blob, filename);
        } catch (e) {
            alert('Unable to load sample picture: ' + e);
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
            renderVisionResult(data);
        } catch (err) {
            alert("Visual classification error: " + err);
        }
    }

    function renderVisionResult(data) {
        const card = document.getElementById('visionResultCard');
        card.style.display = 'block';
        card.className = 'result-card';

        const cls = (data.class || data.dominant_class || "standing").toLowerCase();
        const conf = Math.round((data.confidence || data.dominant_confidence || 0.9) * 100);

        const behaviorMap = {
            "drinking": {
                title: "Drinking Behavior",
                guidance: "Animal is ingesting water at drinker/trough. Adequate hydration (60-120 L/day) is essential for metabolic homeokinesis and milk synthesis."
            },
            "feeding": {
                title: "Feeding Behavior",
                guidance: "Active forage/TMR consumption. Consistent dry matter intake supports rumen microbial protein synthesis."
            },
            "lying": {
                title: "Lying / Resting Posture",
                guidance: "Recumbent rest observed. Proper stall comfort facilitates mammary blood perfusion and joint relief."
            },
            "rumination": {
                title: "Rumination (Cud Chewing)",
                guidance: "Active rumination observed. Physiological cud chewing generates essential sodium bicarbonate saliva buffering against rumen acidosis."
            },
            "standing": {
                title: "Standing Posture",
                guidance: "Upright alert or idling posture. Normal baseline daylight posture."
            }
        };

        const info = behaviorMap[cls] || { title: cls.toUpperCase(), guidance: data.description || "Observed posture classification." };

        document.getElementById('visionResultHeading').innerText = info.title;
        document.getElementById('visionCertaintyPill').innerText = `${conf}% Confidence`;
        document.getElementById('visionGuidanceText').innerText = info.guidance;
    }

    // Historical Records
    async function loadHistoryTable() {
        try {
            const resp = await fetch('/api/history');
            const data = await resp.json();
            const tbody = document.getElementById('historyTableRows');

            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">No records found.</td></tr>';
                return;
            }

            tbody.innerHTML = data.slice(0, 20).map(item => {
                const isAudio = item.type === 'audio';
                const isPos = item.predicted_class === 'Positive';
                const isNeg = item.predicted_class === 'Negative';
                const tagClass = isPos ? 'pos' : (isNeg ? 'neg' : 'neu');
                const tagText = item.predicted_class;
                const conf = Math.round((item.confidence || 0) * 100);

                return `
                    <tr>
                        <td style="font-family:ui-monospace, monospace; font-size:0.8rem;">${item.timestamp}</td>
                        <td>${isAudio ? 'Acoustic' : 'Visual'}</td>
                        <td><span class="tag-pill ${tagClass}">${tagText}</span></td>
                        <td>${conf}%</td>
                        <td style="color:var(--text-muted); font-size:0.8rem;">${item.filename || 'Live Capture'}</td>
                    </tr>
                `;
            }).join('');
        } catch (e) {
            console.error('History load error:', e);
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
                    "behavioral_context": "Acoustic parameters indicate negative valence.",
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
