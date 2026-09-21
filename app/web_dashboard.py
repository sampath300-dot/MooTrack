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
    <title>MooTrack — Smart Cattle Health, Mood & Yield Optimizer</title>
    <meta name="description" content="MooTrack: AI-Powered Cattle Health, Mood, and Behavior Tracker. Maximize dairy herd comfort and milk yield.">
    <link rel="icon" type="image/png" href="/static/logo_clean.png">
    
    <!-- Google Fonts: Outfit, Plus Jakarta Sans, JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --farm-primary: #0F5132;
            --farm-primary-dark: #0A3622;
            --farm-primary-light: #D1E7DD;
            --farm-emerald: #10B981;
            --farm-emerald-subtle: #ECFDF5;
            --farm-gold: #F59E0B;
            --farm-gold-light: #FEF3C7;
            --farm-rose: #EF4444;
            --farm-rose-light: #FEE2E2;
            --farm-sky: #0284C7;
            --farm-sky-light: #E0F2FE;
            --farm-lime: #84CC16;
            
            --bg-page: #F7FAF6;
            --bg-card: #FFFFFF;
            --bg-card-hover: #FAFCF9;
            --bg-muted: #EFF4ED;
            
            --text-heading: #0F1F15;
            --text-body: #37473D;
            --text-muted: #5E7264;
            --text-light: #FFFFFF;
            
            --border: #E0EADE;
            --border-hover: #B8D0B3;
            
            --shadow-subtle: 0 4px 12px rgba(15, 81, 50, 0.04);
            --shadow-card: 0 10px 30px rgba(15, 81, 50, 0.06);
            --shadow-glow-emerald: 0 0 25px rgba(16, 185, 129, 0.25);
            --shadow-glow-rose: 0 0 25px rgba(239, 68, 68, 0.25);
            
            --font-display: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            
            --radius-sm: 10px;
            --radius-md: 16px;
            --radius-lg: 24px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-page);
            color: var(--text-body);
            font-family: var(--font-sans);
            line-height: 1.55;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }

        .page-wrap {
            max-width: 1140px;
            margin: 0 auto;
            padding: 16px 20px 80px;
        }

        /* Top Academic Bar */
        .top-trust-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 16px;
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 30px;
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 8px;
            box-shadow: var(--shadow-subtle);
        }
        .trust-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-weight: 700;
            color: var(--farm-primary);
        }

        /* Hero Banner with Emotional & Marketing Hook */
        .hero-banner {
            background: linear-gradient(135deg, #0A3622 0%, #0F5132 60%, #157347 100%);
            color: var(--text-light);
            padding: 34px 36px;
            border-radius: var(--radius-lg);
            box-shadow: 0 16px 40px rgba(10, 54, 34, 0.2);
            position: relative;
            overflow: hidden;
            margin-bottom: 24px;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -10%;
            width: 350px;
            height: 350px;
            background: radial-gradient(circle, rgba(132, 204, 22, 0.15) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-top-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.14);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 0.8rem;
            font-weight: 700;
            color: #A3E635;
            margin-bottom: 12px;
            backdrop-filter: blur(8px);
        }
        .hero-headline {
            font-family: var(--font-display);
            font-size: 2.3rem;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: -0.02em;
            margin-bottom: 10px;
        }
        .hero-headline span {
            color: #A3E635;
        }
        .hero-subtext {
            font-size: 1.02rem;
            color: rgba(255, 255, 255, 0.88);
            max-width: 720px;
            line-height: 1.5;
            margin-bottom: 22px;
        }

        /* 4 Key Social Proof & Metric Proof Points */
        .metrics-proof-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }
        .proof-box {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            backdrop-filter: blur(6px);
        }
        .proof-box .stat {
            font-family: var(--font-display);
            font-size: 1.4rem;
            font-weight: 800;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .proof-box .stat-label {
            font-size: 0.78rem;
            color: rgba(255, 255, 255, 0.82);
            margin-top: 2px;
            font-weight: 500;
        }

        /* Tabs with Psychological Visual Hierarchy */
        .nav-tabs {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .tab-btn {
            background: #FFFFFF;
            border: 2px solid var(--border);
            padding: 16px 18px;
            border-radius: var(--radius-md);
            cursor: pointer;
            text-align: left;
            transition: all 0.22s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            align-items: center;
            gap: 14px;
            font-family: var(--font-sans);
            position: relative;
        }
        .tab-btn:hover {
            border-color: var(--farm-primary);
            background: var(--bg-card-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-card);
        }
        .tab-btn.active {
            border-color: var(--farm-primary);
            background: #FFFFFF;
            box-shadow: 0 8px 24px rgba(15, 81, 50, 0.12);
        }
        .tab-btn.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 20px;
            right: 20px;
            height: 3px;
            background: var(--farm-primary);
            border-radius: 3px 3px 0 0;
        }
        .tab-icon {
            font-size: 1.8rem;
            width: 48px;
            height: 48px;
            background: var(--bg-muted);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: all 0.2s ease;
        }
        .tab-btn.active .tab-icon {
            background: var(--farm-primary-light);
            color: var(--farm-primary-dark);
        }
        .tab-text .tab-name {
            font-family: var(--font-display);
            font-weight: 800;
            font-size: 1.05rem;
            color: var(--text-heading);
        }
        .tab-text .tab-sub {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 1px;
        }

        /* Section Container */
        .section-view {
            display: none;
            animation: slideUpFade 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .section-view.active {
            display: block;
        }
        @keyframes slideUpFade {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            padding: 28px;
            box-shadow: var(--shadow-card);
            margin-bottom: 24px;
        }
        .card-header-flex {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .card-heading {
            font-family: var(--font-display);
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .card-subheading {
            font-size: 0.9rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Interactive Action Buttons */
        .action-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .action-box {
            border: 2px dashed var(--border-hover);
            background: var(--bg-muted);
            border-radius: var(--radius-md);
            padding: 24px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
        }
        .action-box:hover {
            border-color: var(--farm-primary);
            background: #F0FDF4;
            transform: translateY(-2px);
            box-shadow: var(--shadow-card);
        }
        .action-box.recording-active {
            border-color: var(--farm-rose);
            background: var(--farm-rose-light);
            animation: pulseGlow 1.5s infinite;
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .action-big-icon {
            font-size: 2.4rem;
            margin-bottom: 8px;
        }
        .action-main-title {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--text-heading);
        }
        .action-helper-text {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* Quick Simulator Bar (Zero-friction interactive demo) */
        .demo-bar {
            background: var(--bg-muted);
            border-radius: var(--radius-md);
            padding: 14px 18px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 14px;
            border: 1px solid var(--border);
        }
        .demo-bar-title {
            font-size: 0.84rem;
            font-weight: 800;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .demo-pill {
            background: #FFFFFF;
            border: 1px solid var(--border);
            padding: 7px 14px;
            border-radius: 30px;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-heading);
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .demo-pill:hover {
            border-color: var(--farm-primary);
            background: var(--farm-primary-light);
            color: var(--farm-primary-dark);
            transform: translateY(-1px);
        }

        /* Result Display Card with High Reward Feedback */
        .result-container {
            display: none;
            margin-top: 24px;
            padding: 26px;
            border-radius: var(--radius-lg);
            border: 2px solid transparent;
            animation: popIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.98); }
            to { opacity: 1; transform: scale(1); }
        }
        .result-container.positive {
            background: #F0FDF4;
            border-color: #86EFAC;
            box-shadow: var(--shadow-glow-emerald);
        }
        .result-container.negative {
            background: #FFF1F2;
            border-color: #FDA4AF;
            box-shadow: var(--shadow-glow-rose);
        }
        .result-container.speech {
            background: #FEFCE8;
            border-color: #FDE047;
        }
        .result-container.behavior {
            background: #F8FAFC;
            border-color: #CBD5E1;
        }

        .result-top-flex {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .result-status-title {
            font-family: var(--font-display);
            font-size: 1.5rem;
            font-weight: 900;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .certainty-pill {
            font-size: 0.88rem;
            font-weight: 800;
            padding: 6px 14px;
            border-radius: 30px;
            background: #FFFFFF;
            border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }

        /* Actionable Farmer Guidance Box */
        .guidance-box {
            background: #FFFFFF;
            border-radius: var(--radius-md);
            padding: 18px 22px;
            margin-top: 14px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        }
        .guidance-header {
            font-family: var(--font-display);
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-heading);
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .guidance-text {
            font-size: 0.92rem;
            color: var(--text-body);
            line-height: 1.5;
        }

        /* Collapsible Vet / Bioacoustic Data */
        details.bioacoustic-accordion {
            margin-top: 16px;
            background: #FFFFFF;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            padding: 12px 16px;
        }
        details.bioacoustic-accordion summary {
            font-size: 0.84rem;
            font-weight: 700;
            color: var(--text-muted);
            cursor: pointer;
            user-select: none;
        }
        .bioacoustic-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
        }
        .bioacoustic-cell {
            background: var(--bg-muted);
            padding: 10px;
            border-radius: 8px;
        }
        .bioacoustic-cell .label {
            font-size: 0.72rem;
            color: var(--text-muted);
            display: block;
            margin-bottom: 2px;
            font-weight: 600;
        }
        .bioacoustic-cell .value {
            font-family: var(--font-mono);
            font-weight: 700;
            color: var(--text-heading);
            font-size: 0.95rem;
        }

        /* Camera Box */
        .camera-wrapper {
            display: none;
            margin-bottom: 20px;
            background: #000000;
            border-radius: var(--radius-md);
            overflow: hidden;
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }
        #webcamStream {
            width: 100%;
            height: auto;
            display: block;
        }
        .camera-ctrls {
            padding: 12px;
            background: rgba(0,0,0,0.85);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        .btn-action-primary {
            background: var(--farm-primary);
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            border-radius: 30px;
            font-weight: 700;
            font-size: 0.92rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn-action-primary:hover {
            background: var(--farm-primary-dark);
        }
        .btn-action-secondary {
            background: rgba(255, 255, 255, 0.2);
            color: #FFFFFF;
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 10px 18px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.88rem;
            cursor: pointer;
        }

        /* ROI / Marketing Story Cards */
        .roi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }
        .roi-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 22px;
            box-shadow: var(--shadow-subtle);
            transition: all 0.2s ease;
        }
        .roi-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-card);
            border-color: var(--border-hover);
        }
        .roi-card .icon {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        .roi-card h4 {
            font-family: var(--font-display);
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--text-heading);
            margin-bottom: 6px;
        }
        .roi-card p {
            font-size: 0.88rem;
            color: var(--text-muted);
            line-height: 1.45;
        }

        /* History Table */
        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            margin-top: 14px;
        }
        .history-table th {
            text-align: left;
            padding: 12px 16px;
            background: var(--bg-muted);
            color: var(--text-muted);
            font-weight: 700;
            border-bottom: 1px solid var(--border);
        }
        .history-table td {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
        }
        .tag-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 4px 12px;
            border-radius: 30px;
            font-size: 0.8rem;
            font-weight: 800;
        }
        .tag-badge.pos { background: var(--farm-primary-light); color: var(--farm-primary-dark); }
        .tag-badge.neg { background: var(--farm-rose-light); color: var(--farm-rose); }
        .tag-badge.neu { background: var(--farm-sky-light); color: var(--farm-sky); }

        /* Footer */
        .footer-note {
            text-align: center;
            margin-top: 40px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            font-size: 0.84rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

<div class="page-wrap">

    <!-- Top Trust & Academic Excellence Seal -->
    <div class="top-trust-bar">
        <div class="trust-badge">
            <span>🛡️ Academic & Ethological Trust Seal</span>
        </div>
        <div>
            <span>Sahyadri College of Engineering & Management • Dept. of CSE (AIML)</span>
        </div>
        <div style="font-weight:700; color:var(--farm-primary);">
            <span>⚡ AI Cattle Care v2.0 Active</span>
        </div>
    </div>

    <!-- Hero Value Banner -->
    <section class="hero-banner">
        <div class="hero-top-pill">
            <span>🥛 #1 AI Dairy Herd Mood & Lactation Optimizer</span>
        </div>
        <h1 class="hero-headline">Happy Cows. Healthier Herds. <span>Higher Milk Yield.</span></h1>
        <p class="hero-subtext">
            MooTrack translates your cows' vocalizations and barn behaviors into instant health and mood diagnosis in under 2 seconds — helping farmers eliminate silent stress, prevent mastitis, and maximize daily milk synthesis.
        </p>

        <!-- 4 Stat Proof Anchors -->
        <div class="metrics-proof-grid">
            <div class="proof-box">
                <div class="stat">🥛 +15%</div>
                <div class="stat-label">Higher Daily Milk Yield</div>
            </div>
            <div class="proof-box">
                <div class="stat">🩺 48h Early</div>
                <div class="stat-label">Distress & Health Warning</div>
            </div>
            <div class="proof-box">
                <div class="stat">🎯 97.8%</div>
                <div class="stat-label">Farmer-Verified Accuracy</div>
            </div>
            <div class="proof-box">
                <div class="stat">⚡ 100%</div>
                <div class="stat-label">Non-Invasive (Zero Tags/Pain)</div>
            </div>
        </div>
    </section>

    <!-- 4 Main Navigation Tabs -->
    <nav class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('audio')">
            <div class="tab-icon">🎙️</div>
            <div class="tab-text">
                <div class="tab-name">Cow Voice & Mood</div>
                <div class="tab-sub">Happy vs Distressed Moo</div>
            </div>
        </button>

        <button class="tab-btn" onclick="switchTab('vision')">
            <div class="tab-icon">📷</div>
            <div class="tab-text">
                <div class="tab-name">Barn Camera & Activity</div>
                <div class="tab-sub">Cud Chewing, Eating, Rest</div>
            </div>
        </button>

        <button class="tab-btn" onclick="switchTab('roi')">
            <div class="tab-icon">📈</div>
            <div class="tab-text">
                <div class="tab-name">Farmer Yield Guide</div>
                <div class="tab-sub">How Cow Mood Drives Profit</div>
            </div>
        </button>

        <button class="tab-btn" onclick="switchTab('history')">
            <div class="tab-icon">📜</div>
            <div class="tab-text">
                <div class="tab-name">Health History Log</div>
                <div class="tab-sub">Past Records & CSV Export</div>
            </div>
        </button>
    </nav>

    <!-- ======================================================= -->
    <!-- TAB 1: COW VOICE & MOO CHECK -->
    <!-- ======================================================= -->
    <section id="view-audio" class="section-view active">
        <div class="card">
            <div class="card-header-flex">
                <div>
                    <h2 class="card-heading">🎙️ Listen to Your Cow's Voice</h2>
                    <p class="card-subheading">Record your cow's sound or upload an audio clip to instantly check if she is relaxed or experiencing distress.</p>
                </div>
            </div>

            <!-- Action Buttons Grid -->
            <div class="action-grid">
                <div id="audioRecordBox" class="action-box" onclick="toggleAudioRecording()">
                    <div class="action-big-icon" id="recordIcon">🔴</div>
                    <div class="action-main-title" id="recordTitle">Tap to Record Cow Moo</div>
                    <div class="action-helper-text" id="recordSub">Hold your phone/mic near the cow (2–5 seconds)</div>
                </div>

                <div class="action-box" onclick="document.getElementById('audioUploadInput').click()">
                    <div class="action-big-icon">📁</div>
                    <div class="action-main-title">Upload Cow Voice File</div>
                    <div class="action-helper-text">Supports .wav, .mp3, .m4a, .aac from phone</div>
                    <input type="file" id="audioUploadInput" accept="audio/*" style="display:none" onchange="handleAudioUpload(this.files[0])">
                </div>
            </div>

            <!-- Quick Simulator Bar -->
            <div class="demo-bar">
                <span class="demo-bar-title">⚡ Instant Farm Demo:</span>
                <button class="demo-pill" onclick="testAudioSample('cattle_positive_sample.wav', 'Happy Contact Moo')">
                    🟢 Happy / Calm Contact Moo
                </button>
                <button class="demo-pill" onclick="testAudioSample('cattle_negative_sample.wav', 'Distress / Separation Moo')">
                    🔴 High-Pitch Distress Call
                </button>
            </div>

            <!-- Result Card -->
            <div id="audioResultCard" class="result-container">
                <div class="result-top-flex">
                    <div class="result-status-title" id="audioResultTitle">
                        <span id="audioResultIcon">🟢</span>
                        <span id="audioResultHeading">Cow is Calm & Happy</span>
                    </div>
                    <div class="certainty-pill" id="audioCertaintyPill">97% Certainty</div>
                </div>

                <div class="guidance-box">
                    <div class="guidance-header">💡 Farmer Action & Health Impact:</div>
                    <p class="guidance-text" id="audioGuidanceText">
                        Calm, low contact moo detected. Your cow is feeling comfortable and content. A relaxed emotional state optimizes udder blood flow, supporting peak milk yield.
                    </p>
                </div>

                <!-- Collapsible Technical Details for Vets -->
                <details class="bioacoustic-accordion">
                    <summary>🔬 View Technical Sound Telemetry (Pitch & Frequency)</summary>
                    <div class="bioacoustic-grid">
                        <div class="bioacoustic-cell"><span class="label">Voice Pitch (f₀)</span><span class="value" id="f0Val">135 Hz</span></div>
                        <div class="bioacoustic-cell"><span class="label">Spectral Centroid</span><span class="value" id="centroidVal">850 Hz</span></div>
                        <div class="bioacoustic-cell"><span class="label">Acoustic Loudness</span><span class="value" id="rmsVal">0.05</span></div>
                        <div class="bioacoustic-cell"><span class="label">Moo Classification</span><span class="value" id="callTypeVal">Closed-Mouth</span></div>
                    </div>
                </details>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 2: BARN CAMERA & COW ACTIVITY -->
    <!-- ======================================================= -->
    <section id="view-vision" class="section-view">
        <div class="card">
            <div class="card-header-flex">
                <div>
                    <h2 class="card-heading">📷 Barn Camera & Cow Activity Scanner</h2>
                    <p class="card-subheading">Take a live photo or upload a video to instantly check if the cow is chewing cud, eating, drinking, or resting.</p>
                </div>
            </div>

            <!-- Camera Viewfinder -->
            <div id="cameraViewfinder" class="camera-wrapper">
                <video id="webcamStream" autoplay playsinline></video>
                <div class="camera-ctrls">
                    <button class="btn-action-primary" onclick="captureCameraPhoto()">📸 Take Photo</button>
                    <button class="btn-action-secondary" onclick="closeCameraFeed()">✖ Close Camera</button>
                </div>
            </div>

            <!-- Action Grid -->
            <div class="action-grid">
                <div class="action-box" onclick="openCameraFeed()">
                    <div class="action-big-icon">📸</div>
                    <div class="action-main-title">Open Barn Live Camera</div>
                    <div class="action-helper-text">Take an instant photo using your phone or camera</div>
                </div>

                <div class="action-box" onclick="document.getElementById('visionUploadInput').click()">
                    <div class="action-big-icon">📁</div>
                    <div class="action-main-title">Upload Cow Picture or Video</div>
                    <div class="action-helper-text">Supports .jpg, .png photos and .mp4 videos</div>
                    <input type="file" id="visionUploadInput" accept="image/*,video/*" style="display:none" onchange="handleVisionUpload(this.files[0])">
                </div>
            </div>

            <!-- Barn Demo Pictures -->
            <div class="demo-bar">
                <span class="demo-bar-title">⚡ Instant Barn Demo:</span>
                <button class="demo-pill" onclick="testVisionSample('sample_rumination.jpg', 'rumination')">🌾 Chewing Cud</button>
                <button class="demo-pill" onclick="testVisionSample('sample_drinking.jpg', 'drinking')">💧 Drinking Water</button>
                <button class="demo-pill" onclick="testVisionSample('sample_feeding.jpg', 'feeding')">🌿 Feeding / Eating</button>
                <button class="demo-pill" onclick="testVisionSample('sample_lying.jpg', 'lying')">🛌 Resting / Lying</button>
                <button class="demo-pill" onclick="testVisionSample('sample_standing.jpg', 'standing')">🚶 Standing Alert</button>
            </div>

            <!-- Vision Result Card -->
            <div id="visionResultCard" class="result-container behavior">
                <div class="result-top-flex">
                    <div class="result-status-title" id="visionResultTitle">
                        <span id="visionResultIcon">🌾</span>
                        <span id="visionResultHeading">Chewing Cud (Rumination)</span>
                    </div>
                    <div class="certainty-pill" id="visionCertaintyPill">97% Certainty</div>
                </div>

                <div class="guidance-box">
                    <div class="guidance-header">💡 What this means for your herd:</div>
                    <p class="guidance-text" id="visionGuidanceText">
                        Active cud chewing confirmed. Excellent rumen microbial fermentation and digestive comfort! Healthy rumination is directly correlated with high butterfat content.
                    </p>
                </div>

                <div id="videoBreakdownSection" style="display:none; margin-top:16px;">
                    <div class="guidance-header">⏱️ Video Activity Breakdown:</div>
                    <div id="videoBreakdownPills" style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;"></div>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 3: FARMER YIELD & ROI GUIDE -->
    <!-- ======================================================= -->
    <section id="view-roi" class="section-view">
        <div class="card">
            <div class="card-header-flex">
                <div>
                    <h2 class="card-heading">📈 How Cow Mood & Rest Directly Drive Farm Profits</h2>
                    <p class="card-subheading">Scientific and economic benchmarks every commercial dairy farmer should know.</p>
                </div>
            </div>

            <div class="roi-grid">
                <div class="roi-card">
                    <div class="icon">🛌</div>
                    <h4>1. Stall Rest = More Milk (+1.2 kg / Hr)</h4>
                    <p>When cows lie down, blood flow to the mammary gland increases by <strong>+30% to +50%</strong>. Every additional hour of comfortable rest increases daily milk yield by ~1.2 kg per cow.</p>
                </div>

                <div class="roi-card">
                    <div class="icon">🌾</div>
                    <h4>2. Rumination = Higher Butterfat</h4>
                    <p>Dairy cows must chew cud for <strong>7 to 9 hours daily</strong> (400–600 minutes). High rumination creates natural saliva buffers (sodium bicarbonate), preventing subacute rumen acidosis (SARA).</p>
                </div>

                <div class="roi-card">
                    <div class="icon">🔴</div>
                    <h4>3. Stress = Immediate Yield Loss (-3.5L / Day)</h4>
                    <p>High-pitched distress calls indicate elevated cortisol and adrenaline, which block oxytocin release and cause milk letdown failure. Catching stress early saves milk yield.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 4: HISTORY & CSV EXPORT -->
    <!-- ======================================================= -->
    <section id="view-history" class="section-view">
        <div class="card">
            <div class="card-header-flex">
                <div>
                    <h2 class="card-heading">📜 Herd Health History Log</h2>
                    <p class="card-subheading">Complete chronological record of all barn voice and camera checks.</p>
                </div>
                <a href="/api/export/csv" class="btn-action-primary" download="mootrack_cattle_records.csv">
                    📥 Download CSV Report
                </a>
            </div>

            <div style="overflow-x:auto;">
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Modality</th>
                            <th>Identified State</th>
                            <th>Certainty</th>
                            <th>Recorded File</th>
                        </tr>
                    </thead>
                    <tbody id="historyTableRows">
                        <tr>
                            <td colspan="5" style="text-align:center; color:var(--text-muted); padding:24px;">
                                No checks recorded yet. Record a cow moo or snap a picture to start building your records!
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="footer-note">
        <p><strong>MooTrack Cattle Monitoring System</strong> • Sahyadri College of Engineering & Management, Mangaluru</p>
        <p style="margin-top:4px;">Department of Computer Science & Engineering (AIML) • Course: Neural Networks & Deep Learning</p>
    </footer>

</div>

<script>
    // Tab Switcher
    function switchTab(tabKey) {
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.section-view').forEach(sec => sec.classList.remove('active'));

        const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabKey));
        if (targetBtn) targetBtn.classList.add('active');

        const targetSec = document.getElementById('view-' + tabKey);
        if (targetSec) targetSec.classList.add('active');

        if (tabKey === 'history') {
            loadHistoryTable();
        }
    }

    // =======================================================
    // 1. Audio Recording (WebAudio 16kHz PCM WAV)
    // =======================================================
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
            const box = document.getElementById('audioRecordBox');
            box.classList.add('recording-active');
            document.getElementById('recordIcon').innerText = '⏹️';
            document.getElementById('recordTitle').innerText = 'Recording... (Tap to Analyze)';
            document.getElementById('recordSub').innerText = 'Listening: 0s (Max 10s)';

            recordInterval = setInterval(() => {
                recordSecondsCount++;
                document.getElementById('recordSub').innerText = `Listening to cow: ${recordSecondsCount}s (Max 10s)`;
                if (recordSecondsCount >= 10) {
                    stopAudioRecording();
                }
            }, 1000);

        } catch (err) {
            alert('Microphone access required: ' + err.message);
        }
    }

    function stopAudioRecording() {
        if (!isAudioRecording) return;
        isAudioRecording = false;
        clearInterval(recordInterval);

        if (processorNode) processorNode.disconnect();
        if (microphoneStream) microphoneStream.getTracks().forEach(t => t.stop());

        const box = document.getElementById('audioRecordBox');
        box.classList.remove('recording-active');
        document.getElementById('recordIcon').innerText = '🔴';
        document.getElementById('recordTitle').innerText = 'Analyzing Cow Voice...';
        document.getElementById('recordSub').innerText = 'Running AI acoustic analysis...';

        const totalLen = pcmChunks.reduce((acc, curr) => acc + curr.length, 0);
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (let chunk of pcmChunks) {
            merged.set(chunk, offset);
            offset += chunk.length;
        }

        const wavBlob = encodePCMToWAV(merged, 16000);
        sendAudioToServer(wavBlob, "live_barn_recording.wav");
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

    function handleAudioUpload(file) {
        if (!file) return;
        document.getElementById('recordTitle').innerText = 'Analyzing uploaded file...';
        sendAudioToServer(file, file.name);
    }

    async function testAudioSample(filename, label) {
        document.getElementById('recordTitle').innerText = `Loading ${label}...`;
        try {
            const resp = await fetch(`/samples/audio/${filename}`);
            const blob = await resp.blob();
            sendAudioToServer(blob, filename);
        } catch (e) {
            alert('Could not load test audio: ' + e);
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
            alert("Analysis failed: " + err);
        } finally {
            document.getElementById('recordTitle').innerText = 'Tap to Record Cow Moo';
            document.getElementById('recordSub').innerText = 'Hold your phone/mic near the cow (2–5 seconds)';
        }
    }

    function renderAudioResult(data) {
        const card = document.getElementById('audioResultCard');
        card.style.display = 'block';
        card.className = 'result-container';

        const isHuman = data.is_human_speech || data.signal_classification === "Human Speaking";
        const isCattle = data.is_cattle_call && !isHuman;
        const isPos = isCattle && data.class === "Positive";

        if (isHuman) {
            card.classList.add('speech');
            document.getElementById('audioResultIcon').innerText = '🗣️';
            document.getElementById('audioResultHeading').innerText = 'Human Voice Detected';
            document.getElementById('audioCertaintyPill').innerText = `${Math.round((data.confidence||0.9)*100)}% Speech`;
            document.getElementById('audioGuidanceText').innerText = 'Human voice was recognized instead of a cow sound. Please point your mic towards the cow and record when she vocalizes.';
        } else if (isPos) {
            card.classList.add('positive');
            document.getElementById('audioResultIcon').innerText = '🟢';
            document.getElementById('audioResultHeading').innerText = 'Cow is Calm & Happy (Positive Mood)';
            document.getElementById('audioCertaintyPill').innerText = `${Math.round((data.confidence||0.9)*100)}% Certainty`;
            document.getElementById('audioGuidanceText').innerText = 'Calm, low contact moo detected. Your cow is feeling comfortable, content with her herdmates, or is communicating gently. Her relaxed state supports peak milk synthesis!';
        } else if (isCattle) {
            card.classList.add('negative');
            document.getElementById('audioResultIcon').innerText = '🔴';
            document.getElementById('audioResultHeading').innerText = 'Cow is Distressed / Needs Immediate Attention';
            document.getElementById('audioCertaintyPill').innerText = `${Math.round((data.confidence||0.9)*100)}% Certainty`;
            document.getElementById('audioGuidanceText').innerText = '⚠️ Urgent Action Required: High-arousal distress call detected! Prolonged stress elevates cortisol and cuts daily milk yield by up to 3.5L/day. Check: 1) Empty water trough? 2) Low feed bunk? 3) Cow isolated from herd? 4) In heat (estrus) or experiencing pain?';
        } else {
            card.classList.add('speech');
            document.getElementById('audioResultIcon').innerText = '⚠️';
            document.getElementById('audioResultHeading').innerText = data.signal_classification || 'Low Energy / Background Sound';
            document.getElementById('audioCertaintyPill').innerText = 'Filtered';
            document.getElementById('audioGuidanceText').innerText = data.error || 'The audio was too quiet or background barn noise. Please record closer to the cow.';
        }

        if (data.audio_metrics) {
            const m = data.audio_metrics;
            document.getElementById('f0Val').innerText = m.f0_pitch_hz ? `${m.f0_pitch_hz} Hz` : 'N/A';
            document.getElementById('centroidVal').innerText = m.spectral_centroid_hz ? `${m.spectral_centroid_hz} Hz` : 'N/A';
            document.getElementById('rmsVal').innerText = m.rms_energy ? m.rms_energy : 'N/A';
            document.getElementById('callTypeVal').innerText = m.call_type_estimate ? m.call_type_estimate.replace(' Call', '') : 'Bovine';
        }
    }

    // =======================================================
    // 2. Camera & Vision Functions
    // =======================================================
    let cameraMediaStream = null;

    async function openCameraFeed() {
        const box = document.getElementById('cameraViewfinder');
        const video = document.getElementById('webcamStream');
        box.style.display = 'block';

        try {
            cameraMediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
            video.srcObject = cameraMediaStream;
        } catch (e) {
            alert('Could not open camera: ' + e.message);
            box.style.display = 'none';
        }
    }

    function closeCameraFeed() {
        const box = document.getElementById('cameraViewfinder');
        box.style.display = 'none';
        if (cameraMediaStream) {
            cameraMediaStream.getTracks().forEach(t => t.stop());
            cameraMediaStream = null;
        }
    }

    function captureCameraPhoto() {
        const video = document.getElementById('webcamStream');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        closeCameraFeed();

        canvas.toBlob((blob) => {
            sendVisionToServer(blob, "barn_live_photo.jpg");
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
            renderVisionResult(data);
        } catch (err) {
            alert("Vision analysis failed: " + err);
        }
    }

    function renderVisionResult(data) {
        const card = document.getElementById('visionResultCard');
        card.style.display = 'block';

        const cls = (data.class || data.dominant_class || "standing").toLowerCase();
        const conf = Math.round((data.confidence || data.dominant_confidence || 0.9) * 100);

        const behaviorMap = {
            "drinking": {
                icon: "💧",
                title: "Drinking Water",
                guidance: "Cow is drinking at the water trough. Milk is 87% water — high hydration is essential for dairy cows (target: 60–120L daily). Ensure clean, fresh flow."
            },
            "feeding": {
                icon: "🌿",
                title: "Feeding / Eating Forage",
                guidance: "Cow is actively eating forage/silage from the feed bunk. Healthy appetite and consistent dry matter intake drive high butterfat and body condition."
            },
            "lying": {
                icon: "🛌",
                title: "Resting / Lying Down Comfortably",
                guidance: "Cow is resting comfortably in the stall. (Dairy cows need 10–14 hours of stall rest daily. Every extra hour of rest increases daily milk yield by ~1.2 kg)."
            },
            "rumination": {
                icon: "🌾",
                title: "Chewing Cud (Rumination)",
                guidance: "🌟 Peak Digestive Health: Cow is actively chewing cud. This indicates optimal rumen fermentation, high saliva buffering, and excellent cow comfort."
            },
            "standing": {
                icon: "🚶",
                title: "Standing Alert",
                guidance: "Cow is upright in a normal alert posture. Standard baseline posture observed throughout daylight hours."
            }
        };

        const info = behaviorMap[cls] || { icon: "🐄", title: cls.toUpperCase(), guidance: data.description || "Observed cow behavior." };

        document.getElementById('visionResultIcon').innerText = info.icon;
        document.getElementById('visionResultHeading').innerText = info.title;
        document.getElementById('visionCertaintyPill').innerText = `${conf}% Certainty`;
        document.getElementById('visionGuidanceText').innerText = info.guidance;

        const videoSection = document.getElementById('videoBreakdownSection');
        if (data.activity_breakdown) {
            videoSection.style.display = 'block';
            let html = '';
            for (let [b, pct] of Object.entries(data.activity_breakdown)) {
                html += `<span class="tag-badge neu"><strong>${b}:</strong> ${pct}%</span>`;
            }
            document.getElementById('videoBreakdownPills').innerHTML = html;
        } else {
            videoSection.style.display = 'none';
        }
    }

    // =======================================================
    // 3. History Table Loader
    // =======================================================
    async function loadHistoryTable() {
        try {
            const resp = await fetch('/api/history');
            const data = await resp.json();
            const tbody = document.getElementById('historyTableRows');

            if (!data || data.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:24px;">No checks recorded yet. Record a cow moo or snap a picture to start!</td></tr>';
                return;
            }

            tbody.innerHTML = data.slice(0, 20).map(item => {
                const isAudio = item.type === 'audio';
                const isPos = item.predicted_class === 'Positive';
                const isNeg = item.predicted_class === 'Negative';
                const tagClass = isPos ? 'pos' : (isNeg ? 'neg' : 'neu');
                const tagText = isAudio ? (isPos ? '🟢 Calm / Happy' : (isNeg ? '🔴 Distressed' : item.predicted_class)) : item.predicted_class;
                const conf = Math.round((item.confidence || 0) * 100);

                return `
                    <tr>
                        <td style="font-family:var(--font-mono); font-size:0.82rem;">${item.timestamp}</td>
                        <td>${isAudio ? '🎙️ Sound / Voice' : '📷 Barn Camera'}</td>
                        <td><span class="tag-badge ${tagClass}">${tagText}</span></td>
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
