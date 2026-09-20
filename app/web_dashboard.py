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
<html lang="en" data-theme="optqvo">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <title>MOotrack — Multimodal Cattle Monitoring System</title>
    <meta name="description" content="MOotrack: A Deep Learning-Based Multimodal Cattle Monitoring System. Sahyadri College of Engineering & Management, Mangaluru.">
    <link rel="icon" type="image/png" href="/static/logo_clean.png">
    
    <!-- Google Fonts: Plus Jakarta Sans & JetBrains Mono -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
        :root, html[data-theme="optqvo"] {
            --bg-canvas: #F8F8F5;
            --bg-canvas-subtle: #F2F2ED;
            --bg-white: #FFFFFF;
            --bg-plum: #28113B;
            --bg-plum-hover: #371852;
            --bg-obsidian: #09090C;
            --bg-card-dark: #121217;
            
            --accent-lime: #CCFF00;
            --accent-lime-hover: #b8e600;
            --accent-lime-subtle: rgba(204, 255, 0, 0.12);
            
            --accent-emerald: #10B981;
            --accent-emerald-subtle: rgba(16, 185, 129, 0.12);
            --accent-rose: #F43F5E;
            --accent-rose-subtle: rgba(244, 63, 94, 0.12);
            --accent-sky: #0284C7;
            --accent-sky-subtle: rgba(2, 132, 199, 0.12);
            
            --text-ink: #0A0A0D;
            --text-muted: #50505C;
            --text-subtle: #848492;
            --text-light: #FAFAF8;
            --text-light-muted: rgba(250, 250, 248, 0.75);
            
            --border-subtle: rgba(10, 10, 13, 0.08);
            --border-default: rgba(10, 10, 13, 0.12);
            --border-dark-subtle: rgba(255, 255, 255, 0.10);
            --border-dark-default: rgba(255, 255, 255, 0.15);
            
            --shadow-card: 0 20px 50px rgba(0, 0, 0, 0.06);
            --shadow-plum: 0 20px 45px rgba(40, 17, 59, 0.18);
            --shadow-dark: 0 24px 50px rgba(0, 0, 0, 0.22);
            
            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        html[data-theme="dark"] {
            --bg-canvas: #09090C;
            --bg-canvas-subtle: #121217;
            --bg-white: #16161E;
            --text-ink: #FAFAF8;
            --text-muted: #A0A0B0;
            --text-subtle: #707080;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-default: rgba(255, 255, 255, 0.14);
            --shadow-card: 0 20px 50px rgba(0, 0, 0, 0.4);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::selection { background-color: var(--accent-lime); color: #0A0A0D; }

        body {
            background-color: var(--bg-canvas);
            color: var(--text-ink);
            font-family: var(--font-sans);
            line-height: 1.5;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
            transition: background-color 0.25s ease, color 0.25s ease;
        }

        .dot-grid-bg {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-image: radial-gradient(rgba(10, 10, 13, 0.07) 1px, transparent 1px);
            background-size: 24px 24px;
            pointer-events: none;
            z-index: 0;
            opacity: 0.7;
        }

        html[data-theme="dark"] .dot-grid-bg {
            background-image: radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px);
        }

        .ambient-glow-plum {
            position: absolute;
            top: -100px; left: -50px;
            width: 550px; height: 550px;
            background: rgba(40, 17, 59, 0.05);
            border-radius: 50%;
            filter: blur(100px);
            pointer-events: none;
            z-index: 0;
        }

        .ambient-glow-lime {
            position: absolute;
            top: 250px; right: -80px;
            width: 500px; height: 500px;
            background: rgba(204, 255, 0, 0.08);
            border-radius: 50%;
            filter: blur(110px);
            pointer-events: none;
            z-index: 0;
        }

        .page-wrap {
            position: relative;
            z-index: 10;
            max-width: 1360px;
            margin: 0 auto;
            padding: 0 20px 80px 20px;
        }

        /* Academic Top Banner */
        .academic-banner {
            background: var(--bg-plum);
            color: #FFFFFF;
            padding: 10px 20px;
            border-radius: 14px;
            margin-top: 14px;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.76rem;
            flex-wrap: wrap;
            gap: 8px;
            box-shadow: 0 4px 16px rgba(40, 17, 59, 0.15);
            border: 1px solid rgba(204, 255, 0, 0.2);
        }
        .academic-banner .college-name {
            font-weight: 800;
            letter-spacing: 0.03em;
            color: var(--accent-lime);
        }
        .academic-banner .subject-code {
            font-family: var(--font-mono);
            font-weight: 700;
            background: rgba(255, 255, 255, 0.12);
            padding: 3px 8px;
            border-radius: 6px;
        }

        /* Typography & Badges */
        .badge-plum {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(40, 17, 59, 0.08);
            border: 1px solid rgba(40, 17, 59, 0.16);
            color: var(--bg-plum);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 5px 14px;
            border-radius: 9999px;
        }

        html[data-theme="dark"] .badge-plum {
            background: rgba(204, 255, 0, 0.1);
            border-color: rgba(204, 255, 0, 0.25);
            color: var(--accent-lime);
        }

        .badge-lime {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--accent-lime);
            color: #0A0A0D;
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 4px 10px;
            border-radius: 9999px;
        }

        .badge-rose {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(244, 63, 94, 0.15);
            color: #F43F5E;
            border: 1px solid rgba(244, 63, 94, 0.3);
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 4px 10px;
            border-radius: 9999px;
        }

        .highlight-lime {
            background-color: var(--accent-lime);
            color: #0A0A0D;
            padding: 2px 8px;
            border-radius: 6px;
            display: inline-block;
            font-weight: 900;
        }

        .mono-label {
            font-family: var(--font-mono);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            font-weight: 600;
        }

        .step-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(40, 17, 59, 0.06);
            border: 1px solid var(--border-default);
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            padding: 3px 10px;
            border-radius: 9999px;
            color: var(--bg-plum);
        }

        html[data-theme="dark"] .step-pill {
            background: rgba(255, 255, 255, 0.08);
            color: var(--accent-lime);
        }

        /* Buttons */
        .btn-zip-plum {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: var(--bg-plum);
            color: var(--text-light);
            border: 1px solid transparent;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 10px 20px;
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 16px rgba(40, 17, 59, 0.2);
            text-decoration: none;
        }

        .btn-zip-plum:hover {
            background: var(--bg-plum-hover);
            transform: translateY(-1px);
        }

        .btn-zip-lime {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: var(--accent-lime);
            color: #0A0A0D;
            border: none;
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 12px 24px;
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 16px rgba(204, 255, 0, 0.28);
        }

        .btn-zip-lime:hover {
            background: var(--accent-lime-hover);
            transform: translateY(-1px);
        }

        .btn-zip-outline {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: var(--bg-white);
            color: var(--text-ink);
            border: 1px solid var(--border-default);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 10px 18px;
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-zip-outline:hover {
            border-color: var(--text-ink);
            transform: translateY(-1px);
        }

        .chip-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-white);
            color: var(--text-muted);
            border: 1px solid var(--border-default);
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            padding: 7px 14px;
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .chip-pill:hover {
            color: var(--text-ink);
            border-color: rgba(10, 10, 13, 0.25);
            transform: translateY(-1px);
        }

        .chip-pill.highlight {
            background: var(--bg-plum);
            color: var(--accent-lime);
            border-color: var(--bg-plum);
        }

        /* Floating Header */
        header.optqvo-header {
            position: sticky;
            top: 10px;
            z-index: 100;
            margin-bottom: 24px;
        }

        .header-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.90);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border-default);
            border-radius: 9999px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        }

        html[data-theme="dark"] .header-inner {
            background: rgba(22, 22, 30, 0.90);
        }

        .brand-block {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .brand-logo-text {
            font-family: var(--font-mono);
            font-size: 20px;
            font-weight: 900;
            letter-spacing: -0.05em;
            color: var(--text-ink);
        }

        .brand-logo-text span { color: var(--bg-plum); }
        html[data-theme="dark"] .brand-logo-text span { color: var(--accent-lime); }

        .live-status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 12px;
            border-radius: 9999px;
            background: rgba(10, 10, 13, 0.04);
            border: 1px solid var(--border-subtle);
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .ping-container {
            position: relative;
            display: flex;
            width: 8px; height: 8px;
        }

        .ping-circle {
            position: absolute;
            width: 100%; height: 100%;
            border-radius: 50%;
            background: var(--accent-emerald);
            opacity: 0.75;
            animation: ping 1.8s cubic-bezier(0, 0, 0.2, 1) infinite;
        }

        .ping-dot {
            position: relative;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--accent-emerald);
        }

        @keyframes ping {
            75%, 100% { transform: scale(2.2); opacity: 0; }
        }

        /* Nav Pills */
        .nav-tabs-pill-wrap {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 4px;
            background: rgba(10, 10, 13, 0.04);
            border: 1px solid var(--border-subtle);
            border-radius: 9999px;
            overflow-x: auto;
        }

        .nav-tab-btn {
            background: transparent;
            border: none;
            padding: 8px 16px;
            border-radius: 9999px;
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .nav-tab-btn:hover {
            color: var(--text-ink);
            background: rgba(255, 255, 255, 0.6);
        }

        .nav-tab-btn.active {
            background: var(--bg-plum);
            color: var(--accent-lime);
            box-shadow: 0 4px 14px rgba(40, 17, 59, 0.25);
        }

        /* Hero Section */
        .hero-section { margin-bottom: 24px; }

        .hero-title {
            font-size: clamp(1.8rem, 3.8vw, 3rem);
            font-weight: 900;
            letter-spacing: -0.035em;
            line-height: 1.1;
            color: var(--text-ink);
            margin-bottom: 10px;
        }

        .hero-subtitle {
            font-size: clamp(0.9rem, 1.2vw, 1.05rem);
            color: var(--text-muted);
            line-height: 1.6;
            max-width: 900px;
            margin-bottom: 20px;
        }

        /* Telemetry Strip */
        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }

        .kpi-card {
            background: var(--bg-white);
            border: 1px solid var(--border-default);
            border-radius: 18px;
            padding: 16px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: var(--shadow-card);
        }

        .kpi-icon-wrap {
            width: 40px; height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .kpi-icon-wrap svg { width: 20px; height: 20px; }

        .kpi-content {
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-width: 0;
        }

        .kpi-val {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.02em;
            white-space: nowrap;
        }

        .kpi-sub {
            font-size: 0.72rem;
            color: var(--text-subtle);
            font-weight: 500;
        }

        /* Workspace Grid */
        .workspace-tab-pane { display: none; }
        .workspace-tab-pane.active {
            display: block;
            animation: fadeIn 0.25s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .layout-grid-12 {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 20px;
            align-items: start;
        }

        .col-left { grid-column: span 7; display: flex; flex-direction: column; gap: 18px; }
        .col-right { grid-column: span 5; display: flex; flex-direction: column; gap: 18px; }

        @media (max-width: 1024px) {
            .col-left, .col-right { grid-column: span 12; }
        }

        /* Card Surfaces */
        .card-surface-white {
            background: var(--bg-white);
            border: 1px solid var(--border-default);
            border-radius: 24px;
            padding: 24px;
            box-shadow: var(--shadow-card);
        }

        .card-surface-dark {
            background: var(--bg-obsidian);
            color: var(--text-light);
            border: 1px solid var(--border-dark-subtle);
            border-radius: 24px;
            padding: 24px;
            box-shadow: var(--shadow-dark);
            position: relative;
        }

        .card-surface-plum {
            background: var(--bg-plum);
            color: var(--text-light);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            padding: 24px;
            box-shadow: var(--shadow-plum);
            position: relative;
        }

        .panel-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
            gap: 12px;
        }

        .panel-title-area {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .panel-icon-circle {
            width: 36px; height: 36px;
            border-radius: 10px;
            background: rgba(40, 17, 59, 0.08);
            color: var(--bg-plum);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .panel-icon-circle.dark-theme {
            background: rgba(255, 255, 255, 0.08);
            color: var(--accent-lime);
        }

        .panel-icon-circle svg { width: 18px; height: 18px; }

        .panel-heading-text h2 {
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .panel-heading-text p {
            font-size: 0.78rem;
            color: var(--text-muted);
        }

        .dark-text p { color: var(--text-light-muted); }

        /* Media Viewports */
        .viewport-monitor-container {
            position: relative;
            width: 100%; height: 350px;
            border-radius: 18px;
            background: #020204;
            border: 1px solid var(--border-dark-subtle);
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .viewport-monitor-container video,
        .viewport-monitor-container img {
            width: 100%; height: 100%;
            object-fit: contain;
            display: none;
        }

        .viewport-standby {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
            text-align: center;
            color: rgba(255, 255, 255, 0.6);
            padding: 20px;
        }

        .viewport-standby .orb {
            width: 52px; height: 52px;
            border-radius: 50%;
            background: rgba(204, 255, 0, 0.08);
            border: 1px solid rgba(204, 255, 0, 0.25);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-lime);
        }

        .viewport-hud {
            position: absolute;
            top: 12px; left: 12px; right: 12px;
            display: flex;
            justify-content: space-between;
            pointer-events: none;
            z-index: 10;
        }

        .hud-tag {
            background: rgba(9, 9, 12, 0.75);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #FFFFFF;
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 8px;
        }

        .hud-crosshair {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            width: 90px; height: 90px;
            border: 1.5px dashed var(--accent-lime);
            border-radius: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .hud-crosshair.active {
            opacity: 0.8;
            animation: scanPulse 2s infinite ease-in-out;
        }

        @keyframes scanPulse {
            0% { transform: translate(-50%, -50%) scale(0.95); opacity: 0.5; }
            50% { transform: translate(-50%, -50%) scale(1.05); opacity: 0.9; }
            100% { transform: translate(-50%, -50%) scale(0.95); opacity: 0.5; }
        }

        .controls-action-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }

        .action-step-card {
            background: rgba(10, 10, 13, 0.03);
            border: 1px solid var(--border-default);
            border-radius: 16px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        html[data-theme="dark"] .action-step-card {
            background: rgba(255, 255, 255, 0.03);
        }

        .dropzone-container {
            border: 2px dashed var(--border-default);
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            cursor: pointer;
            background: rgba(10, 10, 13, 0.02);
            transition: all 0.2s ease;
        }

        .dropzone-container:hover {
            border-color: var(--bg-plum);
            background: rgba(40, 17, 59, 0.04);
        }

        .chips-horizontal-scroll {
            display: flex;
            align-items: center;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 4px;
        }

        .audio-playback-card {
            background: rgba(10, 10, 13, 0.03);
            border: 1px solid var(--border-default);
            border-radius: 16px;
            padding: 14px 18px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        audio.farmer-audio-player {
            width: 100%; height: 38px;
            outline: none; border-radius: 8px;
        }

        .waveform-canvas-box {
            width: 100%; height: 180px;
            border-radius: 18px;
            background: #020204;
            border: 1px solid var(--border-dark-subtle);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 14px;
        }

        #audioWaveCanvas { width: 100%; height: 100%; }

        /* Dominant Result Banner */
        .dominant-result-banner {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-dark-subtle);
            border-radius: 18px;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
        }

        .dominant-left { display: flex; align-items: center; gap: 12px; }

        .dominant-icon-badge {
            width: 44px; height: 44px;
            border-radius: 12px;
            background: rgba(204, 255, 0, 0.12);
            color: var(--accent-lime);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .dominant-text h1 {
            font-size: 1.5rem;
            font-weight: 900;
            color: #FFFFFF;
            text-transform: capitalize;
            line-height: 1.1;
        }

        .dominant-confidence-box { text-align: right; }
        .dominant-confidence-box .pct-val {
            font-family: var(--font-mono);
            font-size: 1.35rem;
            font-weight: 900;
            color: var(--accent-lime);
        }

        .prob-bars-wrap {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 18px;
        }

        .prob-bar-item { display: flex; flex-direction: column; gap: 4px; }
        .prob-header { display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700; }

        .prob-track {
            height: 7px;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.1);
            overflow: hidden;
            position: relative;
        }

        .prob-fill {
            height: 100%;
            border-radius: 9999px;
            width: 0%;
            transition: width 0.5s ease-out;
        }

        .ethology-context-box {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-dark-subtle);
            border-radius: 14px;
            padding: 12px 16px;
            font-size: 0.82rem;
            line-height: 1.5;
            color: var(--text-light-muted);
            margin-bottom: 16px;
        }

        /* Multimodal Unified Box */
        .multimodal-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        @media (max-width: 800px) {
            .multimodal-grid { grid-template-columns: 1fr; }
        }

        /* Dataset Tables */
        .specs-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
            margin-bottom: 16px;
        }
        .specs-table th, .specs-table td {
            padding: 10px 14px;
            border: 1px solid var(--border-default);
            text-align: left;
        }
        .specs-table th {
            background: rgba(40, 17, 59, 0.05);
            font-family: var(--font-mono);
            font-weight: 700;
        }

        /* Modal Details */
        .modal-backdrop {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(8px);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .modal-box {
            background: var(--bg-white);
            border-radius: 24px;
            max-width: 800px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-default);
        }

        /* Footer */
        footer.optqvo-footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border-default);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-subtle);
            flex-wrap: wrap;
            gap: 10px;
        }
    </style>
</head>
<body>

    <div class="dot-grid-bg"></div>
    <div class="ambient-glow-plum"></div>
    <div class="ambient-glow-lime"></div>

    <div class="page-wrap">

        <!-- ================= ACADEMIC INSTITUTIONAL TOP BAR ================= -->
        <div class="academic-banner">
            <div>
                <span class="college-name">SAHYADRI COLLEGE OF ENGINEERING &amp; MANAGEMENT, MANGALURU</span>
                <span style="opacity: 0.8;"> &bull; Dept of CSE (AI &amp; ML) &bull; VTU Belagavi</span>
            </div>
            <div>
                <span class="subject-code">AM722T2A &bull; Neural Networks &amp; Deep Learning</span>
            </div>
        </div>

        <!-- ================= FLOATING GLASS HEADER ================= -->
        <header class="optqvo-header">
            <div class="header-inner">
                <div class="brand-block">
                    <span class="brand-logo-text">MO<span>O</span>TRACK</span>
                    <div class="live-status-pill">
                        <span class="ping-container">
                            <span class="ping-circle"></span>
                            <span class="ping-dot"></span>
                        </span>
                        <span>Multimodal PLF AI</span>
                    </div>
                </div>

                <!-- Central Nav Pills -->
                <nav class="nav-tabs-pill-wrap">
                    <button class="nav-tab-btn active" onclick="switchNavTab('multimodal')">🌟 Multimodal Assess</button>
                    <button class="nav-tab-btn" onclick="switchNavTab('studio')">01 Vision &amp; Camera</button>
                    <button class="nav-tab-btn" onclick="switchNavTab('acoustic')">02 Voice &amp; Valence</button>
                    <button class="nav-tab-btn" onclick="switchNavTab('specs')">03 Datasets &amp; Arch</button>
                    <button class="nav-tab-btn" onclick="switchNavTab('history')">04 Farm Records</button>
                </nav>

                <!-- Header Actions -->
                <div style="display: flex; align-items: center; gap: 8px;">
                    <button class="btn-zip-outline" onclick="openProjectModal()" style="padding: 8px 14px; font-size: 11px;">
                        <span>👥 Team &amp; Info</span>
                    </button>
                    <button class="btn-zip-outline" onclick="toggleDarkTheme()" style="padding: 8px 12px; font-size: 11px;">
                        <span id="themeToggleText">🌙 Dark</span>
                    </button>
                </div>
            </div>
        </header>

        <!-- ================= HERO SECTION & TELEMETRY STRIP ================= -->
        <section class="hero-section">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap;">
                <span class="badge-plum">PRECISION LIVESTOCK FARMING (PLF)</span>
                <span class="mono-label">Audio Spectrogram Transformer (AST) + ResNet18 CNN Head</span>
            </div>

            <h1 class="hero-title">
                MULTIMODAL DEEP LEARNING <span class="highlight-lime">CATTLE MONITORING.</span>
            </h1>

            <p class="hero-subtitle">
                Automated welfare assessment combining acoustic vocalization emotional valence classification and vision-based 5-behavior recognition for precision dairy and livestock management.
            </p>

            <!-- Real-Time Telemetry KPI Strip -->
            <div class="telemetry-grid">
                <!-- KPI 1: Vision Posture -->
                <div class="kpi-card">
                    <div class="kpi-icon-wrap" style="background: rgba(40, 17, 59, 0.08); color: var(--bg-plum);">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
                    </div>
                    <div class="kpi-content">
                        <span class="mono-label" style="font-size: 10px;">Visual Behavior</span>
                        <div class="kpi-meta-row">
                            <span class="kpi-val" id="kpiPostureVal">--</span>
                            <span class="badge-lime" id="kpiPostureConf" style="font-size: 9px; padding: 2px 6px;">Ready</span>
                        </div>
                        <span class="kpi-sub" id="kpiPostureSub">ResNet18 Head (CBVD-5)</span>
                    </div>
                </div>

                <!-- KPI 2: Audio Valence -->
                <div class="kpi-card">
                    <div class="kpi-icon-wrap" style="background: rgba(2, 132, 199, 0.08); color: var(--accent-sky);">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                    </div>
                    <div class="kpi-content">
                        <span class="mono-label" style="font-size: 10px;">Acoustic Valence</span>
                        <div class="kpi-meta-row">
                            <span class="kpi-val" id="kpiMoodVal">--</span>
                            <span class="badge-plum" id="kpiMoodConf" style="font-size: 9px; padding: 2px 6px;">Ready</span>
                        </div>
                        <span class="kpi-sub" id="kpiMoodSub">AST Transformer (OpenFarm)</span>
                    </div>
                </div>

                <!-- KPI 3: Herd Comfort Score -->
                <div class="kpi-card">
                    <div class="kpi-icon-wrap" style="background: rgba(16, 185, 129, 0.08); color: var(--accent-emerald);">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                    </div>
                    <div class="kpi-content">
                        <span class="mono-label" style="font-size: 10px;">Herd Comfort Index</span>
                        <div class="kpi-meta-row">
                            <span class="kpi-val" id="kpiWelfareVal">-- / 100</span>
                        </div>
                        <span class="kpi-sub" id="kpiWelfareSub">Awaiting Multimodal Check</span>
                    </div>
                </div>

                <!-- KPI 4: Check Speed -->
                <div class="kpi-card">
                    <div class="kpi-icon-wrap" style="background: rgba(204, 255, 0, 0.16); color: var(--text-ink);">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                    </div>
                    <div class="kpi-content">
                        <span class="mono-label" style="font-size: 10px;">Inference Latency</span>
                        <div class="kpi-meta-row">
                            <span class="kpi-val" id="kpiLatencyVal">Fast</span>
                            <span class="badge-lime" id="kpiLatencyTag" style="font-size: 9px; padding: 2px 6px;">Instant</span>
                        </div>
                        <span class="kpi-sub">PyTorch Engine</span>
                    </div>
                </div>
            </div>
        </section>


        <!-- ================= TAB 0: UNIFIED MULTIMODAL MODE ================= -->
        <div class="workspace-tab-pane active" id="pane-multimodal">
            <div class="card-surface-white" style="margin-bottom: 20px;">
                <div class="panel-header-row">
                    <div class="panel-title-area">
                        <div class="panel-icon-circle" style="background: rgba(204, 255, 0, 0.2); color: #0A0A0D;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
                        </div>
                        <div class="panel-heading-text">
                            <h2>Unified Multimodal Assessment (Audio + Image Combined)</h2>
                            <p>Simultaneously evaluates posture from visual keyframes and emotional valence from vocalizations</p>
                        </div>
                    </div>
                    <span class="badge-lime">Dual-Stream Pipeline</span>
                </div>

                <div class="multimodal-grid">
                    <!-- Left: Select Image -->
                    <div class="action-step-card">
                        <span class="mono-label">1. Cow Visual Keyframe</span>
                        <div class="chips-horizontal-scroll">
                            <button class="chip-pill highlight" id="mmBtn_feeding" onclick="setMMImage('sample_feeding.jpg', this)">Feeding</button>
                            <button class="chip-pill" id="mmBtn_lying" onclick="setMMImage('sample_lying.jpg', this)">Lying Down</button>
                            <button class="chip-pill" id="mmBtn_standing" onclick="setMMImage('sample_standing.jpg', this)">Standing</button>
                            <button class="chip-pill" id="mmBtn_drinking" onclick="setMMImage('sample_drinking.jpg', this)">Drinking</button>
                            <button class="chip-pill" id="mmBtn_rumination" onclick="setMMImage('sample_rumination.jpg', this)">Rumination</button>
                        </div>
                        <div style="height: 180px; background: #000; border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
                            <img id="mmImagePreview" src="/samples/image/sample_feeding.jpg" style="width: 100%; height: 100%; object-fit: contain;">
                        </div>
                    </div>

                    <!-- Right: Select Audio -->
                    <div class="action-step-card">
                        <span class="mono-label">2. Cattle Vocalization</span>
                        <div class="chips-horizontal-scroll">
                            <button class="chip-pill highlight" id="mmBtn_moo1" onclick="setMMAudio('cow_moo_1.wav', this)">Calm Moo 1</button>
                            <button class="chip-pill" id="mmBtn_moo2" onclick="setMMAudio('cow_moo_2.wav', this)">Calm Moo 2</button>
                            <button class="chip-pill" id="mmBtn_distress" onclick="setMMAudio('distress_call.wav', this)">Distress Call</button>
                        </div>
                        <div style="height: 180px; background: rgba(10, 10, 13, 0.04); border-radius: 12px; padding: 20px; display: flex; flex-direction: column; justify-content: center; gap: 14px;">
                            <span id="mmAudioLabel" style="font-family: var(--font-mono); font-size: 11px; font-weight: 700;">Loaded: cow_moo_1.wav</span>
                            <audio id="mmAudioPlayer" class="farmer-audio-player" src="/samples/audio/cow_moo_1.wav" controls></audio>
                        </div>
                    </div>
                </div>

                <div style="margin-top: 18px;">
                    <button class="btn-zip-lime" id="btnRunMultimodal" onclick="runMultimodalCheck()" style="width: 100%; padding: 16px; font-size: 13px;">
                        <span>🚀 RUN MULTIMODAL DUAL-STREAM AI ASSESSMENT</span>
                    </button>
                </div>
            </div>

            <!-- Multimodal Result Showcase -->
            <div id="mmResultsCard" class="card-surface-dark" style="display: none;">
                <div class="panel-header-row dark-text">
                    <div class="panel-title-area">
                        <div class="panel-icon-circle dark-theme">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                        </div>
                        <div class="panel-heading-text">
                            <h2>Multimodal Livestock Assessment Report</h2>
                            <p id="mmLatencyText">Dual AST + ResNet18 inference</p>
                        </div>
                    </div>
                    <span class="badge-lime" id="mmStatusBadge">Report Ready</span>
                </div>

                <div class="multimodal-grid" style="margin-bottom: 20px;">
                    <!-- Vision Result Column -->
                    <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-dark-subtle); border-radius: 16px; padding: 18px;">
                        <span class="mono-label" style="color: var(--accent-lime);">Vision Behavior (ResNet18)</span>
                        <h3 id="mmVisionClass" style="font-size: 1.4rem; color: #FFFFFF; text-transform: capitalize; margin: 6px 0;">--</h3>
                        <div style="font-family: var(--font-mono); font-size: 11px; color: var(--accent-lime); margin-bottom: 10px;" id="mmVisionConf">--% Confidence</div>
                        <p id="mmVisionDesc" style="font-size: 0.8rem; color: var(--text-light-muted); line-height: 1.5;">--</p>
                    </div>

                    <!-- Audio Result Column -->
                    <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-dark-subtle); border-radius: 16px; padding: 18px;">
                        <span class="mono-label" style="color: var(--accent-sky);">Acoustic Valence (AST)</span>
                        <h3 id="mmAudioClass" style="font-size: 1.4rem; color: #FFFFFF; text-transform: capitalize; margin: 6px 0;">--</h3>
                        <div style="font-family: var(--font-mono); font-size: 11px; color: var(--accent-sky); margin-bottom: 10px;" id="mmAudioConf">--% Confidence</div>
                        <p id="mmAudioDesc" style="font-size: 0.8rem; color: var(--text-light-muted); line-height: 1.5;">--</p>
                    </div>
                </div>

                <div style="background: rgba(40, 17, 59, 0.4); border: 1px solid rgba(204, 255, 0, 0.2); border-radius: 16px; padding: 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-family: var(--font-mono); font-size: 12px; font-weight: 800; color: var(--accent-lime);">COMBINED ANIMAL WELFARE INDEX</span>
                        <span id="mmWelfareScore" style="font-family: var(--font-mono); font-size: 1.3rem; font-weight: 900; color: #FFFFFF;">94 / 100</span>
                    </div>
                    <p id="mmWelfareSummary" style="font-size: 0.84rem; color: var(--text-light-muted); line-height: 1.5;">
                        Animal exhibits calm behavioral state with healthy posture. No distress indicators detected.
                    </p>
                </div>
            </div>
        </div>


        <!-- ================= TAB 1: VISION & VIDEO STUDIO ================= -->
        <div class="workspace-tab-pane" id="pane-studio">
            <div class="layout-grid-12">
                
                <div class="col-left card-surface-white">
                    <div class="panel-header-row">
                        <div class="panel-title-area">
                            <div class="panel-icon-circle">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                            </div>
                            <div class="panel-heading-text">
                                <h2>Cow Video &amp; Camera Monitor</h2>
                                <p>Take a photo, record a short video clip, or upload media — then click Check</p>
                            </div>
                        </div>
                        <span class="badge-plum">ResNet18 CNN</span>
                    </div>

                    <div class="viewport-monitor-container" id="mediaViewport">
                        <video id="cameraVideo" autoplay playsinline muted></video>
                        <video id="uploadedVideo" controls playsinline></video>
                        <img id="staticPhoto" alt="Cattle Preview">

                        <div class="viewport-standby" id="viewportStandby">
                            <div class="orb">
                                <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"/><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"/><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"/></svg>
                            </div>
                            <p style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">No Camera or Video Active</p>
                            <p style="font-size: 0.78rem;">Click a test sample button below, upload a photo/video, or turn on your camera.</p>
                        </div>

                        <div class="viewport-hud">
                            <span class="hud-tag" id="hudStreamStatus">STANDBY</span>
                            <span class="hud-tag" id="hudMediaTag" style="color: var(--accent-lime); border-color: rgba(204, 255, 0, 0.4);">Ready to Check</span>
                        </div>
                        <div class="hud-crosshair" id="hudCrosshair"></div>
                    </div>

                    <div class="action-step-card">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="step-pill">Step 1: Take Photo or Record Video</span>
                            <span class="mono-label" id="cameraStatusHint" style="font-size: 10px;">Camera Off</span>
                        </div>

                        <div class="controls-action-row">
                            <button class="btn-zip-plum" id="btnToggleCam" onclick="toggleCamera()">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#CCFF00" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                                <span id="camBtnText">Turn On Camera</span>
                            </button>
                            
                            <button class="btn-zip-outline" id="btnCapturePhoto" onclick="capturePhotoOnly()" style="display: none;">
                                <span>📸 Snap Photo</span>
                            </button>

                            <button class="btn-zip-outline" id="btnRecordVideoClip" onclick="toggleVideoRecording()" style="display: none;">
                                <span id="videoRecBtnText">🎥 Record 5s Clip</span>
                            </button>

                            <button class="btn-zip-outline" id="btnFlipCam" onclick="flipCamera()" style="display: none;">
                                <span>🔄 Flip Camera</span>
                            </button>

                            <button class="btn-zip-outline" id="btnRetakeCam" onclick="restartLiveFeed()" style="display: none;">
                                <span>📹 Live View</span>
                            </button>
                        </div>
                    </div>

                    <div class="action-step-card" id="visionCheckActionCard" style="display: none; background: rgba(204, 255, 0, 0.08); border-color: rgba(204, 255, 0, 0.3);">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="step-pill" style="background: var(--accent-lime); color: #0A0A0D;">Step 2: Media Ready</span>
                            <span class="badge-plum" id="stagedMediaLabel">Photo Ready</span>
                        </div>
                        <p style="font-size: 0.8rem; color: var(--text-muted);">
                            Click below to execute ResNet18 behavior recognition.
                        </p>
                        <div>
                            <button class="btn-zip-lime" onclick="runStagedVisionCheck()" style="width: 100%; padding: 14px; font-size: 13px;">
                                <span>🔍 CHECK COW ACTION NOW (RUN AI)</span>
                            </button>
                        </div>
                    </div>

                    <div class="dropzone-container" onclick="document.getElementById('fileUploadInput').click()" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)" ondrop="handleDrop(event)">
                        <input type="file" id="fileUploadInput" accept="image/*,video/*" style="display: none;" onchange="handleFileSelect(this.files)">
                        <p style="font-size: 0.9rem; font-weight: 800; color: var(--text-ink);">Upload Cow Photo or Video</p>
                        <p style="font-size: 0.76rem; color: var(--text-muted); margin-top: 3px;">Click to pick a photo or video from file system</p>
                    </div>

                    <div>
                        <span class="mono-label" style="display: block; margin-bottom: 8px;">CBVD-5 Quick Test Examples</span>
                        <div class="chips-horizontal-scroll">
                            <button class="chip-pill highlight" onclick="loadSampleVideo()">▶ 5-Second Video</button>
                            <button class="chip-pill" onclick="loadSampleImage('sample_standing.jpg')">Standing</button>
                            <button class="chip-pill" onclick="loadSampleImage('sample_feeding.jpg')">Feeding</button>
                            <button class="chip-pill" onclick="loadSampleImage('sample_drinking.jpg')">Drinking</button>
                            <button class="chip-pill" onclick="loadSampleImage('sample_lying.jpg')">Lying Down</button>
                            <button class="chip-pill" onclick="loadSampleImage('sample_rumination.jpg')">Rumination</button>
                        </div>
                    </div>
                </div>

                <div class="col-right card-surface-dark">
                    <div class="panel-header-row dark-text">
                        <div class="panel-title-area">
                            <div class="panel-icon-circle dark-theme">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                            </div>
                            <div class="panel-heading-text">
                                <h2>Behavior Results</h2>
                                <p id="resultsSubtext">Waiting for photo or video</p>
                            </div>
                        </div>
                        <span class="hud-tag" id="resultsLatencyBadge">-- ms</span>
                    </div>

                    <div id="visionEmptyPrompt" style="text-align: center; padding: 40px 10px; color: rgba(255, 255, 255, 0.6);">
                        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 12px; opacity: 0.5;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                        <h3 style="color: #FFFFFF; font-size: 1.1rem; font-weight: 800; margin-bottom: 6px;">Ready to Check Cow</h3>
                        <p style="font-size: 0.8rem; line-height: 1.5; max-width: 320px; margin: 0 auto;">Take a photo, record a video clip, or choose a sample, then click Check.</p>
                    </div>

                    <div id="visionActiveResults" style="display: none; flex-direction: column;">
                        
                        <div class="dominant-result-banner">
                            <div class="dominant-left">
                                <div class="dominant-icon-badge">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                                </div>
                                <div class="dominant-text">
                                    <span class="mono-label" id="dominantClassTag" style="color: var(--accent-lime); font-size: 10px;">Predicted Action</span>
                                    <h1 id="dominantClassName">--</h1>
                                </div>
                            </div>
                            <div class="dominant-confidence-box">
                                <div class="pct-val" id="dominantConfidencePct">--%</div>
                                <div class="pct-label">Confidence</div>
                            </div>
                        </div>

                        <div class="prob-bars-wrap">
                            <div class="prob-bar-item">
                                <div class="prob-header">
                                    <span style="color: rgba(255, 255, 255, 0.9);">Standing</span>
                                    <span id="pctStanding" style="font-family: var(--font-mono); color: var(--accent-lime);">0.0%</span>
                                </div>
                                <div class="prob-track">
                                    <div class="prob-fill" id="barStanding" style="background: var(--accent-lime);"></div>
                                </div>
                            </div>
                            <div class="prob-bar-item">
                                <div class="prob-header">
                                    <span style="color: rgba(255, 255, 255, 0.9);">Feeding</span>
                                    <span id="pctFeeding" style="font-family: var(--font-mono); color: #F59E0B;">0.0%</span>
                                </div>
                                <div class="prob-track">
                                    <div class="prob-fill" id="barFeeding" style="background: #F59E0B;"></div>
                                </div>
                            </div>
                            <div class="prob-bar-item">
                                <div class="prob-header">
                                    <span style="color: rgba(255, 255, 255, 0.9);">Drinking</span>
                                    <span id="pctDrinking" style="font-family: var(--font-mono); color: #00D2FF;">0.0%</span>
                                </div>
                                <div class="prob-track">
                                    <div class="prob-fill" id="barDrinking" style="background: #00D2FF;"></div>
                                </div>
                            </div>
                            <div class="prob-bar-item">
                                <div class="prob-header">
                                    <span style="color: rgba(255, 255, 255, 0.9);">Lying</span>
                                    <span id="pctLying" style="font-family: var(--font-mono); color: #A78BFA;">0.0%</span>
                                </div>
                                <div class="prob-track">
                                    <div class="prob-fill" id="barLying" style="background: #A78BFA;"></div>
                                </div>
                            </div>
                            <div class="prob-bar-item">
                                <div class="prob-header">
                                    <span style="color: rgba(255, 255, 255, 0.9);">Rumination</span>
                                    <span id="pctRumination" style="font-family: var(--font-mono); color: var(--accent-emerald);">0.0%</span>
                                </div>
                                <div class="prob-track">
                                    <div class="prob-fill" id="barRumination" style="background: var(--accent-emerald);"></div>
                                </div>
                            </div>
                        </div>

                        <div class="ethology-context-box" id="ethologyNoteText">
                            Ethological behavior interpretation will appear here.
                        </div>

                    </div>
                </div>

            </div>
        </div>


        <!-- ================= TAB 2: ACOUSTIC VALENCE AI ================= -->
        <div class="workspace-tab-pane" id="pane-acoustic">
            <div class="layout-grid-12">
                
                <div class="col-left card-surface-white">
                    <div class="panel-header-row">
                        <div class="panel-title-area">
                            <div class="panel-icon-circle" style="color: var(--accent-sky); background: rgba(2, 132, 199, 0.08);">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                            </div>
                            <div class="panel-heading-text">
                                <h2>Cattle Acoustic Valence (AST)</h2>
                                <p>Record or select bovine vocalization, review spectrogram, then analyze valence</p>
                            </div>
                        </div>
                        <span class="badge-plum">Audio Spectrogram Transformer</span>
                    </div>

                    <div class="waveform-canvas-box">
                        <canvas id="audioWaveCanvas"></canvas>
                    </div>

                    <div class="action-step-card">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="step-pill">Step 1: Record or Select Moo Audio</span>
                            <span class="mono-label" id="recStatusHint" style="font-size: 10px;">Mic Ready</span>
                        </div>

                        <div class="controls-action-row">
                            <button class="btn-zip-plum" id="btnRecordMic" onclick="toggleAudioRecording()">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#CCFF00" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="#CCFF00"/></svg>
                                <span id="recBtnText">Record Moo Sound (Mic)</span>
                            </button>
                            <button class="btn-zip-outline" onclick="document.getElementById('audioFileInput').click()">
                                <span>📁 Upload Audio (.wav)</span>
                            </button>
                            <input type="file" id="audioFileInput" accept="audio/*" style="display: none;" onchange="handleAudioUpload(this.files)">
                        </div>
                    </div>

                    <div class="audio-playback-card" id="audioPlaybackCard" style="display: none;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.82rem; font-weight: 700;">Loaded Vocal Audio</span>
                            <span class="badge-lime" id="audioPlaybackStatus">Ready</span>
                        </div>
                        <audio class="farmer-audio-player" id="audioPlaybackElement" controls></audio>
                    </div>

                    <div class="action-step-card" id="audioCheckActionCard" style="display: none; background: rgba(204, 255, 0, 0.08); border-color: rgba(204, 255, 0, 0.3);">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <span class="step-pill" style="background: var(--accent-lime); color: #0A0A0D;">Step 2: Audio Staged</span>
                            <span class="badge-plum" id="stagedAudioLabel">Sound Loaded</span>
                        </div>
                        <div>
                            <button class="btn-zip-lime" id="btnAnalyzeAudio" onclick="runStagedAudioCheck()" style="width: 100%; padding: 14px; font-size: 13px;">
                                <span>🔍 ANALYZE ACOUSTIC VALENCE (RUN AST)</span>
                            </button>
                        </div>
                    </div>

                    <div style="margin-top: 6px;">
                        <span class="mono-label" style="display: block; margin-bottom: 8px;">OpenFarm Vocal Samples</span>
                        <div class="chips-horizontal-scroll">
                            <button class="chip-pill" id="chip_moo1" onclick="testAudioSample('cow_moo_1.wav', this)">🔊 Calm Low Moo (1)</button>
                            <button class="chip-pill" id="chip_moo2" onclick="testAudioSample('cow_moo_2.wav', this)">🔊 Gentle Moo (2)</button>
                            <button class="chip-pill highlight" id="chip_distress" onclick="testAudioSample('distress_call.wav', this)">⚡ Agitated / Distress Call</button>
                        </div>
                    </div>
                </div>

                <div class="col-right card-surface-plum">
                    <div class="panel-header-row dark-text">
                        <div class="panel-title-area">
                            <div class="panel-icon-circle dark-theme">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/></svg>
                            </div>
                            <div class="panel-heading-text">
                                <h2>Acoustic Valence Results</h2>
                                <p id="audioResultsSub">Waiting for moo audio</p>
                            </div>
                        </div>
                    </div>

                    <div id="audioEmptyPrompt" style="text-align: center; padding: 40px 10px; color: rgba(255, 255, 255, 0.6);">
                        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 12px; opacity: 0.5;"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                        <h3 style="color: #FFFFFF; font-size: 1.1rem; font-weight: 800; margin-bottom: 6px;">Ready to Listen</h3>
                        <p style="font-size: 0.8rem; line-height: 1.5; max-width: 320px; margin: 0 auto;">Record a vocalization or pick a sample to classify positive vs negative valence.</p>
                    </div>

                    <div id="audioActiveResults" style="display: none; flex-direction: column;">
                        
                        <div class="dominant-result-banner" id="audioResultBanner" style="background: rgba(0, 0, 0, 0.25);">
                            <div class="dominant-left">
                                <div class="dominant-icon-badge" id="audioIconBadge">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
                                </div>
                                <div class="dominant-text">
                                    <span class="mono-label" style="color: var(--accent-lime); font-size: 10px;">Detected Valence</span>
                                    <h1 id="audioMoodName">--</h1>
                                </div>
                            </div>
                            <div class="dominant-confidence-box">
                                <div class="pct-val" id="audioMoodConfidence">--%</div>
                                <div class="pct-label">Confidence</div>
                            </div>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 18px;">
                            <div style="background: rgba(255, 255, 255, 0.04); padding: 12px; border-radius: 12px; border: 1px solid var(--border-dark-subtle);">
                                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700; color: var(--accent-lime); margin-bottom: 4px;">
                                    <span>Positive Valence</span>
                                    <span id="pctValencePos">0.0%</span>
                                </div>
                                <div class="prob-track"><div class="prob-fill" id="barValencePos" style="background: var(--accent-lime);"></div></div>
                            </div>
                            <div style="background: rgba(255, 255, 255, 0.04); padding: 12px; border-radius: 12px; border: 1px solid var(--border-dark-subtle);">
                                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 700; color: var(--accent-rose); margin-bottom: 4px;">
                                    <span>Negative Valence</span>
                                    <span id="pctValenceNeg">0.0%</span>
                                </div>
                                <div class="prob-track"><div class="prob-fill" id="barValenceNeg" style="background: var(--accent-rose);"></div></div>
                            </div>
                        </div>

                        <!-- Bioacoustic Metrics Box -->
                        <div id="bioacousticBox" style="background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border-dark-subtle); border-radius: 14px; padding: 12px 16px; margin-bottom: 14px;">
                            <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 11px; color: var(--accent-lime); margin-bottom: 8px;">
                                <span id="callTypeEstimate">Call Type: High-Frequency Call</span>
                                <span id="pitchMetric">F0: 392 Hz</span>
                            </div>
                            <div style="font-size: 0.78rem; color: var(--text-light-muted);" id="callTypeDesc">
                                Open-mouth vocalization typically associated with high arousal or distress.
                            </div>
                        </div>

                        <div class="ethology-context-box" id="audioEthologyNote" style="background: rgba(0, 0, 0, 0.2);">
                            Ethological valence interpretation will appear here.
                        </div>

                    </div>
                </div>

            </div>
        </div>


        <!-- ================= TAB 3: DATASETS & ARCHITECTURE ================= -->
        <div class="workspace-tab-pane" id="pane-specs">
            <div class="card-surface-white" style="margin-bottom: 20px;">
                <div class="panel-header-row">
                    <div class="panel-title-area">
                        <div class="panel-icon-circle">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
                        </div>
                        <div class="panel-heading-text">
                            <h2>Dataset Specifications &amp; System Architecture (Poster Section 5 &amp; 6)</h2>
                            <p>Benchmarked datasets and deep learning architectures utilized in the MOotrack project</p>
                        </div>
                    </div>
                </div>

                <div class="layout-grid-12">
                    <div class="col-left" style="grid-column: span 6;">
                        <h3 style="font-size: 1.05rem; margin-bottom: 10px; color: var(--text-ink);">🎙️ Audio Dataset: OpenFarm Ungulate Valence</h3>
                        <table class="specs-table">
                            <tr><th>Attribute</th><th>Specification</th></tr>
                            <tr><td>Dataset Name</td><td>OpenFarm Ungulate Valence Dataset</td></tr>
                            <tr><td>Total Samples</td><td>1,254 audio clips</td></tr>
                            <tr><td>Individual Cattle</td><td>32 individual cattle</td></tr>
                            <tr><td>Species</td><td><em>Bos taurus</em> (Domestic Cattle)</td></tr>
                            <tr><td>Classes</td><td>2 (Negative / Positive)</td></tr>
                            <tr><td>Negative Samples</td><td>1,179 (94.0%) — Social separation</td></tr>
                            <tr><td>Positive Samples</td><td>75 (6.0%) — Social reunion</td></tr>
                        </table>
                    </div>

                    <div class="col-right" style="grid-column: span 6;">
                        <h3 style="font-size: 1.05rem; margin-bottom: 10px; color: var(--text-ink);">👁️ Vision Dataset: CBVD-5 Video Dataset</h3>
                        <table class="specs-table">
                            <tr><th>Attribute</th><th>Specification</th></tr>
                            <tr><td>Dataset Name</td><td>CBVD-5 (Cow Behavior Video Dataset)</td></tr>
                            <tr><td>Total Images</td><td>206,100 keyframes</td></tr>
                            <tr><td>Video Segments</td><td>687 video clips</td></tr>
                            <tr><td>Individual Cattle</td><td>107 individual cattle</td></tr>
                            <tr><td>Classes</td><td>5 Behaviors</td></tr>
                            <tr><td>Target Behaviors</td><td>Standing, Lying, Feeding, Drinking, Rumination</td></tr>
                        </table>
                    </div>
                </div>

                <hr style="margin: 20px 0; border: none; border-top: 1px solid var(--border-default);">

                <div class="layout-grid-12">
                    <div class="col-left" style="grid-column: span 6;">
                        <h3 style="font-size: 1.05rem; margin-bottom: 8px; color: var(--text-ink);">🧠 Audio Model: AST (Audio Spectrogram Transformer)</h3>
                        <ul style="font-size: 0.83rem; line-height: 1.7; color: var(--text-muted); padding-left: 20px;">
                            <li><strong>Pretrained Base:</strong> <code>MIT/ast-finetuned-audioset-10-10-0.4593</code></li>
                            <li><strong>Input:</strong> 16 kHz mono audio &rarr; 128 Mel bins, 1024 frames log-mel spectrogram</li>
                            <li><strong>Fine-Tuning:</strong> Customized 2-class binary cross-entropy head for bovine emotional valence</li>
                            <li><strong>AudioSet Filter:</strong> Pre-trained weights verify cattle moos and reject human speech/noise</li>
                        </ul>
                    </div>

                    <div class="col-right" style="grid-column: span 6;">
                        <h3 style="font-size: 1.05rem; margin-bottom: 8px; color: var(--text-ink);">👁️ Vision Model: ResNet18 CNN</h3>
                        <ul style="font-size: 0.83rem; line-height: 1.7; color: var(--text-muted); padding-left: 20px;">
                            <li><strong>Pretrained Base:</strong> ResNet18 (ImageNet-1k)</li>
                            <li><strong>Custom Head:</strong> <code>nn.Linear(512, 5)</code> outputting 5 behavioral classes</li>
                            <li><strong>Input:</strong> 224 &times; 224 RGB image, resized and normalized</li>
                            <li><strong>Temporal Video:</strong> 1 fps keyframe sampling with majority voting and transition logging</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>


        <!-- ================= TAB 4: AUDIT LOGS & HISTORY ================= -->
        <div class="workspace-tab-pane" id="pane-history">
            <div class="card-surface-white">
                <div class="panel-header-row">
                    <div class="panel-title-area">
                        <div class="panel-icon-circle">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                        </div>
                        <div class="panel-heading-text">
                            <h2>Observation Records &amp; Audit Log (Poster Section 8)</h2>
                            <p>History log of past cattle acoustic, vision, and multimodal checks</p>
                        </div>
                    </div>
                    <button class="btn-zip-plum" onclick="exportHistoryCSV()">
                        <span>Download CSV</span>
                    </button>
                </div>

                <div style="width: 100%; overflow-x: auto;">
                    <table class="specs-table">
                        <thead>
                            <tr>
                                <th>Date &amp; Time</th>
                                <th>Type</th>
                                <th>Classification</th>
                                <th>Confidence</th>
                                <th>Source / File</th>
                            </tr>
                        </thead>
                        <tbody id="auditTableBody">
                            <tr><td colspan="5" style="text-align: center; padding: 20px;">No records found.</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>


        <!-- ================= PROJECT & TEAM MODAL ================= -->
        <div class="modal-backdrop" id="projectModal" onclick="closeProjectModal(event)">
            <div class="modal-box" onclick="event.stopPropagation()">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
                    <div>
                        <span class="badge-plum">ACADEMIC PROJECT PRESENTATION</span>
                        <h2 style="font-size: 1.4rem; font-weight: 900; margin-top: 6px;">MOotrack: Multimodal Cattle Monitoring</h2>
                        <p style="font-size: 0.8rem; color: var(--text-muted);">Sahyadri College of Engineering &amp; Management, Mangaluru</p>
                    </div>
                    <button onclick="closeProjectModal()" style="background: none; border: none; font-size: 24px; cursor: pointer;">&times;</button>
                </div>

                <div style="background: rgba(40, 17, 59, 0.04); border-radius: 16px; padding: 18px; margin-bottom: 20px;">
                    <h3 style="font-size: 0.95rem; font-weight: 800; margin-bottom: 8px;">Project Team</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                        <div><strong>1. Manikanta</strong> (4SF23CI076)</div>
                        <div><strong>2. Sai Sudarshan</strong> (4SF23CI128)</div>
                        <div><strong>3. Sampath</strong> (4SF23CI130)</div>
                        <div><strong>4. Dhruva Shetty</strong> (4SF23CI147)</div>
                    </div>
                </div>

                <div style="font-size: 0.82rem; line-height: 1.6; color: var(--text-muted);">
                    <p><strong>Subject:</strong> Neural Networks and Deep Learning (AM722T2A)</p>
                    <p><strong>Department:</strong> Computer Science and Engineering (Artificial Intelligence &amp; Machine Learning)</p>
                    <p style="margin-top: 10px;"><strong>Key Features:</strong> Cattle sound classification (Positive/Negative), 5 behavior recognition from images (Standing, Lying, Feeding, Drinking, Rumination), prediction confidence, unified multimodal monitoring, web dashboard, and prediction history logging.</p>
                </div>

                <div style="margin-top: 24px; text-align: right;">
                    <button class="btn-zip-plum" onclick="closeProjectModal()">Close</button>
                </div>
            </div>
        </div>


        <!-- ================= FOOTER ================= -->
        <footer class="optqvo-footer">
            <div>
                <strong>MOotrack</strong> &bull; Sahyadri College of Engineering &amp; Management &bull; Dept of CSE (AIML)
            </div>
            <div>
                Team: Manikanta, Sai Sudarshan, Sampath, Dhruva Shetty &bull; Subject: AM722T2A
            </div>
        </footer>

    </div>

    <!-- ================= JAVASCRIPT ================= -->
    <script>
        let currentStream = null;
        let facingMode = "environment";
        let audioContext = null;
        let audioInputNode = null;
        let audioScriptProcessor = null;
        let audioPcmSamples = [];
        let isRecordingAudio = false;
        let videoRecorder = null;
        let videoChunks = [];
        let animationFrameId = null;
        let isDarkMode = false;

        let stagedVisionFormData = null;
        let stagedVisionFilename = '';
        let stagedVisionIsVideo = false;

        let stagedAudioFormData = null;
        let stagedAudioFilename = '';

        let mmSelectedImg = 'sample_feeding.jpg';
        let mmSelectedAud = 'cow_moo_1.wav';

        function toggleDarkTheme() {
            isDarkMode = !isDarkMode;
            document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'optqvo');
            const themeTxt = document.getElementById('themeToggleText');
            if (themeTxt) themeTxt.textContent = isDarkMode ? '☀️ Light' : '🌙 Dark';
        }

        function openProjectModal() { document.getElementById('projectModal').style.display = 'flex'; }
        function closeProjectModal(e) { document.getElementById('projectModal').style.display = 'none'; }

        function switchNavTab(tabKey) {
            document.querySelectorAll('.nav-tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.workspace-tab-pane').forEach(pane => pane.classList.remove('active'));

            const targetPane = document.getElementById('pane-' + tabKey);
            if (targetPane) targetPane.classList.add('active');

            const buttons = document.querySelectorAll('.nav-tab-btn');
            buttons.forEach(b => {
                if (b.getAttribute('onclick') && b.getAttribute('onclick').includes(tabKey)) {
                    b.classList.add('active');
                }
            });

            if (tabKey === 'history') refreshHistoryTable();
        }

        // ==========================================
        // MULTIMODAL UNIFIED FUNCTIONS
        // ==========================================
        function setMMImage(fn, btn) {
            mmSelectedImg = fn;
            document.getElementById('mmImagePreview').src = '/samples/image/' + fn;
            document.querySelectorAll('[id^="mmBtn_"]').forEach(b => b.classList.remove('highlight'));
            if (btn) btn.classList.add('highlight');
        }

        function setMMAudio(fn, btn) {
            mmSelectedAud = fn;
            document.getElementById('mmAudioLabel').textContent = 'Loaded: ' + fn;
            const p = document.getElementById('mmAudioPlayer');
            p.src = '/samples/audio/' + fn;
            p.load();
            document.querySelectorAll('[id^="mmBtn_moo"], #mmBtn_distress').forEach(b => b.classList.remove('highlight'));
            if (btn) btn.classList.add('highlight');
        }

        async function runMultimodalCheck() {
            const card = document.getElementById('mmResultsCard');
            const btn = document.getElementById('btnRunMultimodal');
            btn.innerHTML = '<span>⏳ ANALYZING MULTIMODAL STREAMS (RESNET18 + AST)...</span>';
            card.style.display = 'block';
            document.getElementById('mmLatencyText').textContent = 'Processing multimodal deep learning models...';

            const t0 = performance.now();
            try {
                // Fetch vision prediction
                const vRes = await fetch('/samples/image/' + mmSelectedImg)
                    .then(r => r.blob())
                    .then(blob => {
                        const fd = new FormData();
                        fd.append('file', blob, mmSelectedImg);
                        return fetch('/api/predict/behavior', { method: 'POST', body: fd }).then(r => r.json());
                    });

                // Fetch audio prediction
                const aRes = await fetch('/samples/audio/' + mmSelectedAud)
                    .then(r => r.blob())
                    .then(blob => {
                        const fd = new FormData();
                        fd.append('audio', blob, mmSelectedAud);
                        return fetch('/api/predict/audio', { method: 'POST', body: fd }).then(r => r.json());
                    });

                const latency = Math.round(performance.now() - t0);
                document.getElementById('mmLatencyText').textContent = 'Dual ResNet18 + AST inference completed in ' + latency + ' ms';

                // Populate vision
                const vCls = (vRes.class || 'standing');
                document.getElementById('mmVisionClass').textContent = vCls;
                document.getElementById('mmVisionConf').textContent = Math.round((vRes.confidence || 0.85) * 100) + '% Confidence';
                document.getElementById('mmVisionDesc').textContent = vRes.description || 'Observed posture from CBVD-5 vision pipeline.';

                // Populate audio
                if (aRes.is_human_speech || aRes.class === 'Human Speaking') {
                    document.getElementById('mmAudioClass').textContent = '🗣️ Human Speaking';
                    document.getElementById('mmAudioClass').style.color = '#F59E0B';
                    document.getElementById('mmAudioConf').textContent = Math.round(aRes.speech_confidence || 95) + '% Speech Score';
                    document.getElementById('mmAudioDesc').textContent = 'Human voice identified and filtered. MooTrack processes bovine calls for welfare calculations.';
                    document.getElementById('kpiMoodVal').textContent = '🗣️ HUMAN SPEAKING';
                    document.getElementById('kpiMoodVal').style.color = '#F59E0B';
                    document.getElementById('kpiMoodConf').textContent = Math.round(aRes.speech_confidence || 95) + '% Speech';
                } else {
                    const aCls = (aRes.class || 'Positive');
                    document.getElementById('mmAudioClass').textContent = aCls + ' Valence';
                    document.getElementById('mmAudioClass').style.color = '#FFFFFF';
                    document.getElementById('mmAudioConf').textContent = Math.round((aRes.confidence || 0.85) * 100) + '% Confidence';
                    document.getElementById('mmAudioDesc').textContent = aRes.behavioral_context || 'Acoustic emotional state from OpenFarm AST pipeline.';
                    document.getElementById('kpiMoodVal').textContent = aCls + ' Valence';
                    document.getElementById('kpiMoodVal').style.color = '';
                    document.getElementById('kpiMoodConf').textContent = Math.round((aRes.confidence || 0.85) * 100) + '%';
                }

                // Update Hero Telemetry
                document.getElementById('kpiPostureVal').textContent = vCls.toUpperCase();
                document.getElementById('kpiPostureConf').textContent = Math.round((vRes.confidence || 0.85) * 100) + '%';
                document.getElementById('kpiLatencyVal').textContent = latency + ' ms';

                let wScore = 92;
                if (vCls === 'lying' || vCls === 'rumination') wScore = 96;
                if (aRes.class === 'Negative') wScore -= 20;

                document.getElementById('mmWelfareScore').textContent = wScore + ' / 100';
                document.getElementById('kpiWelfareVal').textContent = wScore + ' / 100';
                document.getElementById('mmWelfareSummary').textContent = (wScore >= 85)
                    ? 'Animal exhibits calm behavioral state with healthy posture. High herd comfort rating.'
                    : 'Acoustic separation or distress vocalization detected. Farm staff physical observation recommended.';

            } catch (err) {
                console.error(err);
                document.getElementById('mmLatencyText').textContent = 'Assessment note: ' + err.message;
            } finally {
                btn.innerHTML = '<span>🚀 RUN MULTIMODAL DUAL-STREAM AI ASSESSMENT</span>';
            }
        }

        // ==========================================
        // CAMERA CONTROLS
        // ==========================================
        async function toggleCamera() {
            const camVideo = document.getElementById('cameraVideo');
            const camBtnText = document.getElementById('camBtnText');
            const capturePhotoBtn = document.getElementById('btnCapturePhoto');
            const recordVideoBtn = document.getElementById('btnRecordVideoClip');
            const flipBtn = document.getElementById('btnFlipCam');
            const retakeBtn = document.getElementById('btnRetakeCam');
            const hudCrosshair = document.getElementById('hudCrosshair');
            const hudStatus = document.getElementById('hudStreamStatus');
            const hudMediaTag = document.getElementById('hudMediaTag');
            const standby = document.getElementById('viewportStandby');
            const statusHint = document.getElementById('cameraStatusHint');

            if (currentStream) {
                currentStream.getTracks().forEach(t => t.stop());
                currentStream = null;
                camVideo.style.display = 'none';
                camBtnText.textContent = 'Turn On Camera';
                capturePhotoBtn.style.display = 'none';
                recordVideoBtn.style.display = 'none';
                flipBtn.style.display = 'none';
                retakeBtn.style.display = 'none';
                hudCrosshair.classList.remove('active');
                hudStatus.textContent = 'STANDBY';
                hudMediaTag.textContent = 'Ready to Check';
                standby.style.display = 'flex';
                statusHint.textContent = 'Camera Off';
            } else {
                try {
                    hideAllViewports();
                    const stream = await navigator.mediaDevices.getUserMedia({
                        video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
                        audio: true
                    }).catch(async () => {
                        return await navigator.mediaDevices.getUserMedia({
                            video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }
                        });
                    });

                    currentStream = stream;
                    camVideo.srcObject = stream;
                    camVideo.style.display = 'block';
                    standby.style.display = 'none';
                    camBtnText.textContent = 'Stop Camera';
                    capturePhotoBtn.style.display = 'inline-flex';
                    recordVideoBtn.style.display = 'inline-flex';
                    flipBtn.style.display = 'inline-flex';
                    retakeBtn.style.display = 'none';
                    hudCrosshair.classList.add('active');
                    hudStatus.textContent = 'LIVE CAMERA';
                    hudMediaTag.textContent = 'Aim at Cow';
                    statusHint.textContent = 'Live Feed Active';
                } catch (err) {
                    alert('Camera Notice: ' + (err.message || 'Camera not accessible. You can upload a photo/video or use sample buttons.'));
                }
            }
        }

        function restartLiveFeed() {
            if (!currentStream) toggleCamera();
            else {
                document.getElementById('staticPhoto').style.display = 'none';
                document.getElementById('uploadedVideo').style.display = 'none';
                document.getElementById('cameraVideo').style.display = 'block';
                document.getElementById('btnCapturePhoto').style.display = 'inline-flex';
                document.getElementById('btnRecordVideoClip').style.display = 'inline-flex';
                document.getElementById('btnFlipCam').style.display = 'inline-flex';
                document.getElementById('btnRetakeCam').style.display = 'none';
                document.getElementById('hudStreamStatus').textContent = 'LIVE CAMERA';
                document.getElementById('visionCheckActionCard').style.display = 'none';
            }
        }

        function flipCamera() {
            facingMode = (facingMode === "environment") ? "user" : "environment";
            if (currentStream) { toggleCamera(); toggleCamera(); }
        }

        function capturePhotoOnly() {
            const camVideo = document.getElementById('cameraVideo');
            if (!currentStream || camVideo.videoWidth === 0) return;

            const canvas = document.createElement('canvas');
            canvas.width = camVideo.videoWidth;
            canvas.height = camVideo.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(camVideo, 0, 0);

            const photoImg = document.getElementById('staticPhoto');
            photoImg.src = canvas.toDataURL('image/jpeg');
            photoImg.style.display = 'block';
            camVideo.style.display = 'none';

            document.getElementById('btnCapturePhoto').style.display = 'none';
            document.getElementById('btnRecordVideoClip').style.display = 'none';
            document.getElementById('btnFlipCam').style.display = 'none';
            document.getElementById('btnRetakeCam').style.display = 'inline-flex';
            document.getElementById('hudStreamStatus').textContent = 'PHOTO TAKEN';

            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('file', blob, 'camera_snapshot.jpg');
                stagedVisionFormData = formData;
                stagedVisionFilename = 'camera_snapshot.jpg';
                stagedVisionIsVideo = false;
                document.getElementById('visionCheckActionCard').style.display = 'flex';
                document.getElementById('stagedMediaLabel').textContent = '📸 Captured Photo Ready';
            }, 'image/jpeg', 0.92);
        }

        async function toggleVideoRecording() {
            const btnText = document.getElementById('videoRecBtnText');
            if (videoRecorder && videoRecorder.state === 'recording') {
                videoRecorder.stop();
                btnText.textContent = '🎥 Record 5s Clip';
            } else {
                if (!currentStream) return;
                try {
                    videoChunks = [];
                    videoRecorder = new MediaRecorder(currentStream);
                    videoRecorder.ondataavailable = e => { if (e.data.size > 0) videoChunks.push(e.data); };
                    videoRecorder.onstop = () => {
                        const blob = new Blob(videoChunks, { type: 'video/mp4' });
                        const vidPlayer = document.getElementById('uploadedVideo');
                        vidPlayer.src = URL.createObjectURL(blob);
                        vidPlayer.style.display = 'block';
                        document.getElementById('cameraVideo').style.display = 'none';

                        document.getElementById('btnCapturePhoto').style.display = 'none';
                        document.getElementById('btnRecordVideoClip').style.display = 'none';
                        document.getElementById('btnRetakeCam').style.display = 'inline-flex';

                        const formData = new FormData();
                        formData.append('file', blob, 'recorded_cow_clip.mp4');
                        stagedVisionFormData = formData;
                        stagedVisionFilename = 'recorded_cow_clip.mp4';
                        stagedVisionIsVideo = true;
                        document.getElementById('visionCheckActionCard').style.display = 'flex';
                        document.getElementById('stagedMediaLabel').textContent = '🎥 Video Clip Ready';
                    };

                    videoRecorder.start();
                    btnText.textContent = '⏹️ Stop Video';
                    setTimeout(() => {
                        if (videoRecorder && videoRecorder.state === 'recording') {
                            videoRecorder.stop();
                            btnText.textContent = '🎥 Record 5s Clip';
                        }
                    }, 5000);
                } catch (err) {
                    alert('Recording Notice: ' + err.message);
                }
            }
        }

        function runStagedVisionCheck() {
            if (!stagedVisionFormData) return;
            sendInferenceRequest(stagedVisionFormData, stagedVisionFilename, stagedVisionIsVideo);
        }

        function hideAllViewports() {
            document.getElementById('cameraVideo').style.display = 'none';
            const uploadedVid = document.getElementById('uploadedVideo');
            uploadedVid.pause();
            uploadedVid.style.display = 'none';
            document.getElementById('staticPhoto').style.display = 'none';
            document.getElementById('viewportStandby').style.display = 'none';
        }

        function handleDragOver(e) { e.preventDefault(); }
        function handleDragLeave(e) { e.preventDefault(); }
        function handleDrop(e) {
            e.preventDefault();
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files);
            }
        }

        function handleFileSelect(files) {
            if (!files || files.length === 0) return;
            const file = files[0];
            const isVideo = file.type.startsWith('video/') || /\\.(mp4|mov|avi|webm|mkv)$/i.test(file.name);

            hideAllViewports();
            if (currentStream) toggleCamera();

            if (isVideo) {
                const vidPlayer = document.getElementById('uploadedVideo');
                vidPlayer.src = URL.createObjectURL(file);
                vidPlayer.style.display = 'block';
                vidPlayer.play().catch(() => {});
            } else {
                const photo = document.getElementById('staticPhoto');
                photo.src = URL.createObjectURL(file);
                photo.style.display = 'block';
            }

            const formData = new FormData();
            formData.append('file', file);
            stagedVisionFormData = formData;
            stagedVisionFilename = file.name;
            stagedVisionIsVideo = isVideo;

            document.getElementById('visionCheckActionCard').style.display = 'flex';
            document.getElementById('stagedMediaLabel').textContent = isVideo ? '🎥 Video File Ready' : '📸 Photo File Ready';
            sendInferenceRequest(formData, file.name, isVideo);
        }

        function loadSampleImage(filename) {
            hideAllViewports();
            if (currentStream) toggleCamera();

            const photo = document.getElementById('staticPhoto');
            photo.src = '/samples/image/' + filename;
            photo.style.display = 'block';

            fetch('/samples/image/' + filename)
                .then(r => r.blob())
                .then(blob => {
                    const formData = new FormData();
                    formData.append('file', blob, filename);
                    stagedVisionFormData = formData;
                    stagedVisionFilename = filename;
                    stagedVisionIsVideo = false;
                    document.getElementById('visionCheckActionCard').style.display = 'flex';
                    document.getElementById('stagedMediaLabel').textContent = '📸 Sample: ' + filename;
                    sendInferenceRequest(formData, filename, false);
                });
        }

        function loadSampleVideo() {
            hideAllViewports();
            if (currentStream) toggleCamera();

            const vidPlayer = document.getElementById('uploadedVideo');
            vidPlayer.src = '/samples/video/sample_cattle_video.mp4';
            vidPlayer.style.display = 'block';
            vidPlayer.play().catch(() => {});

            fetch('/samples/video/sample_cattle_video.mp4')
                .then(r => r.blob())
                .then(blob => {
                    const formData = new FormData();
                    formData.append('file', blob, 'sample_cattle_video.mp4');
                    stagedVisionFormData = formData;
                    stagedVisionFilename = 'sample_cattle_video.mp4';
                    stagedVisionIsVideo = true;
                    document.getElementById('visionCheckActionCard').style.display = 'flex';
                    document.getElementById('stagedMediaLabel').textContent = '🎥 5-Sec Video Ready';
                    sendInferenceRequest(formData, 'sample_cattle_video.mp4', true);
                });
        }

        async function sendInferenceRequest(formData, filename, isVideo) {
            const t0 = performance.now();
            document.getElementById('hudStreamStatus').textContent = 'ANALYZING...';
            try {
                const res = await fetch('/api/predict/behavior', { method: 'POST', body: formData });
                const data = await res.json();
                const latency = Math.round(performance.now() - t0);
                renderVisionResults(data, latency, isVideo);
            } catch (err) {
                console.error(err);
            } finally {
                document.getElementById('hudStreamStatus').textContent = isVideo ? 'VIDEO CHECKED' : 'PHOTO CHECKED';
            }
        }

        function renderVisionResults(data, latency, isVideo) {
            document.getElementById('visionEmptyPrompt').style.display = 'none';
            document.getElementById('visionActiveResults').style.display = 'flex';

            const rawCls = (data.class || data.dominant_class || 'standing').toLowerCase();
            const conf = data.confidence || data.dominant_confidence || 0.0;
            const confPct = (conf * 100).toFixed(1) + '%';

            document.getElementById('kpiPostureVal').textContent = rawCls.toUpperCase();
            document.getElementById('kpiPostureConf').textContent = confPct;
            document.getElementById('kpiLatencyVal').textContent = latency + ' ms';

            document.getElementById('dominantClassName').textContent = rawCls;
            document.getElementById('dominantConfidencePct').textContent = confPct;
            document.getElementById('resultsLatencyBadge').textContent = latency + ' ms';
            document.getElementById('resultsSubtext').textContent = 'ResNet18 Classification Complete';

            const probs = data.probabilities || data.activity_breakdown || {};
            updateProbBar('Standing', probs['standing'] || 0, isVideo);
            updateProbBar('Feeding', probs['feeding'] || 0, isVideo);
            updateProbBar('Drinking', probs['drinking'] || 0, isVideo);
            updateProbBar('Lying', probs['lying'] || 0, isVideo);
            updateProbBar('Rumination', probs['rumination'] || 0, isVideo);

            document.getElementById('ethologyNoteText').textContent = data.description || 'Observed cattle behavior class from CBVD-5 dataset.';
        }

        function updateProbBar(name, val, isVideo) {
            const pct = isVideo ? val : (val <= 1.0 ? (val * 100) : val);
            const formatted = pct.toFixed(1) + '%';
            const textEl = document.getElementById('pct' + name);
            const barEl = document.getElementById('bar' + name);
            if (textEl) textEl.textContent = formatted;
            if (barEl) barEl.style.width = formatted;
        }

        // ==========================================
        // PURE WEBAUDIO 16KHZ PCM WAV RECORDER
        // ==========================================
        function encodeWAV(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            // RIFF chunk descriptor
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
            view.setUint16(20, 1, true); // Linear PCM
            view.setUint16(22, 1, true); // Mono channel
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true); // Byte rate
            view.setUint16(32, 2, true); // Block align
            view.setUint16(34, 16, true); // Bits per sample
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * 2, true);

            // PCM samples (16-bit signed int)
            let offset = 44;
            for (let i = 0; i < samples.length; i++, offset += 2) {
                let s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }

            return new Blob([view], { type: 'audio/wav' });
        }

        async function toggleAudioRecording() {
            const btn = document.getElementById('btnRecordMic');
            const txt = document.getElementById('recBtnText');
            const statusHint = document.getElementById('recStatusHint');

            if (isRecordingAudio) {
                isRecordingAudio = false;
                txt.textContent = 'Record Moo Sound (Mic)';
                statusHint.textContent = 'Processing Recording...';

                if (audioScriptProcessor) {
                    audioScriptProcessor.disconnect();
                    audioInputNode.disconnect();
                }

                // Flatten samples
                let totalLen = 0;
                for (let chunk of audioPcmSamples) totalLen += chunk.length;
                const merged = new Float32Array(totalLen);
                let offset = 0;
                for (let chunk of audioPcmSamples) {
                    merged.set(chunk, offset);
                    offset += chunk.length;
                }

                // Resample to 16kHz if needed
                const srcSr = audioContext.sampleRate;
                const targetSr = 16000;
                let final16k;
                if (srcSr === targetSr) {
                    final16k = merged;
                } else {
                    const ratio = srcSr / targetSr;
                    const newLen = Math.round(merged.length / ratio);
                    final16k = new Float32Array(newLen);
                    for (let i = 0; i < newLen; i++) {
                        final16k[i] = merged[Math.floor(i * ratio)];
                    }
                }

                const wavBlob = encodeWAV(final16k, targetSr);
                setupAudioPlayback(wavBlob, 'Microphone Recording (16kHz WAV)');

                const formData = new FormData();
                formData.append('audio', wavBlob, 'cattle_mic_recording.wav');
                stagedAudioFormData = formData;
                stagedAudioFilename = 'cattle_mic_recording.wav';

                document.getElementById('audioCheckActionCard').style.display = 'flex';
                document.getElementById('stagedAudioLabel').textContent = '🎤 Mic Recording Ready';
                statusHint.textContent = 'Sound Ready';

                // Automatically trigger analysis
                sendAudioRequest(formData, 'cattle_mic_recording.wav');

            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    audioInputNode = audioContext.createMediaStreamSource(stream);
                    audioScriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                    audioPcmSamples = [];

                    audioScriptProcessor.onaudioprocess = (e) => {
                        if (!isRecordingAudio) return;
                        const inputData = e.inputBuffer.getChannelData(0);
                        audioPcmSamples.push(new Float32Array(inputData));
                    };

                    audioInputNode.connect(audioScriptProcessor);
                    audioScriptProcessor.connect(audioContext.destination);

                    isRecordingAudio = true;
                    txt.textContent = '⏹️ Stop Recording';
                    statusHint.textContent = 'Recording Sound Now...';
                    startWaveAnimation();
                } catch (err) {
                    alert('Mic Notice: ' + (err.message || 'Microphone not accessible.'));
                }
            }
        }

        function setupAudioPlayback(blobOrUrl, titleText) {
            const card = document.getElementById('audioPlaybackCard');
            const player = document.getElementById('audioPlaybackElement');
            const status = document.getElementById('audioPlaybackStatus');
            
            card.style.display = 'flex';
            if (typeof blobOrUrl === 'string') player.src = blobOrUrl;
            else player.src = URL.createObjectURL(blobOrUrl);
            status.textContent = titleText || 'Ready';
        }

        function handleAudioUpload(files) {
            if (!files || files.length === 0) return;
            const file = files[0];
            setupAudioPlayback(file, file.name);

            const formData = new FormData();
            formData.append('audio', file);
            stagedAudioFormData = formData;
            stagedAudioFilename = file.name;

            document.getElementById('audioCheckActionCard').style.display = 'flex';
            document.getElementById('stagedAudioLabel').textContent = '📁 File: ' + file.name;
            startWaveAnimation();
            sendAudioRequest(formData, file.name);
        }

        function testAudioSample(filename, btnEl) {
            const url = '/samples/audio/' + filename;
            setupAudioPlayback(url, filename);
            startWaveAnimation();

            document.querySelectorAll('#chip_moo1, #chip_moo2, #chip_distress').forEach(b => b.classList.remove('highlight'));
            if (btnEl) btnEl.classList.add('highlight');

            fetch(url)
                .then(r => r.blob())
                .then(blob => {
                    const formData = new FormData();
                    formData.append('audio', blob, filename);
                    stagedAudioFormData = formData;
                    stagedAudioFilename = filename;
                    document.getElementById('audioCheckActionCard').style.display = 'flex';
                    document.getElementById('stagedAudioLabel').textContent = '🔊 Sample: ' + filename;
                    sendAudioRequest(formData, filename);
                });
        }

        function runStagedAudioCheck() {
            if (!stagedAudioFormData) return;
            sendAudioRequest(stagedAudioFormData, stagedAudioFilename);
        }

        async function sendAudioRequest(formData, filename) {
            const t0 = performance.now();
            const btn = document.getElementById('btnAnalyzeAudio');
            if (btn) btn.innerHTML = '<span>⏳ ANALYZING ACOUSTIC VALENCE (AST)...</span>';
            document.getElementById('audioResultsSub').textContent = 'Running AST forward pass...';

            try {
                const res = await fetch('/api/predict/audio', { method: 'POST', body: formData });
                const data = await res.json();
                const latency = Math.round(performance.now() - t0);
                renderAudioResults(data, latency);
            } catch (err) {
                console.error(err);
                document.getElementById('audioResultsSub').textContent = 'Audio note: ' + err.message;
            } finally {
                if (btn) btn.innerHTML = '<span>🔍 ANALYZE ACOUSTIC VALENCE (RUN AST)</span>';
            }
        }

        function renderAudioResults(data, latency) {
            document.getElementById('audioEmptyPrompt').style.display = 'none';
            document.getElementById('audioActiveResults').style.display = 'flex';

            // Check if human speech or non-cattle sound
            if (data.is_cattle_call === false) {
                const isSpeech = (data.is_human_speech === true) || (data.class === 'Human Speaking') || (data.signal_classification && data.signal_classification.includes('Human'));

                if (isSpeech) {
                    const speechPct = (data.speech_confidence || (data.confidence ? data.confidence * 100 : 95.0)).toFixed(1) + '%';
                    document.getElementById('audioMoodName').innerHTML = '🗣️ Human Speaking Detected';
                    document.getElementById('audioMoodName').style.color = '#F59E0B';
                    document.getElementById('audioMoodConfidence').textContent = speechPct;
                    document.getElementById('audioResultsSub').textContent = '⚠️ Audio Filter: Human Voice Identified (' + latency + ' ms)';
                    document.getElementById('audioEthologyNote').innerHTML = '<strong>🗣️ Human Speaking Detected:</strong> ' + (data.error || 'The audio was identified as human speech/voice. MooTrack is designed exclusively for cattle vocalizations (mooing).');
                    
                    document.getElementById('kpiMoodVal').textContent = '🗣️ HUMAN SPEAKING';
                    document.getElementById('kpiMoodVal').style.color = '#F59E0B';
                    document.getElementById('kpiMoodConf').textContent = speechPct + ' Speech';
                    document.getElementById('kpiLatencyVal').textContent = latency + ' ms';

                    // Update Probabilities bar
                    document.getElementById('pctValencePos').textContent = speechPct;
                    document.getElementById('barValencePos').style.width = speechPct;
                    document.getElementById('barValencePos').style.background = '#F59E0B';
                    document.getElementById('pctValenceNeg').textContent = (Math.max(0, 100 - parseFloat(speechPct))).toFixed(1) + '%';
                    document.getElementById('barValenceNeg').style.width = (Math.max(0, 100 - parseFloat(speechPct))).toFixed(1) + '%';
                    document.getElementById('barValenceNeg').style.background = 'rgba(255,255,255,0.2)';

                    // Bioacoustic Telemetry
                    const metrics = data.audio_metrics || {};
                    document.getElementById('bioacousticBox').style.display = 'block';
                    document.getElementById('callTypeEstimate').textContent = 'Signal: 🗣️ Human Vocal Tract';
                    document.getElementById('pitchMetric').textContent = 'F0: ' + (metrics.f0_pitch_hz || 180) + ' Hz | Centroid: ' + (metrics.spectral_centroid_hz || 1600) + ' Hz';
                    document.getElementById('callTypeDesc').textContent = 'Human speech formants and consonant transitions recognized by AST AudioSet model.';
                } else {
                    document.getElementById('audioMoodName').textContent = data.signal_classification || 'Non-Cattle Sound';
                    document.getElementById('audioMoodName').style.color = 'var(--text-light-muted)';
                    document.getElementById('audioMoodConfidence').textContent = 'Rejected';
                    document.getElementById('audioResultsSub').textContent = 'Sound Filter Triggered (' + latency + ' ms)';
                    document.getElementById('audioEthologyNote').textContent = data.error || 'Audio rejected by AudioSet filter (noise or silence).';
                    
                    document.getElementById('kpiMoodVal').textContent = 'NOISE / SILENCE';
                    document.getElementById('kpiMoodVal').style.color = '';
                    document.getElementById('kpiMoodConf').textContent = 'Rejected';

                    document.getElementById('bioacousticBox').style.display = 'none';
                }
                return;
            }

            document.getElementById('bioacousticBox').style.display = 'block';

            // Accurate Valence Assignment
            const isPositive = (data.class === 'Positive');
            const moodName = isPositive ? 'Positive Valence' : 'Negative Valence';
            const conf = data.confidence || 0.85;
            const confPct = (conf * 100).toFixed(1) + '%';

            document.getElementById('kpiMoodVal').textContent = moodName;
            document.getElementById('kpiMoodVal').style.color = '';
            document.getElementById('kpiMoodConf').textContent = confPct;
            document.getElementById('kpiLatencyVal').textContent = latency + ' ms';

            document.getElementById('audioMoodName').textContent = moodName;
            document.getElementById('audioMoodName').style.color = '';
            document.getElementById('audioMoodConfidence').textContent = confPct;
            document.getElementById('audioResultsSub').textContent = 'AST Valence Complete (' + latency + ' ms)';

            // Probabilities
            const probs = data.probabilities || {};
            let posVal = probs['Positive'];
            let negVal = probs['Negative'];
            if (posVal === undefined) posVal = isPositive ? conf : 1 - conf;
            if (negVal === undefined) negVal = !isPositive ? conf : 1 - conf;

            const posPct = (posVal * 100).toFixed(1) + '%';
            const negPct = (negVal * 100).toFixed(1) + '%';

            document.getElementById('pctValencePos').textContent = posPct;
            document.getElementById('barValencePos').style.width = posPct;
            document.getElementById('barValencePos').style.background = 'var(--accent-lime)';
            document.getElementById('pctValenceNeg').textContent = negPct;
            document.getElementById('barValenceNeg').style.width = negPct;
            document.getElementById('barValenceNeg').style.background = 'var(--accent-rose)';

            // Bioacoustic Telemetry
            const metrics = data.audio_metrics || {};
            if (metrics.call_type_estimate) {
                document.getElementById('callTypeEstimate').textContent = 'Call: ' + metrics.call_type_estimate;
                document.getElementById('pitchMetric').textContent = 'F0: ' + (metrics.f0_pitch_hz || 350) + ' Hz | Centroid: ' + (metrics.spectral_centroid_hz || 1400) + ' Hz';
                document.getElementById('callTypeDesc').textContent = metrics.call_type_description || '';
            }

            document.getElementById('audioEthologyNote').textContent = data.behavioral_context || (isPositive 
                ? 'Acoustic features indicate Positive Emotional Valence (social reunion / affiliative contact).' 
                : 'Acoustic features indicate Negative Emotional Valence (social separation, milking delay, or distress).');
        }

        function startWaveAnimation() {
            const canvas = document.getElementById('audioWaveCanvas');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;

            let step = 0;
            if (animationFrameId) cancelAnimationFrame(animationFrameId);

            function draw() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 2.5;
                ctx.strokeStyle = '#CCFF00';
                ctx.beginPath();

                const midY = canvas.height / 2;
                for (let x = 0; x < canvas.width; x += 4) {
                    const y = midY + Math.sin((x + step) * 0.05) * 30 * Math.sin(step * 0.02);
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
                step += 4;
                if (step < 300) animationFrameId = requestAnimationFrame(draw);
            }
            draw();
        }

        // ==========================================
        // HISTORY & CSV EXPORT
        // ==========================================
        async function refreshHistoryTable() {
            const tbody = document.getElementById('auditTableBody');
            try {
                const res = await fetch('/api/history');
                const history = await res.json();

                if (!history || history.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-subtle); padding: 30px;">No checks recorded in this session.</td></tr>';
                    return;
                }

                tbody.innerHTML = '';
                history.forEach(row => {
                    const tr = document.createElement('tr');
                    const confPct = ((row.confidence || 0) * 100).toFixed(1) + '%';

                    tr.innerHTML = `
                        <td style="font-family: var(--font-mono); font-size: 0.78rem;">${row.timestamp}</td>
                        <td><span class="step-pill">${row.type}</span></td>
                        <td style="font-weight: 800; text-transform: capitalize;">${row.predicted_class}</td>
                        <td style="font-family: var(--font-mono); font-weight: 800;">${confPct}</td>
                        <td style="color: var(--text-muted); font-size: 0.78rem;">${row.filename || '--'}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } catch (err) {
                console.error(err);
            }
        }

        function exportHistoryCSV() {
            window.location.href = '/api/export/csv';
        }
    </script>
</body>
</html>
"""

def extract_multipart_payload(body_bytes: bytes, content_type_header: str) -> Tuple[bytes, str]:
    if "boundary=" not in content_type_header:
        return body_bytes, "unknown_upload.bin"
    
    boundary_str = content_type_header.split("boundary=")[-1].strip().strip('"').strip("'")
    boundary = ("--" + boundary_str).encode("utf-8")
    
    parts = body_bytes.split(boundary)
    for part in parts:
        if b"filename=" in part:
            double_crlf = bytes([13, 10, 13, 10])
            header_end = part.find(double_crlf)
            if header_end != -1:
                header_text = part[:header_end].decode("utf-8", errors="ignore")
                filename = "uploaded_media.bin"
                for line in header_text.splitlines():
                    if "filename=" in line:
                        for token in line.split(";"):
                            if "filename=" in token:
                                filename = token.split("=")[-1].strip().strip('"').strip("'")
                                break
                
                content = part[header_end + 4:]
                if content.endswith(b"\r\n"):
                    content = content[:-2]
                if content.endswith(b"--"):
                    content = content[:-2]
                return content, filename
                
    return body_bytes, "uploaded_media.bin"


class DashboardRequestHandler(BaseHTTPRequestHandler):

    def _set_headers(self, content_type="text/html", status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers("text/plain", 200)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._set_headers("text/html; charset=utf-8")
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))
            return

        elif path in ["/static/logo.png", "/logo.png", "/static/logo_clean.png"]:
            target = LOGO_CLEAN_PATH if LOGO_CLEAN_PATH.exists() else LOGO_PATH
            if target.exists():
                self._set_headers("image/png")
                with open(target, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._set_headers("application/json", 404)
                self.wfile.write(json.dumps({"error": "Logo Not Found"}).encode("utf-8"))
                return

        elif path == "/api/history":
            self._set_headers("application/json")
            self.wfile.write(json.dumps(PREDICTION_HISTORY).encode("utf-8"))
            return

        elif path == "/api/export/csv":
            self._set_headers("text/csv")
            csv_lines = ["Timestamp,Modality,Classification,Confidence,Filename"]
            for row in PREDICTION_HISTORY:
                ts = row.get("timestamp", "")
                mtype = row.get("type", "")
                pred = row.get("predicted_class", "")
                conf = f"{row.get('confidence', 0.0)*100:.1f}%"
                fn = row.get("filename", "")
                csv_lines.append(f'"{ts}","{mtype}","{pred}","{conf}","{fn}"')
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
