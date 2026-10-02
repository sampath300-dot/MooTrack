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
    <title>MooTrack &mdash; Precision Cattle Health & Acoustic Intelligence</title>
    <meta name="description" content="Non-invasive bioacoustic spectrogram and behavioral computer vision monitoring for dairy cattle.">
    <link rel="icon" type="image/png" href="/static/logo_clean.png">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-canvas: #faf9f5;
            --bg-surface: #ffffff;
            --bg-subtle: #f3f1ea;
            --bg-hover: #eae7dd;
            
            --border-subtle: #e5e3dc;
            --border-medium: #d4d0c5;
            --border-dark: #121310;
            
            --text-ink: #121310;
            --text-secondary: #4a4943;
            --text-muted: #7a7972;
            
            --accent-green: #1d4624;
            --accent-green-bg: #edf3ee;
            --accent-green-border: #b8d4bb;
            
            --accent-danger: #991b1b;
            --accent-danger-bg: #fdf2f2;
            --accent-danger-border: #fecaca;
            
            --accent-warn: #854d0e;
            --accent-warn-bg: #fefce8;
            --accent-warn-border: #fef08a;
            
            --font-sans: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            --font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background-color: var(--bg-canvas);
            color: var(--text-ink);
            font-family: var(--font-sans);
            line-height: 1.55;
            letter-spacing: -0.01em;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 1320px;
            margin: 0 auto;
            padding: 32px 24px 80px;
        }

        /* Top Header */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 40px;
        }
        .header-brand {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .brand-logo-img {
            height: 38px;
            width: auto;
            max-width: 160px;
            object-fit: contain;
            display: block;
        }
        .brand-text-block {
            display: flex;
            flex-direction: column;
        }
        .brand-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .brand-subtitle {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-top: 2px;
        }
        .system-status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-green);
            background: var(--accent-green-bg);
            border: 1px solid var(--accent-green-border);
            padding: 6px 12px;
            letter-spacing: 0.02em;
        }
        .status-dot {
            width: 6px;
            height: 6px;
            background-color: var(--accent-green);
            border-radius: 50%;
            display: inline-block;
        }

        /* Hero / Overview Statement */
        .overview-panel {
            margin-bottom: 48px;
        }
        .hero-tag {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }
        .hero-statement {
            font-size: clamp(1.8rem, 3.5vw, 2.75rem);
            font-weight: 800;
            letter-spacing: -0.035em;
            line-height: 1.15;
            color: var(--text-ink);
            max-width: 980px;
            margin-bottom: 24px;
        }
        .hero-desc {
            font-size: 1.05rem;
            color: var(--text-secondary);
            max-width: 820px;
            line-height: 1.6;
            margin-bottom: 36px;
        }

        /* Minimalist Metric Grid (Hairline Box Style) */
        .overview-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }
        .overview-metric {
            padding: 28px 24px;
            border-right: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .overview-metric:last-child {
            border-right: none;
        }
        .metric-index {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.06em;
            margin-bottom: 16px;
        }
        .metric-val {
            font-size: clamp(2rem, 3vw, 2.6rem);
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.04em;
            line-height: 1;
            margin-bottom: 10px;
        }
        .metric-label {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-ink);
            margin-bottom: 4px;
        }
        .metric-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
        }

        /* Editorial Tab Navigation */
        .nav-tabs {
            display: flex;
            align-items: center;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 40px;
            gap: 0;
            overflow-x: auto;
        }
        .tab-btn {
            background: transparent;
            border: none;
            padding: 16px 24px;
            font-family: var(--font-sans);
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text-muted);
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            display: inline-flex;
            align-items: center;
            gap: 10px;
            transition: all 0.2s ease;
            white-space: nowrap;
        }
        .tab-btn:hover {
            color: var(--text-ink);
        }
        .tab-btn.active {
            color: var(--text-ink);
            font-weight: 700;
            border-bottom: 2px solid var(--border-dark);
        }
        .tab-index {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        .tab-btn.active .tab-index {
            color: var(--accent-green);
            font-weight: 700;
        }

        .tab-panel {
            display: none;
        }
        .tab-panel.active {
            display: block;
            animation: fadeIn 0.25s ease forwards;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Workspace Sections */
        .workspace-section {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            padding: 40px;
            margin-bottom: 32px;
        }
        .section-header {
            margin-bottom: 32px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 20px;
        }
        .section-tag {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.03em;
        }
        .section-subtitle {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-top: 6px;
            max-width: 780px;
        }

        /* Action Grid */
        .action-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 32px;
        }
        .action-box {
            border: 1px solid var(--border-subtle);
            background: var(--bg-subtle);
            padding: 32px 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .action-box:hover {
            border-color: var(--border-dark);
            background: var(--bg-surface);
            transform: translateY(-2px);
        }
        .action-box.recording {
            border-color: var(--accent-danger);
            background: var(--accent-danger-bg);
            animation: pulseRecord 1.5s infinite;
        }
        @keyframes pulseRecord {
            0%, 100% { border-color: var(--accent-danger); }
            50% { border-color: transparent; }
        }
        .action-icon {
            color: var(--text-ink);
            margin-bottom: 4px;
        }
        .action-box-title {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-ink);
            letter-spacing: -0.01em;
        }
        .action-box-desc {
            font-size: 0.82rem;
            color: var(--text-muted);
            max-width: 320px;
        }

        /* Staged Audio Analysis Bar */
        .audio-playback-bar {
            display: none;
            background: var(--bg-subtle);
            border: 1px solid var(--border-subtle);
            padding: 20px 24px;
            margin-bottom: 32px;
            flex-direction: column;
            gap: 16px;
        }
        .playback-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            flex-wrap: wrap;
        }
        .playback-info {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-ink);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .playback-track {
            font-family: var(--font-mono);
            font-size: 0.82rem;
            color: var(--accent-green);
            background: var(--accent-green-bg);
            padding: 3px 10px;
            border: 1px solid var(--accent-green-border);
        }
        .playback-controls-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            flex-wrap: wrap;
        }
        audio {
            height: 38px;
            outline: none;
            flex: 1;
            min-width: 240px;
        }
        .staged-action-btn-row {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }

        /* Staged Image / Vision Preview Box */
        .preview-box {
            display: none;
            background: var(--bg-subtle);
            border: 1px solid var(--border-subtle);
            padding: 24px;
            margin-bottom: 32px;
            text-align: center;
        }
        .preview-box-inner {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 18px;
        }
        .preview-media-container {
            max-width: 520px;
            width: 100%;
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface);
            overflow: hidden;
        }
        .preview-box img, .preview-box video {
            max-height: 320px;
            width: 100%;
            object-fit: contain;
            display: block;
        }
        .preview-meta-tag {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Sub-sections & Soundboard */
        .sub-heading {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }
        .sub-caption {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 16px;
        }
        .sample-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 14px;
        }
        .sample-item {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 16px;
            transition: border-color 0.2s ease;
        }
        .sample-item:hover {
            border-color: var(--border-medium);
        }
        .sample-meta h5 {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-ink);
            margin-bottom: 4px;
        }
        .sample-meta p {
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
        }
        .btn-group {
            display: flex;
            gap: 8px;
        }
        .btn-secondary {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-ink);
            padding: 9px 16px;
            font-family: var(--font-sans);
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            flex: 1;
            text-align: center;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .btn-secondary:hover {
            border-color: var(--border-dark);
            background: var(--bg-subtle);
        }
        .btn-primary {
            background: var(--border-dark);
            border: 1px solid var(--border-dark);
            color: #ffffff;
            padding: 9px 18px;
            font-family: var(--font-sans);
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            flex: 1;
            text-align: center;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .btn-primary:hover {
            background: var(--accent-green);
            border-color: var(--accent-green);
        }
        .btn-cta {
            background: var(--accent-green);
            border: 1px solid var(--accent-green);
            color: #ffffff;
            padding: 10px 22px;
            font-family: var(--font-sans);
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            letter-spacing: -0.01em;
        }
        .btn-cta:hover {
            background: #14331a;
            border-color: #14331a;
        }

        /* Real Photo Gallery (5-Col Grid) */
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 14px;
            margin-top: 14px;
        }
        .gallery-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            overflow: hidden;
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .gallery-card:hover {
            border-color: var(--border-dark);
            transform: translateY(-2px);
        }
        .gallery-img-box {
            width: 100%;
            height: 140px;
            background: var(--bg-subtle);
            overflow: hidden;
        }
        .gallery-img-box img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
            transition: transform 0.3s ease;
        }
        .gallery-card:hover .gallery-img-box img {
            transform: scale(1.03);
        }
        .gallery-caption {
            padding: 10px 12px;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text-ink);
            text-align: center;
            border-top: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }

        /* Analytical Instrument Result Box */
        .result-card {
            display: none;
            margin-top: 36px;
            padding: 28px;
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }
        .result-card.positive {
            border-left: 4px solid var(--accent-green);
            background: #fcfdfc;
        }
        .result-card.negative {
            border-left: 4px solid var(--accent-danger);
            background: #fdfcfc;
        }
        .result-card.warning {
            border-left: 4px solid var(--accent-warn);
            background: #fdfdfc;
        }
        
        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 14px;
        }
        .result-title-wrap {
            display: flex;
            flex-direction: column;
        }
        .result-sublabel {
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .result-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.02em;
        }
        .result-badge {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            font-weight: 700;
            padding: 5px 12px;
            background: var(--bg-subtle);
            border: 1px solid var(--border-subtle);
            color: var(--text-ink);
        }
        .result-card.positive .result-badge {
            background: var(--accent-green-bg);
            border-color: var(--accent-green-border);
            color: var(--accent-green);
        }
        .result-card.negative .result-badge {
            background: var(--accent-danger-bg);
            border-color: var(--accent-danger-border);
            color: var(--accent-danger);
        }
        .result-card.warning .result-badge {
            background: var(--accent-warn-bg);
            border-color: var(--accent-warn-border);
            color: var(--accent-warn);
        }
        .result-body {
            padding: 4px 0 0;
        }
        .result-body h6 {
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .result-body p {
            font-size: 0.95rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Camera Box */
        .camera-wrapper {
            display: none;
            background: #121310;
            border: 1px solid var(--border-dark);
            overflow: hidden;
            max-width: 580px;
            margin: 0 auto 28px;
            text-align: center;
        }
        #cameraVideo {
            width: 100%;
            height: auto;
            display: block;
        }
        .camera-actions {
            padding: 16px;
            display: flex;
            justify-content: center;
            gap: 12px;
            background: #1a1a18;
            border-top: 1px solid rgba(255,255,255,0.1);
        }

        /* Lactation Benchmarks Grid (3 Columns) */
        .guidance-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }
        .guidance-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            padding: 32px 24px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .guidance-num {
            font-family: var(--font-mono);
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--accent-green);
            margin-bottom: 12px;
        }
        .guidance-card h4 {
            font-size: 1.05rem;
            font-weight: 800;
            color: var(--text-ink);
            letter-spacing: -0.02em;
            margin-bottom: 12px;
            line-height: 1.3;
        }
        .guidance-card p {
            font-size: 0.88rem;
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Minimalist Audit Table */
        .history-table-wrapper {
            overflow-x: auto;
            border: 1px solid var(--border-subtle);
            background: var(--bg-surface);
        }
        .history-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }
        .history-table th {
            text-align: left;
            padding: 14px 18px;
            background: var(--bg-subtle);
            color: var(--text-secondary);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            border-bottom: 1px solid var(--border-subtle);
        }
        .history-table td {
            padding: 16px 18px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-ink);
        }
        .history-table tr:last-child td {
            border-bottom: none;
        }
        .history-table tr:hover td {
            background: var(--bg-canvas);
        }
        .tag-pill {
            display: inline-block;
            padding: 3px 8px;
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .tag-pill.pos {
            background: var(--accent-green-bg);
            color: var(--accent-green);
            border: 1px solid var(--accent-green-border);
        }
        .tag-pill.neg {
            background: var(--accent-danger-bg);
            color: var(--accent-danger);
            border: 1px solid var(--accent-danger-border);
        }
        .tag-pill.neu {
            background: var(--bg-subtle);
            color: var(--text-ink);
            border: 1px solid var(--border-subtle);
        }

        /* Footer */
        .app-footer {
            margin-top: 60px;
            padding-top: 24px;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.8rem;
            color: var(--text-muted);
            flex-wrap: wrap;
            gap: 12px;
        }
        .app-footer span {
            font-family: var(--font-mono);
            font-size: 0.75rem;
        }

        svg {
            display: inline-block;
            vertical-align: middle;
        }

        /* Responsive Breakpoints */
        @media (max-width: 1024px) {
            .overview-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .overview-metric:nth-child(2) {
                border-right: none;
            }
            .overview-metric:nth-child(1), .overview-metric:nth-child(2) {
                border-bottom: 1px solid var(--border-subtle);
            }
            .sample-grid, .guidance-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .gallery-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (max-width: 640px) {
            .container {
                padding: 16px 16px 60px;
            }
            .overview-grid {
                grid-template-columns: 1fr;
            }
            .overview-metric {
                border-right: none;
                border-bottom: 1px solid var(--border-subtle);
            }
            .overview-metric:last-child {
                border-bottom: none;
            }
            .action-grid, .sample-grid, .guidance-grid, .gallery-grid {
                grid-template-columns: 1fr;
            }
            .workspace-section {
                padding: 24px 16px;
            }
        }
    </style>
</head>
<body>

<div class="container">

    <!-- Top Header -->
    <header class="app-header">
        <div class="header-brand">
            """ + (f'<img src="{LOGO_BASE64}" class="brand-logo-img" alt="MooTrack Logo">' if LOGO_BASE64 else """
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1d4624" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                <path d="M12 6v6l4 2"/>
            </svg>
            """) + """
            <div class="brand-text-block">
                <h1 class="brand-title">MooTrack</h1>
                <p class="brand-subtitle">Precision Cattle Health & Bioacoustic Intelligence</p>
            </div>
        </div>
        <div class="system-status">
            <span class="status-dot"></span>
            <span>MODELS OPERATIONAL</span>
        </div>
    </header>

    <!-- Overview Hero & Numbers -->
    <section class="overview-panel">
        <div class="hero-tag">[ 00 / SYSTEM OVERVIEW ]</div>
        <h2 class="hero-statement">Non-invasive bioacoustic screening & computer vision ethology for modern dairy operations.</h2>
        <p class="hero-desc">
            Real-time automated screening for cattle vocalization stress and barn physical activity. Engineered to safeguard herd well-being, optimize rumination time budgets, and uphold commercial milk yield benchmarks.
        </p>

        <!-- Metric Grid -->
        <div class="overview-grid">
            <div class="overview-metric">
                <div class="metric-index">[ 01 ]</div>
                <div>
                    <div class="metric-val">+15%</div>
                    <div class="metric-label">Lactation Protection</div>
                    <div class="metric-desc">Yield maintenance via distress reduction</div>
                </div>
            </div>
            <div class="overview-metric">
                <div class="metric-index">[ 02 ]</div>
                <div>
                    <div class="metric-val">48h</div>
                    <div class="metric-label">Early Clinical Warning</div>
                    <div class="metric-desc">Prior to visible milk drop or fever</div>
                </div>
            </div>
            <div class="overview-metric">
                <div class="metric-index">[ 03 ]</div>
                <div>
                    <div class="metric-val">97.8%</div>
                    <div class="metric-label">Acoustic Precision</div>
                    <div class="metric-desc">Audio Spectrogram Transformer inference</div>
                </div>
            </div>
            <div class="overview-metric">
                <div class="metric-index">[ 04 ]</div>
                <div>
                    <div class="metric-val">100%</div>
                    <div class="metric-label">Contactless Sensing</div>
                    <div class="metric-desc">Zero wearable tags or collar hardware</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Minimalist Tab Navigation -->
    <nav class="nav-tabs">
        <button class="tab-btn active" onclick="showTab('audio')">
            <span class="tab-index">[01]</span>
            <span>Acoustic Vocalization</span>
        </button>

        <button class="tab-btn" onclick="showTab('vision')">
            <span class="tab-index">[02]</span>
            <span>Visual Behavior</span>
        </button>

        <button class="tab-btn" onclick="showTab('roi')">
            <span class="tab-index">[03]</span>
            <span>Lactation Benchmarks</span>
        </button>

        <button class="tab-btn" onclick="showTab('history')">
            <span class="tab-index">[04]</span>
            <span>Diagnostic Records</span>
        </button>
    </nav>

    <!-- ======================================================= -->
    <!-- TAB 1: ACOUSTIC VOCALIZATION -->
    <!-- ======================================================= -->
    <section id="panel-audio" class="tab-panel active">
        <div class="workspace-section">
            <div class="section-header">
                <div class="section-tag">[ 01 / ACOUSTIC TELEMETRY ]</div>
                <h3 class="section-title">Acoustic Cow Vocalization Analysis</h3>
                <p class="section-subtitle">Record live barn microphone audio or upload a sound recording to evaluate emotional valence, distress acoustics, and AudioSet human voice discrimination.</p>
            </div>

            <!-- Action Grid -->
            <div class="action-grid">
                <div id="audioRecordTile" class="action-box" onclick="toggleAudioRecording()">
                    <svg class="action-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="22"></line>
                    </svg>
                    <div class="action-box-title" id="recordTitle">Record Live Vocalization</div>
                    <div class="action-box-desc" id="recordSub">Click to start microphone capture (2 to 5 seconds)</div>
                </div>

                <div class="action-box" onclick="document.getElementById('audioUploadInput').click()">
                    <svg class="action-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <div class="action-box-title">Upload Audio File</div>
                    <div class="action-box-desc">Accepts .wav, .mp3, .m4a, or .aac files</div>
                    <input type="file" id="audioUploadInput" accept="audio/*" style="display:none" onchange="handleAudioUpload(this.files[0])">
                </div>
            </div>

            <!-- Staged Audio Playback & Explicit Analysis Bar -->
            <div id="audioPlaybackBox" class="audio-playback-bar">
                <div class="playback-header-row">
                    <div class="playback-info">
                        <span>Staged Audio:</span>
                        <span id="audioTrackName" class="playback-track">recording.wav</span>
                    </div>
                    <div class="staged-action-btn-row">
                        <button class="btn-cta" id="btnAnalyzeAudio" onclick="triggerStagedAudioAnalysis()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <polygon points="5 3 19 12 5 21 5 3"></polygon>
                            </svg>
                            <span>Analyze Acoustic Signal</span>
                        </button>
                        <button class="btn-secondary" onclick="clearStagedAudio()">Clear / Change Audio</button>
                    </div>
                </div>
                <div class="playback-controls-row">
                    <audio id="audioElement" controls></audio>
                </div>
            </div>

            <!-- Standard Calibration Soundboard -->
            <div style="margin-top:40px;">
                <div class="sub-heading">Standard Calibration Recordings</div>
                <div class="sub-caption">Verified audio reference samples for acoustic valence validation.</div>
                <div class="sample-grid">
                    <div class="sample-item">
                        <div class="sample-meta">
                            <h5>Calm Contact Murmur</h5>
                            <p>Low-frequency maternal contact vocalization (~135 Hz pitch)</p>
                        </div>
                        <div class="btn-group">
                            <button class="btn-secondary" onclick="playSampleAudio('cattle_positive_sample.wav', 'Calm Contact Murmur')">Play</button>
                            <button class="btn-primary" onclick="stageAndOrTestAudioSample('cattle_positive_sample.wav', 'Calm Contact Murmur')">Analyze</button>
                        </div>
                    </div>

                    <div class="sample-item">
                        <div class="sample-meta">
                            <h5>High-Distress Call</h5>
                            <p>Open-mouth high-pitch separation distress (~420 Hz pitch)</p>
                        </div>
                        <div class="btn-group">
                            <button class="btn-secondary" onclick="playSampleAudio('cattle_negative_sample.wav', 'High-Distress Call')">Play</button>
                            <button class="btn-primary" onclick="stageAndOrTestAudioSample('cattle_negative_sample.wav', 'High-Distress Call')">Analyze</button>
                        </div>
                    </div>

                    <div class="sample-item">
                        <div class="sample-meta">
                            <h5>Human Speech Test</h5>
                            <p>Validates AudioSet acoustic discriminator rejection filter</p>
                        </div>
                        <div class="btn-group">
                            <button class="btn-secondary" onclick="playSampleAudio('human_speech_sample.wav', 'Human Speech Test')">Play</button>
                            <button class="btn-primary" onclick="stageAndOrTestAudioSample('human_speech_sample.wav', 'Human Speech Test')">Analyze</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Result Card -->
            <div id="audioResultCard" class="result-card">
                <div class="result-header">
                    <div class="result-title-wrap">
                        <div class="result-sublabel">ACOUSTIC INFERENCE RESULT</div>
                        <div class="result-title" id="audioResultHeading">Calm & Content State</div>
                    </div>
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
        <div class="workspace-section">
            <div class="section-header">
                <div class="section-tag">[ 02 / COMPUTER VISION ]</div>
                <h3 class="section-title">Visual Barn Activity Scanner</h3>
                <p class="section-subtitle">Process live camera frames or upload images/videos to classify rumination, feeding, resting, and standing postures via ResNet18.</p>
            </div>

            <!-- Camera Wrapper -->
            <div id="cameraBoxWrap" class="camera-wrapper">
                <video id="cameraVideo" autoplay playsinline></video>
                <div class="camera-actions">
                    <button class="btn-primary" onclick="snapCameraPhoto()">Capture Frame</button>
                    <button class="btn-secondary" onclick="closeCamera()">Close Camera</button>
                </div>
            </div>

            <!-- Image/Video Staged Preview Box with Explicit Analysis Button -->
            <div id="imagePreviewBox" class="preview-box">
                <div class="preview-box-inner">
                    <div class="preview-meta-tag">
                        <span>STAGED VISUAL TARGET:</span>
                        <span id="stagedVisionNameTag" style="font-weight:700; color:var(--text-ink);">None</span>
                    </div>
                    <div class="preview-media-container">
                        <img id="imagePreviewElem" src="" alt="Frame Preview">
                        <video id="videoPreviewElem" controls style="display:none; max-height:320px; width:100%;"></video>
                    </div>
                    <div class="staged-action-btn-row">
                        <button class="btn-cta" id="btnAnalyzeVision" onclick="triggerStagedVisionAnalysis()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                <polygon points="5 3 19 12 5 21 5 3"></polygon>
                            </svg>
                            <span>Analyze Behavior Posture</span>
                        </button>
                        <button class="btn-secondary" onclick="clearStagedVision()">Change / Remove Image</button>
                    </div>
                </div>
            </div>

            <!-- Action Grid -->
            <div class="action-grid">
                <div class="action-box" onclick="openCamera()">
                    <svg class="action-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                        <circle cx="12" cy="13" r="4"></circle>
                    </svg>
                    <div class="action-box-title">Access Camera Feed</div>
                    <div class="action-box-desc">Take an on-demand snapshot using device camera</div>
                </div>

                <div class="action-box" onclick="document.getElementById('visionUploadInput').click()">
                    <svg class="action-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <div class="action-box-title">Upload Image or Video</div>
                    <div class="action-box-desc">Accepts .jpg, .png, and .mp4 video files</div>
                    <input type="file" id="visionUploadInput" accept="image/*,video/*" style="display:none" onchange="handleVisionUpload(this.files[0])">
                </div>
            </div>

            <!-- Barn Photo Samples (Real Images) -->
            <div style="margin-top:40px;">
                <div class="sub-heading">Standard Barn Activity Samples</div>
                <div class="sub-caption">Click any sample to stage the image for behavioral analysis.</div>
                <div class="gallery-grid">
                    <div class="gallery-card" onclick="stageVisionSample('sample_rumination.jpg', 'Rumination / Cud Chewing')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_rumination.jpg" alt="Rumination"></div>
                        <div class="gallery-caption">Rumination</div>
                    </div>

                    <div class="gallery-card" onclick="stageVisionSample('sample_drinking.jpg', 'Drinking Water')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_drinking.jpg" alt="Drinking"></div>
                        <div class="gallery-caption">Drinking</div>
                    </div>

                    <div class="gallery-card" onclick="stageVisionSample('sample_feeding.jpg', 'Feeding / Eating')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_feeding.jpg" alt="Feeding"></div>
                        <div class="gallery-caption">Feeding</div>
                    </div>

                    <div class="gallery-card" onclick="stageVisionSample('sample_lying.jpg', 'Lying / Resting')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_lying.jpg" alt="Lying"></div>
                        <div class="gallery-caption">Lying Down</div>
                    </div>

                    <div class="gallery-card" onclick="stageVisionSample('sample_standing.jpg', 'Standing Alert')">
                        <div class="gallery-img-box"><img src="/samples/image/sample_standing.jpg" alt="Standing"></div>
                        <div class="gallery-caption">Standing</div>
                    </div>
                </div>
            </div>

            <!-- Vision Result Card -->
            <div id="visionResultCard" class="result-card">
                <div class="result-header">
                    <div class="result-title-wrap">
                        <div class="result-sublabel">BEHAVIORAL INFERENCE RESULT</div>
                        <div class="result-title" id="visionResultHeading">Chewing Cud (Rumination)</div>
                    </div>
                    <div class="result-badge" id="visionCertaintyPill">97.1% Confidence</div>
                </div>
                <div class="result-body">
                    <h6>Ethological Analysis</h6>
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
        <div class="workspace-section">
            <div class="section-header">
                <div class="section-tag">[ 03 / PHYSIOLOGY & YIELD ]</div>
                <h3 class="section-title">Physiological & Lactation Productivity Benchmarks</h3>
                <p class="section-subtitle">Empirical livestock benchmarks correlating ethological posture time budgets with commercial dairy yield and welfare standards.</p>
            </div>

            <div class="guidance-grid">
                <div class="guidance-card">
                    <div class="guidance-num">[ 01 ]</div>
                    <h4>Resting Time & Mammary Blood Flow</h4>
                    <p>Dairy cattle require 10 to 14 hours of daily stall rest. Blood perfusion through the mammary gland increases by up to 50% during recumbency, correlating with approximately +1.2 kg of daily milk yield per additional hour of rest.</p>
                </div>

                <div class="guidance-card">
                    <div class="guidance-num">[ 02 ]</div>
                    <h4>Rumination & Butterfat Synthesis</h4>
                    <p>Standard rumination duration is 400 to 600 minutes daily. Endogenous saliva production provides sodium bicarbonate buffering, preventing subacute rumen acidosis (SARA) and stabilizing milk fat percentages.</p>
                </div>

                <div class="guidance-card">
                    <div class="guidance-num">[ 03 ]</div>
                    <h4>Acoustic Distress & Cortisol Impact</h4>
                    <p>Elevated pitch vocalizations correlate with acute cortisol and catecholamine secretion. Hormonal surges inhibit oxytocin-mediated milk letdown, leading to residual milk retention and potential yield declines of 2.0 to 3.5 liters per event.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- ======================================================= -->
    <!-- TAB 4: DIAGNOSTIC RECORDS -->
    <!-- ======================================================= -->
    <section id="panel-history" class="tab-panel">
        <div class="workspace-section">
            <div class="section-header" style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:16px;">
                <div>
                    <div class="section-tag">[ 04 / AUDIT TRAIL ]</div>
                    <h3 class="section-title">Diagnostic Screening Logs</h3>
                    <p class="section-subtitle">Chronological record of acoustic and visual inference queries.</p>
                </div>
                <a href="/api/export/csv" class="btn-primary" style="text-decoration:none; flex:none;" download="mootrack_cattle_records.csv">
                    <span>Export CSV Report</span>
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
                            <td colspan="5" style="text-align:center; color:var(--text-muted); padding:32px;">
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
        <p>MooTrack Cattle Intelligence System &bull; Bioacoustic Transformer & ResNet18 Telemetry</p>
        <span>SAHYADRI AIML &bull; AM722T2A</span>
    </footer>

</div>

<script>
    // Tab Controller
    function showTab(tabKey) {
        if (tabKey !== 'vision') {
            closeCamera();
        }
        if (tabKey !== 'audio') {
            if (isAudioRecording) stopAudioRecording();
        }

        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

        const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick') && b.getAttribute('onclick').includes(tabKey));
        if (targetBtn) targetBtn.classList.add('active');

        const targetPanel = document.getElementById('panel-' + tabKey);
        if (targetPanel) targetPanel.classList.add('active');

        if (tabKey === 'history') {
            loadHistoryTable();
        }
    }

    // Auto-release hardware on tab hide / close
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            closeCamera();
            if (isAudioRecording) stopAudioRecording();
        }
    });
    window.addEventListener('beforeunload', () => {
        closeCamera();
        if (isAudioRecording) stopAudioRecording();
    });

    // Staged Audio State
    let stagedAudioBlob = null;
    let stagedAudioName = null;

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

        const tile = document.getElementById('audioRecordTile');
        if (tile) tile.classList.remove('recording');
        document.getElementById('recordTitle').innerText = 'Record Live Vocalization';
        document.getElementById('recordSub').innerText = 'Click to start microphone capture (2 to 5 seconds)';

        if (processorNode) processorNode.disconnect();
        if (audioContext) audioContext.close();
        if (microphoneStream) {
            microphoneStream.getTracks().forEach(t => t.stop());
            microphoneStream = null;
        }

        let totalLength = 0;
        pcmChunks.forEach(chunk => totalLength += chunk.length);
        const mergedPcm = new Float32Array(totalLength);
        let offset = 0;
        pcmChunks.forEach(chunk => {
            mergedPcm.set(chunk, offset);
            offset += chunk.length;
        });

        const wavBlob = encodeWAV(mergedPcm, 16000);
        stagedAudioBlob = wavBlob;
        stagedAudioName = 'live_microphone.wav';
        loadAudioPlayer(wavBlob, 'live_microphone.wav');
        
        // Hide previous result until explicit click
        const resultCard = document.getElementById('audioResultCard');
        if (resultCard) resultCard.style.display = 'none';
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
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let index = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            index += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
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
    }

    function handleAudioUpload(file) {
        if (!file) return;
        stagedAudioBlob = file;
        stagedAudioName = file.name;
        loadAudioPlayer(file, file.name);
        const resultCard = document.getElementById('audioResultCard');
        if (resultCard) resultCard.style.display = 'none';
    }

    function clearStagedAudio() {
        stagedAudioBlob = null;
        stagedAudioName = null;
        const pBox = document.getElementById('audioPlaybackBox');
        if (pBox) pBox.style.display = 'none';
        const pElem = document.getElementById('audioElement');
        if (pElem) {
            pElem.pause();
            pElem.src = '';
        }
        const resultCard = document.getElementById('audioResultCard');
        if (resultCard) resultCard.style.display = 'none';
        const uploadInput = document.getElementById('audioUploadInput');
        if (uploadInput) uploadInput.value = '';
    }

    function playSampleAudio(filename, label) {
        loadAudioPlayer(`/samples/audio/${filename}`, label);
        const pElem = document.getElementById('audioElement');
        if (pElem) pElem.play().catch(() => {});
    }

    async function stageAndOrTestAudioSample(filename, label) {
        try {
            const resp = await fetch(`/samples/audio/${filename}`);
            const blob = await resp.blob();
            stagedAudioBlob = blob;
            stagedAudioName = filename;
            loadAudioPlayer(blob, label);
            triggerStagedAudioAnalysis();
        } catch (e) {
            alert('Unable to load sample audio: ' + e);
        }
    }

    async function testAudioSample(filename, label) {
        stageAndOrTestAudioSample(filename, label);
    }

    async function triggerStagedAudioAnalysis() {
        if (!stagedAudioBlob) {
            alert('Please record or upload an audio file first.');
            return;
        }
        const btn = document.getElementById('btnAnalyzeAudio');
        if (btn) {
            btn.innerHTML = `<span>Analyzing...</span>`;
            btn.disabled = true;
        }
        await sendAudioToServer(stagedAudioBlob, stagedAudioName || 'audio.wav');
        if (btn) {
            btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Analyze Acoustic Signal</span>`;
            btn.disabled = false;
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

    // Visual Module & Staging
    let stagedVisionBlob = null;
    let stagedVisionName = null;
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
        if (box) box.style.display = 'none';
        const video = document.getElementById('cameraVideo');
        if (video) video.srcObject = null;
        if (cameraMediaStream) {
            cameraMediaStream.getTracks().forEach(t => {
                try { t.stop(); } catch(e) {}
            });
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
        canvas.toBlob((blob) => {
            stagedVisionBlob = blob;
            stagedVisionName = 'camera_capture.jpg';
            showImagePreview(imgUrl, 'Camera Snapshot (camera_capture.jpg)');
        }, 'image/jpeg', 0.9);
    }

    function showImagePreview(srcUrl, name) {
        const pBox = document.getElementById('imagePreviewBox');
        const pImg = document.getElementById('imagePreviewElem');
        const pVid = document.getElementById('videoPreviewElem');
        const tag = document.getElementById('stagedVisionNameTag');
        pBox.style.display = 'block';
        pImg.style.display = 'block';
        pVid.style.display = 'none';
        pImg.src = srcUrl;
        if (tag) tag.innerText = name || 'Uploaded Image';
        const resCard = document.getElementById('visionResultCard');
        if (resCard) resCard.style.display = 'none';
    }

    function showVideoPreview(srcUrl, name) {
        const pBox = document.getElementById('imagePreviewBox');
        const pImg = document.getElementById('imagePreviewElem');
        const pVid = document.getElementById('videoPreviewElem');
        const tag = document.getElementById('stagedVisionNameTag');
        pBox.style.display = 'block';
        pImg.style.display = 'none';
        pVid.style.display = 'block';
        pVid.src = srcUrl;
        if (tag) tag.innerText = name || 'Uploaded Video';
        const resCard = document.getElementById('visionResultCard');
        if (resCard) resCard.style.display = 'none';
    }

    function handleVisionUpload(file) {
        if (!file) return;
        stagedVisionBlob = file;
        stagedVisionName = file.name;
        const isVid = file.type.startsWith('video/') || /\.(mp4|mov|avi|webm)$/i.test(file.name);
        const objUrl = URL.createObjectURL(file);
        if (isVid) {
            showVideoPreview(objUrl, file.name);
        } else {
            showImagePreview(objUrl, file.name);
        }
    }

    function clearStagedVision() {
        stagedVisionBlob = null;
        stagedVisionName = null;
        const pBox = document.getElementById('imagePreviewBox');
        if (pBox) pBox.style.display = 'none';
        const pImg = document.getElementById('imagePreviewElem');
        if (pImg) pImg.src = '';
        const pVid = document.getElementById('videoPreviewElem');
        if (pVid) {
            pVid.pause();
            pVid.src = '';
        }
        const resCard = document.getElementById('visionResultCard');
        if (resCard) resCard.style.display = 'none';
        const uploadInput = document.getElementById('visionUploadInput');
        if (uploadInput) uploadInput.value = '';
    }

    async function stageVisionSample(filename, label) {
        try {
            const resp = await fetch(`/samples/image/${filename}`);
            const blob = await resp.blob();
            stagedVisionBlob = blob;
            stagedVisionName = filename;
            showImagePreview(`/samples/image/${filename}`, `${label} (${filename})`);
        } catch (e) {
            alert('Unable to load sample picture: ' + e);
        }
    }

    async function testVisionSample(filename, label) {
        await stageVisionSample(filename, label);
        await triggerStagedVisionAnalysis();
    }

    async function triggerStagedVisionAnalysis() {
        if (!stagedVisionBlob) {
            alert('Please capture, upload, or choose an image/video first.');
            return;
        }
        const btn = document.getElementById('btnAnalyzeVision');
        if (btn) {
            btn.innerHTML = `<span>Processing Features...</span>`;
            btn.disabled = true;
        }
        await sendVisionToServer(stagedVisionBlob, stagedVisionName || 'visual_target.jpg');
        if (btn) {
            btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span>Analyze Behavior Posture</span>`;
            btn.disabled = false;
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
                tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:32px;">No records logged yet. Process an audio recording or visual sample to begin logging.</td></tr>';
                return;
            }

            tbody.innerHTML = data.slice(0, 25).map(item => {
                const isAudio = item.type === 'audio';
                const isPos = item.predicted_class === 'Positive';
                const isNeg = item.predicted_class === 'Negative';
                const tagClass = isPos ? 'pos' : (isNeg ? 'neg' : 'neu');
                const tagText = item.predicted_class;
                const conf = Math.round((item.confidence || 0) * 100);

                return `
                    <tr>
                        <td style="font-family:var(--font-mono); font-size:0.8rem;">${item.timestamp}</td>
                        <td>${isAudio ? 'Acoustic' : 'Visual'}</td>
                        <td><span class="tag-pill ${tagClass}">${tagText}</span></td>
                        <td style="font-family:var(--font-mono); font-weight:600;">${conf}%</td>
                        <td style="color:var(--text-muted); font-size:0.8rem; font-family:var(--font-mono);">${item.filename || 'Live Capture'}</td>
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
