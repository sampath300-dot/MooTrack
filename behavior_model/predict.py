"""
MooTrack — Cattle Behaviour Recognition & Video Prediction Engine
Supports fine-tuned ResNet18 inference on single cow images/crops
as well as temporal video feeds (frame extraction, timeline breakdown, and time-budget aggregation).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models

from behavior_model.config import (
    ALT_MODEL_PATH,
    BEHAVIOR_CLASSES,
    BEHAVIOR_DESCRIPTIONS,
    CBVD_TO_STANDARD,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    PREDICTION_THRESHOLD,
    SAVED_MODEL_DIR,
    SAVED_MODEL_PATH,
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_MAX_FRAMES,
    VIDEO_SAMPLE_FPS,
)
from behavior_model.preprocessing import load_and_preprocess_image
from behavior_model.video_processor import (
    aggregate_video_predictions,
    extract_video_frames,
    get_video_metadata,
    is_video_file,
)

_BEHAVIOR_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None


def find_model_checkpoint() -> Optional[Path]:
    """
    Finds available model checkpoint by checking standard paths and directory contents.
    """
    candidates = [
        SAVED_MODEL_PATH,
        ALT_MODEL_PATH,
        SAVED_MODEL_DIR / "best_model.pth",
        SAVED_MODEL_DIR / "best_resnet18.pth",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    # Search saved model directory for any .pth / .pt file
    if SAVED_MODEL_DIR.exists():
        for pth in SAVED_MODEL_DIR.glob("*.pth"):
            if pth.is_file():
                return pth
        for pth in SAVED_MODEL_DIR.glob("*.pt"):
            if pth.is_file():
                return pth

    return None


def get_behavior_model() -> Tuple[nn.Module, torch.device]:
    """
    Lazy-loads and caches the fine-tuned ResNet18 cattle behaviour model on device.
    """
    global _BEHAVIOR_MODEL, _DEVICE

    if _BEHAVIOR_MODEL is not None and _DEVICE is not None:
        return _BEHAVIOR_MODEL, _DEVICE

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize ResNet18 architecture
    try:
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        try:
            model = models.resnet18(weights=None)
        except Exception:
            model = models.resnet18(pretrained=False)

    # 2. Replace final fully connected classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # 3. Load trained weights if checkpoint exists
    ckpt_path = find_model_checkpoint()
    if ckpt_path is not None and ckpt_path.exists():
        try:
            checkpoint = torch.load(ckpt_path, map_location=_DEVICE)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"[Warning] Could not load checkpoint from {ckpt_path}: {e}")
    else:
        print(f"[Notice] Checkpoint not found at {SAVED_MODEL_PATH}. Initialized with base weights.")

    model.to(_DEVICE)
    model.eval()
    _BEHAVIOR_MODEL = model

    return _BEHAVIOR_MODEL, _DEVICE


def load_model() -> nn.Module:
    """
    Alias for get_behavior_model() returning the PyTorch module for backwards compatibility.
    """
    model, _ = get_behavior_model()
    return model


def predict_behavior_image(
    image_input: Union[str, Path, Image.Image, np.ndarray]
) -> Dict[str, Any]:
    """
    Predicts cattle behavior class from an image, photo, or single keyframe.

    Pipeline:
    Image Input -> ResNet18 Preprocessing (Resize/Crop/Normalize) -> Forward Pass -> Softmax/Sigmoid -> Behavior Dict
    """
    model, device = get_behavior_model()

    input_tensor = load_and_preprocess_image(image_input).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = ID2LABEL[predicted_idx]
    confidence = float(np.round(probabilities[predicted_idx], 4))

    prob_dict = {
        ID2LABEL[i]: float(np.round(probabilities[i], 4))
        for i in range(NUM_CLASSES)
    }

    description = BEHAVIOR_DESCRIPTIONS.get(
        predicted_class,
        "Observed cattle behavior class from CBVD-5 dataset."
    )

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "description": description,
        "model": "ResNet18 (Convolutional Neural Network)",
        "device": str(device),
    }


def predict_behavior_video(
    video_path: Union[str, Path],
    sample_fps: float = VIDEO_SAMPLE_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    include_thumbnails: bool = True,
) -> Dict[str, Any]:
    """
    Processes a video file by extracting keyframes, predicting cattle behaviors per frame,
    and aggregating temporal behavior patterns across the full video duration.

    Returns:
    {
        "class": "feeding",
        "confidence": 0.88,
        "probabilities": {"standing": 0.20, "feeding": 0.70, "lying": 0.05, ...},
        "activity_breakdown": {"feeding": 70.0, "standing": 25.0, ...},
        "dominant_frequency_percent": 70.0,
        "total_frames_analyzed": 15,
        "transitions_count": 2,
        "transitions": [...],
        "timeline": [
            {"timestamp_seconds": 0.0, "class": "standing", "confidence": 0.91, "probabilities": {...}},
            {"timestamp_seconds": 1.0, "class": "feeding", "confidence": 0.88, "probabilities": {...}},
            ...
        ],
        "summary": "...",
        "video_metadata": {...}
    }
    """
    path_obj = Path(video_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {path_obj}")

    # Extract sampled frames
    extracted_frames, metadata = extract_video_frames(
        path_obj,
        sample_fps=sample_fps,
        max_frames=max_frames,
        include_thumbnails=include_thumbnails,
    )

    if not extracted_frames:
        raise ValueError(f"Could not extract any valid frames from video: {path_obj}")

    model, device = get_behavior_model()
    frame_results: List[Dict[str, Any]] = []

    for frame_data in extracted_frames:
        pil_img = frame_data["image"]
        input_tensor = load_and_preprocess_image(pil_img).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probabilities))
        pred_class = ID2LABEL[pred_idx]
        conf = float(np.round(probabilities[pred_idx], 4))

        prob_dict = {
            ID2LABEL[i]: float(np.round(probabilities[i], 4))
            for i in range(NUM_CLASSES)
        }

        entry = {
            "frame_index": frame_data["frame_index"],
            "timestamp_seconds": frame_data["timestamp_seconds"],
            "timestamp_formatted": frame_data["timestamp_formatted"],
            "class": pred_class,
            "confidence": conf,
            "probabilities": prob_dict,
        }
        if "thumbnail_base64" in frame_data:
            entry["thumbnail_base64"] = frame_data["thumbnail_base64"]

        frame_results.append(entry)

    # Aggregate temporal results
    aggregated = aggregate_video_predictions(frame_results, metadata)
    aggregated["device"] = str(device)
    return aggregated


def predict_behavior(
    input_source: Union[str, Path, Image.Image, np.ndarray],
    sample_fps: float = VIDEO_SAMPLE_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
) -> Dict[str, Any]:
    """
    Unified predictor for MooTrack cattle behavior recognition.
    Automatically detects whether input is a video file or an image,
    and routes to the appropriate vision pipeline.
    """
    if isinstance(input_source, (str, Path)):
        p = Path(input_source)
        if is_video_file(p):
            return predict_behavior_video(p, sample_fps=sample_fps, max_frames=max_frames)
        else:
            return predict_behavior_image(p)
    else:
        return predict_behavior_image(input_source)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        print("=" * 65)
        print("MOOTRACK CATTLE BEHAVIOUR RECOGNITION")
        print("=" * 65)
        print(f"Target Input: {target_path}")

        res = predict_behavior(target_path)
        is_video = "timeline" in res

        if is_video:
            meta = res.get("video_metadata", {})
            print(f"\n[VIDEO ANALYSIS RESULTS]")
            print(f"File: {meta.get('filename')} | Duration: {meta.get('duration_seconds')}s | FPS: {meta.get('fps')}")
            print(f"Sampled Frames: {res.get('total_frames_analyzed')} | Transitions: {res.get('transitions_count')}")
            print(f"\nDominant Behaviour : {res['class'].upper()}")
            print(f"Confidence Level   : {res['confidence'] * 100:.1f}%")
            print(f"Observed Time-Budget:")
            for b_name, b_pct in res.get("activity_breakdown", {}).items():
                bar = "#" * int(b_pct // 5)
                print(f"  - {b_name.capitalize():<12}: {b_pct:>5.1f}%  |{bar:<20}|")
            print(f"\nTimeline Preview (First 5 frames):")
            for t_item in res.get("timeline", [])[:5]:
                print(f"  [{t_item['timestamp_formatted']}] {t_item['class'].capitalize()} (Conf: {t_item['confidence']*100:.1f}%)")
            print(f"\nEthological Summary:\n{res.get('summary')}")
        else:
            print(f"\n[IMAGE ANALYSIS RESULTS]")
            print(f"Predicted Behaviour: {res['class'].upper()}")
            print(f"Confidence Level   : {res['confidence'] * 100:.1f}%")
            print(f"Class Probabilities:")
            for b_name, b_prob in res.get("probabilities", {}).items():
                bar = "#" * int(b_prob * 20)
                print(f"  - {b_name.capitalize():<12}: {b_prob*100:>5.1f}%  |{bar:<20}|")
            print(f"\nEthological Description:\n{res.get('description')}")

        print("=" * 65)
    else:
        print("Usage: python -m behavior_model.predict <path_to_image_or_video>")
