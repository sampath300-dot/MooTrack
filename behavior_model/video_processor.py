"""
MooTrack — Cattle Video Processing & Frame Extraction Engine
Extracts keyframes at specified sampling intervals, computes temporal timelines,
and aggregates cattle behavior predictions over time.
"""

import base64
import io
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from behavior_model.config import (
    BEHAVIOR_CLASSES,
    BEHAVIOR_DESCRIPTIONS,
    KEYFRAME_THUMBNAIL_WIDTH,
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_MAX_FRAMES,
    VIDEO_SAMPLE_FPS,
)


def is_video_file(file_path: Union[str, Path]) -> bool:
    """Checks if the given file path has a supported video extension."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_VIDEO_EXTENSIONS


def get_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts structural metadata from a video file.
    """
    path_str = str(Path(video_path).resolve())
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Video file not found: {path_str}")

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file via OpenCV: {path_str}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0.0

        return {
            "video_path": path_str,
            "filename": Path(video_path).name,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "duration_seconds": round(duration_s, 2),
        }
    finally:
        cap.release()


def image_to_base64_thumbnail(
    image_np_or_pil: Union[np.ndarray, Image.Image],
    max_width: int = KEYFRAME_THUMBNAIL_WIDTH,
    quality: int = 80,
) -> str:
    """
    Converts an RGB image array or PIL Image into a compact base64 data URI JPEG thumbnail.
    """
    if isinstance(image_np_or_pil, np.ndarray):
        pil_img = Image.fromarray(image_np_or_pil)
    else:
        pil_img = image_np_or_pil

    w, h = pil_img.size
    if w > max_width:
        ratio = max_width / float(w)
        new_size = (max_width, int(h * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def extract_video_frames(
    video_path: Union[str, Path],
    sample_fps: float = VIDEO_SAMPLE_FPS,
    max_frames: int = VIDEO_MAX_FRAMES,
    include_thumbnails: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts frames at regular intervals (sample_fps) up to max_frames.
    Returns:
        frames: List of dicts with frame_index, timestamp_s, frame_rgb (PIL.Image), thumbnail_b64
        metadata: Video metadata dict
    """
    path_str = str(Path(video_path).resolve())
    metadata = get_video_metadata(path_str)

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path_str}")

    fps = metadata["fps"] or 25.0
    total_frames = metadata["total_frames"]

    # Calculate frame stride
    stride = max(1, int(round(fps / max(0.1, sample_fps))))

    # If total sampled frames would exceed max_frames, increase stride
    estimated_samples = total_frames // stride
    if estimated_samples > max_frames:
        stride = max(1, total_frames // max_frames)

    extracted_frames = []
    frame_idx = 0
    read_count = 0

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # Convert BGR (OpenCV) to RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                timestamp_s = round(frame_idx / fps, 2)

                frame_entry = {
                    "frame_index": frame_idx,
                    "timestamp_seconds": timestamp_s,
                    "timestamp_formatted": f"{int(timestamp_s // 60):02d}:{int(timestamp_s % 60):02d}.{int((timestamp_s % 1) * 10):01d}",
                    "image": pil_image,
                }

                if include_thumbnails:
                    frame_entry["thumbnail_base64"] = image_to_base64_thumbnail(pil_image)

                extracted_frames.append(frame_entry)
                read_count += 1

                if read_count >= max_frames:
                    break

            frame_idx += 1
    finally:
        cap.release()

    metadata["sampled_frames"] = len(extracted_frames)
    metadata["sample_stride"] = stride

    return extracted_frames, metadata


def aggregate_video_predictions(
    frame_results: List[Dict[str, Any]],
    video_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregates per-frame behavior predictions into an overall video assessment,
    timeline breakdown, and ethological time-budget analysis.
    """
    if not frame_results:
        return {
            "class": "standing",
            "confidence": 0.0,
            "probabilities": {c: 0.0 for c in BEHAVIOR_CLASSES},
            "activity_breakdown": {c: 0.0 for c in BEHAVIOR_CLASSES},
            "dominant_class": "unknown",
            "timeline": [],
            "transitions": [],
            "video_metadata": video_metadata,
            "summary": "No frames were available for analysis.",
        }

    total_sampled = len(frame_results)
    class_counts = {c: 0 for c in BEHAVIOR_CLASSES}
    class_prob_sums = {c: 0.0 for c in BEHAVIOR_CLASSES}
    timeline = []
    transitions = []
    last_class = None

    for idx, item in enumerate(frame_results):
        pred_cls = item.get("class", "standing")
        conf = item.get("confidence", 0.0)
        probs = item.get("probabilities", {})
        t_sec = item.get("timestamp_seconds", round(idx * 1.0, 2))
        t_fmt = item.get("timestamp_formatted", f"{int(t_sec // 60):02d}:{int(t_sec % 60):02d}")

        if pred_cls in class_counts:
            class_counts[pred_cls] += 1
        else:
            class_counts[pred_cls] = 1

        for c in BEHAVIOR_CLASSES:
            class_prob_sums[c] += float(probs.get(c, 0.0))

        timeline_entry = {
            "frame_index": item.get("frame_index", idx),
            "timestamp_seconds": t_sec,
            "timestamp_formatted": t_fmt,
            "class": pred_cls,
            "confidence": conf,
            "probabilities": probs,
        }
        if "thumbnail_base64" in item:
            timeline_entry["thumbnail_base64"] = item["thumbnail_base64"]

        timeline.append(timeline_entry)

        # Detect transitions
        if last_class is not None and pred_cls != last_class:
            transitions.append({
                "timestamp_seconds": t_sec,
                "timestamp_formatted": t_fmt,
                "from_class": last_class,
                "to_class": pred_cls,
            })
        last_class = pred_cls

    # Determine dominant behavior by frame frequency (and average probability)
    dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
    dominant_frequency_pct = round((class_counts[dominant_class] / total_sampled) * 100, 1)

    # Activity breakdown (time budget percentage)
    activity_breakdown = {
        c: round((class_counts.get(c, 0) / total_sampled) * 100, 1)
        for c in BEHAVIOR_CLASSES
    }

    # Mean probabilities across video
    mean_probabilities = {
        c: round(class_prob_sums.get(c, 0.0) / total_sampled, 4)
        for c in BEHAVIOR_CLASSES
    }

    # Mean confidence for dominant behavior
    dominant_confidences = [
        item.get("confidence", 0.0)
        for item in frame_results
        if item.get("class") == dominant_class
    ]
    avg_confidence = round(
        float(np.mean(dominant_confidences)) if dominant_confidences else mean_probabilities[dominant_class],
        4
    )

    # Create summary narrative
    desc = BEHAVIOR_DESCRIPTIONS.get(dominant_class, "Recorded cattle behavior pattern.")
    duration_str = f"{video_metadata.get('duration_seconds', 0.0)}s"
    summary_text = (
        f"Video Analysis ({duration_str}, {total_sampled} frames sampled): "
        f"Dominant behavior is '{dominant_class}' ({dominant_frequency_pct}% of observed video). "
        f"{desc}"
    )

    return {
        "class": dominant_class,
        "confidence": avg_confidence,
        "probabilities": mean_probabilities,
        "activity_breakdown": activity_breakdown,
        "dominant_frequency_percent": dominant_frequency_pct,
        "total_frames_analyzed": total_sampled,
        "transitions_count": len(transitions),
        "transitions": transitions,
        "timeline": timeline,
        "description": desc,
        "summary": summary_text,
        "video_metadata": video_metadata,
        "model": "ResNet18 (Video Temporal Frame Aggregation)",
    }


def generate_synthetic_test_video(
    output_path: Union[str, Path],
    duration_seconds: int = 5,
    fps: int = 20,
) -> Path:
    """
    Creates a valid synthetic cattle test video clip by compositing sample cattle images
    with temporal transitions. Useful for automated tests and standalone verification.
    """
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    test_samples_dir = Path(__file__).resolve().parent / "test_samples"
    sample_files = [
        test_samples_dir / "sample_standing.jpg",
        test_samples_dir / "sample_feeding.jpg",
        test_samples_dir / "sample_lying.jpg",
        test_samples_dir / "sample_rumination.jpg",
        test_samples_dir / "sample_drinking.jpg",
    ]

    loaded_images = []
    for sf in sample_files:
        if sf.exists():
            img = cv2.imread(str(sf))
            if img is not None:
                img = cv2.resize(img, (640, 480))
                loaded_images.append(img)

    if not loaded_images:
        # Fallback to color canvas
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "MooTrack Cattle Test Video", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        loaded_images = [blank]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_p), fourcc, fps, (640, 480))

    total_frames = duration_seconds * fps
    num_states = len(loaded_images)
    frames_per_state = max(1, total_frames // num_states)

    for i in range(total_frames):
        state_idx = min(i // frames_per_state, num_states - 1)
        frame = loaded_images[state_idx].copy()

        # Add timestamp & watermark
        ts_sec = i / fps
        cv2.putText(
            frame,
            f"MooTrack Live Feed - T+{ts_sec:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 230, 153),
            2,
            cv2.LINE_AA,
        )
        out.write(frame)

    out.release()
    return out_p
