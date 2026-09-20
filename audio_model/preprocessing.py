import io
import os
from pathlib import Path
from typing import Union, Dict, Any, Tuple

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import AutoFeatureExtractor


DEFAULT_SAMPLING_RATE = 16000
DEFAULT_MAX_DURATION_SEC = 10.0


def load_and_resample_audio(
    audio_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Dict[str, Any]],
    target_sr: int = DEFAULT_SAMPLING_RATE,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio input from a file path, audio bytes, dictionary, or numpy array,
    convert to mono, and resample to the target sampling rate.

    Supports:
      - File path (str or Path)
      - Raw audio bytes (bytes or BytesIO)
      - Hugging Face audio dictionary: {'bytes': b'...'} or {'array': ..., 'sampling_rate': ...}
      - NumPy 1D/2D array
    """
    if audio_input is None:
        raise ValueError("Audio input cannot be None.")

    # 1. Handle Hugging Face Audio Dictionary format
    if isinstance(audio_input, dict):
        if "bytes" in audio_input and audio_input["bytes"] is not None:
            bio = io.BytesIO(audio_input["bytes"])
            waveform, orig_sr = sf.read(bio, dtype="float32")
        elif "array" in audio_input and "sampling_rate" in audio_input and audio_input["array"] is not None:
            waveform = np.asarray(audio_input["array"], dtype=np.float32)
            orig_sr = audio_input["sampling_rate"]
        elif "path" in audio_input and audio_input["path"] and Path(audio_input["path"]).exists():
            waveform, orig_sr = sf.read(str(audio_input["path"]), dtype="float32")
        else:
            raise ValueError(
                f"Audio dictionary must contain non-null 'bytes' or ('array' and 'sampling_rate'). "
                f"Got keys: {list(audio_input.keys())}"
            )

    # 2. Handle raw bytes / BytesIO (e.g., from web app upload)
    elif isinstance(audio_input, (bytes, io.BytesIO)):
        bio = io.BytesIO(audio_input) if isinstance(audio_input, bytes) else audio_input
        try:
            waveform, orig_sr = sf.read(bio, dtype="float32")
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio bytes: {e}")

    # 3. Handle direct NumPy array
    elif isinstance(audio_input, np.ndarray):
        waveform = audio_input.astype(np.float32)
        orig_sr = target_sr

    # 4. Handle File Path
    elif isinstance(audio_input, (str, Path)):
        audio_path = Path(audio_input)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path.resolve()}")
        if not audio_path.is_file():
            raise ValueError(f"Specified path is not a file: {audio_path.resolve()}")

        try:
            waveform, orig_sr = sf.read(str(audio_path), dtype="float32")
        except Exception as e_sf:
            try:
                waveform, orig_sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
            except Exception as e_lib:
                # If both fail, try reading raw bytes with BytesIO
                try:
                    with open(str(audio_path), "rb") as af:
                        raw_bytes = af.read()
                    waveform, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                except Exception as e_raw:
                    raise RuntimeError(
                        f"Failed to read audio file '{audio_path.name}'. "
                        f"Error details: Soundfile: {e_sf}; Librosa: {e_lib}; Raw: {e_raw}"
                    )
    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio_input)}. "
            f"Expected filepath string, Path, bytes, numpy array, or audio dict."
        )

    # Validate non-empty
    if waveform.size == 0:
        raise ValueError("Loaded audio waveform is empty (0 samples).")

    # 5. Convert multi-channel (stereo) to Mono
    if waveform.ndim > 1:
        if waveform.shape[0] < waveform.shape[1] and waveform.shape[0] in (2, 3, 4, 6):
            waveform = np.mean(waveform, axis=0)
        else:
            waveform = np.mean(waveform, axis=-1)

    # 6. Resample if sampling rate differs from target_sr
    if orig_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)

    # 7. Normalize amplitude if peak > 1.0
    max_val = np.max(np.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val

    # 8. Truncate if exceeds max_duration_sec
    if max_duration_sec is not None and max_duration_sec > 0:
        max_samples = int(max_duration_sec * target_sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

    return waveform.astype(np.float32), target_sr


def extract_ast_features(
    audio_waveform: np.ndarray,
    feature_extractor: AutoFeatureExtractor,
    sampling_rate: int = DEFAULT_SAMPLING_RATE,
    max_length: int = 1024,
) -> torch.Tensor:
    """
    Convert a 1D audio waveform into AST-compatible spectrogram features.
    """
    if audio_waveform.ndim != 1:
        raise ValueError(f"Expected 1D audio waveform array, got shape {audio_waveform.shape}")

    inputs = feature_extractor(
        audio_waveform,
        sampling_rate=sampling_rate,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    return inputs["input_values"]


def preprocess_audio_pipeline(
    audio_input: Union[str, Path, bytes, io.BytesIO, np.ndarray, Dict[str, Any]],
    feature_extractor: AutoFeatureExtractor,
    target_sr: int = DEFAULT_SAMPLING_RATE,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
) -> torch.Tensor:
    """
    End-to-end preprocessing function used by both training and inference.
    """
    waveform, sr = load_and_resample_audio(
        audio_input=audio_input,
        target_sr=target_sr,
        max_duration_sec=max_duration_sec,
    )
    input_values = extract_ast_features(
        audio_waveform=waveform,
        feature_extractor=feature_extractor,
        sampling_rate=sr,
    )
    return input_values
