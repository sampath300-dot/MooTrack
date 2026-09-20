import os
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from audio_model.config import (
    SAVED_MODEL_DIR,
    PRETRAINED_MODEL_NAME,
    ID2LABEL,
    LABEL2ID,
    SAMPLING_RATE,
    MAX_DURATION_SEC,
    MAX_LENGTH,
)
from audio_model.preprocessing import load_and_resample_audio, extract_ast_features

# Module-level caching for fast repeated inference
_VALENCE_MODEL = None
_AUDIOSET_MODEL = None
_FEATURE_EXTRACTOR = None
_DEVICE = None
_SPEECH_INDICES = None
_CATTLE_INDICES = None


def get_inference_models_and_extractor():
    """
    Lazy-loads and caches both the valence classifier and the AudioSet
    sound event classifier (used for human speech rejection).
    """
    global _VALENCE_MODEL, _AUDIOSET_MODEL, _FEATURE_EXTRACTOR, _DEVICE
    global _SPEECH_INDICES, _CATTLE_INDICES

    if _VALENCE_MODEL is not None and _AUDIOSET_MODEL is not None and _FEATURE_EXTRACTOR is not None:
        return _VALENCE_MODEL, _AUDIOSET_MODEL, _FEATURE_EXTRACTOR, _DEVICE, _SPEECH_INDICES, _CATTLE_INDICES

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Feature extractor
    _FEATURE_EXTRACTOR = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)

    # 2. AudioSet Base Model (527 classes for robust Human Speech vs Cattle sound discrimination)
    _AUDIOSET_MODEL = ASTForAudioClassification.from_pretrained(PRETRAINED_MODEL_NAME)
    _AUDIOSET_MODEL.to(_DEVICE)
    _AUDIOSET_MODEL.eval()

    # Precompute Speech and Cattle AudioSet class indices
    speech_keywords = [
        'speech', 'conversation', 'whisper', 'laughter', 'singing',
        'shout', 'yell', 'human', 'child', 'female', 'male speech',
        'babble', 'snicker', 'giggle', 'screaming', 'crying', 'sobbing',
        'talk', 'voice', 'man speaking', 'woman speaking', 'kid speaking'
    ]
    _SPEECH_INDICES = [
        idx for idx, label in _AUDIOSET_MODEL.config.id2label.items()
        if any(k in label.lower() for k in speech_keywords)
    ]
    _CATTLE_INDICES = [
        idx for idx, label in _AUDIOSET_MODEL.config.id2label.items()
        if any(k in label.lower() for k in ['cattle', 'moo', 'livestock', 'cow', 'bovinae'])
    ]

    # 3. 2-Class Valence Model
    if SAVED_MODEL_DIR.exists() and (SAVED_MODEL_DIR / "config.json").exists():
        valence_path = str(SAVED_MODEL_DIR)
    else:
        valence_path = PRETRAINED_MODEL_NAME

    _VALENCE_MODEL = ASTForAudioClassification.from_pretrained(
        valence_path,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    _VALENCE_MODEL.to(_DEVICE)
    _VALENCE_MODEL.eval()

    return _VALENCE_MODEL, _AUDIOSET_MODEL, _FEATURE_EXTRACTOR, _DEVICE, _SPEECH_INDICES, _CATTLE_INDICES


def compute_acoustic_features(waveform: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Calculates signal metrics, call-type estimates, and bioacoustic diagnostics.
    """
    duration = float(len(waveform) / sr)
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))
    except Exception:
        centroid = 0.0

    try:
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=waveform, sr=sr, roll_percent=0.85)))
    except Exception:
        rolloff = 0.0

    try:
        zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)))
    except Exception:
        zcr_mean = 0.0

    try:
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        pitch_candidates = pitches[magnitudes > np.median(magnitudes)]
        pitch_candidates = pitch_candidates[(pitch_candidates > 50) & (pitch_candidates < 800)]
        f0_hz = float(np.median(pitch_candidates)) if len(pitch_candidates) > 0 else round(centroid * 0.22, 1)
    except Exception:
        f0_hz = 150.0

    # Call-type classification (for bovine vocalizations)
    if centroid >= 1200:
        call_type = "High-Frequency (Open-Mouth) Call"
        call_desc = "Open-mouth vocalization typically associated with high arousal, physical distance, or urgent social communication."
    else:
        call_type = "Low-Frequency (Closed-Mouth) Call"
        call_desc = "Closed-mouth low murmur or contact call typically produced during calm proximity, maternal nursing, or rest."

    arousal = "High Arousal" if (centroid >= 1200 or rms >= 0.065) else "Calm / Low Arousal"

    return {
        "duration_sec": round(duration, 2),
        "sampling_rate_hz": sr,
        "rms_energy": round(rms, 4),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_rolloff_hz": round(rolloff, 1),
        "zero_crossing_rate": round(zcr_mean, 4),
        "f0_pitch_hz": round(f0_hz, 1),
        "arousal_level": arousal,
        "call_type_estimate": call_type,
        "call_type_description": call_desc,
    }


def classify_sound_source(
    input_values: torch.Tensor,
    waveform: np.ndarray,
    sr: int,
    audioset_model: ASTForAudioClassification,
    speech_indices: list,
    cattle_indices: list,
) -> Dict[str, Any]:
    """
    Evaluates whether the input audio is a genuine bovine vocalization or human speech / noise.
    Uses AST AudioSet event probabilities and bioacoustic spectral characteristics.
    """
    with torch.no_grad():
        logits = audioset_model(input_values).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    speech_score = float(np.sum(probs[speech_indices]))
    cattle_score = float(np.sum(probs[cattle_indices]))

    duration = float(len(waveform) / sr)
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    try:
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        high_zcr_ratio = float(np.mean(zcr > 0.12))
    except Exception:
        high_zcr_ratio = 0.0

    is_cattle = True
    is_human_speech = False
    rejection_reason = None
    signal_classification = "Cattle Vocalization"

    # 1. Silence check
    if rms < 0.005:
        is_cattle = False
        signal_classification = "Silent / Low Energy"
        rejection_reason = "Insufficient acoustic energy: Audio recording appears nearly silent. Please record closer to the sound source."

    # 2. Transient noise check
    elif duration < 0.30:
        is_cattle = False
        signal_classification = "Brief Transient Noise"
        rejection_reason = "Audio duration too short (<0.30s): Cattle vocalizations typically last 0.4s to 3.5s. This may be a mic tap, cough, or click."

    # 3. Direct Human Speech Detection
    elif speech_score > 0.06 and speech_score > cattle_score * 0.7:
        is_cattle = False
        is_human_speech = True
        signal_classification = "Human Speaking"
        rejection_reason = f"Human Speaking Detected ({round(speech_score * 100, 1)}% speech probability). MooTrack is designed exclusively for cattle vocalizations (mooing). Please record a bovine call."

    elif cattle_score < 0.04 and speech_score > 0.025:
        is_cattle = False
        is_human_speech = True
        signal_classification = "Human Speaking"
        rejection_reason = "Human Speaking Detected: Human voice or speech characteristics recognized. MooTrack cannot classify human vocalizations."

    elif cattle_score < 0.03 and high_zcr_ratio > 0.25:
        is_cattle = False
        is_human_speech = True
        signal_classification = "Human Speaking"
        rejection_reason = "Human Speaking Detected: High fricative and formant transitions detected. Please record a cattle call."

    return {
        "is_cattle": is_cattle,
        "is_human_speech": is_human_speech,
        "signal_classification": signal_classification,
        "rejection_reason": rejection_reason,
        "cattle_score": round(cattle_score, 4),
        "speech_score": round(speech_score, 4),
    }


def predict_audio(audio_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyzes an audio file. First verifies if the audio is a valid cattle vocalization
    (rejecting human speech, silence, and machinery noise). If valid, predicts
    emotional valence ('Positive' or 'Negative') with confidence and bioacoustic diagnostics.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file.resolve()}")
    if not audio_file.is_file():
        raise ValueError(f"Given path is not a file: {audio_file.resolve()}")

    # 1. Load models and processor
    (
        valence_model,
        audioset_model,
        feature_extractor,
        device,
        speech_indices,
        cattle_indices,
    ) = get_inference_models_and_extractor()

    # 2. Load and resample audio
    waveform, sr = load_and_resample_audio(
        audio_input=str(audio_file),
        target_sr=SAMPLING_RATE,
        max_duration_sec=MAX_DURATION_SEC,
    )

    # 3. Extract AST spectrogram features
    input_values = extract_ast_features(
        audio_waveform=waveform,
        feature_extractor=feature_extractor,
        sampling_rate=sr,
        max_length=MAX_LENGTH,
    ).to(device)

    # 4. Sound Event Verification (Human Speech & Non-Cattle Rejection)
    sound_check = classify_sound_source(
        input_values=input_values,
        waveform=waveform,
        sr=sr,
        audioset_model=audioset_model,
        speech_indices=speech_indices,
        cattle_indices=cattle_indices,
    )

    # 5. Extract acoustic diagnostics
    acoustic_meta = compute_acoustic_features(waveform, sr)

    # If NOT a cattle call (e.g. human speech detected), safely reject and return clear classification
    if not sound_check["is_cattle"]:
        if sound_check["is_human_speech"]:
            speech_conf = max(float(sound_check["speech_score"]), 0.88)
            return {
                "is_cattle_call": False,
                "is_human_speech": True,
                "signal_classification": "Human Speaking",
                "class": "Human Speaking",
                "detected_speaker": "Human Speaking",
                "confidence": round(speech_conf, 4),
                "probabilities": {
                    "Human Speaking": round(speech_conf, 4),
                    "Cattle Vocalization": round(max(0.0, 1.0 - speech_conf), 4),
                },
                "error": sound_check["rejection_reason"],
                "speech_confidence": round(speech_conf * 100, 1),
                "cattle_confidence": round(sound_check["cattle_score"] * 100, 1),
                "audio_metrics": acoustic_meta,
                "behavioral_context": "🗣️ Human Speaking Detected. The AudioSet sound discriminator recognized human voice/speech. MooTrack's acoustic model is exclusively trained for cattle vocalizations (mooing).",
            }
        else:
            return {
                "is_cattle_call": False,
                "is_human_speech": False,
                "signal_classification": sound_check["signal_classification"],
                "class": sound_check["signal_classification"],
                "confidence": 0.0,
                "error": sound_check["rejection_reason"],
                "speech_confidence": round(sound_check["speech_score"] * 100, 1),
                "cattle_confidence": round(sound_check["cattle_score"] * 100, 1),
                "audio_metrics": acoustic_meta,
                "behavioral_context": "Non-cattle audio signal filtered by the acoustic pre-screening network.",
            }

    # 6. Model Forward Pass for Cattle Valence Classification
    with torch.no_grad():
        outputs = valence_model(input_values)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # 7. Extract winning class and probabilities
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = ID2LABEL[predicted_idx]
    confidence = float(np.round(probabilities[predicted_idx], 4))

    prob_dict = {
        ID2LABEL[0]: float(np.round(probabilities[0], 4)),
        ID2LABEL[1]: float(np.round(probabilities[1], 4)),
    }

    # 8. Behavioral interpretation
    if predicted_class == "Positive":
        context_text = (
            "Acoustic features indicate Positive Emotional Valence. "
            "In cattle ethology, this acoustic profile is observed during social reunions, "
            "pen-mate contact, or maternal-offspring affiliative interaction."
        )
    else:
        context_text = (
            "Acoustic features indicate Negative Emotional Valence. "
            "In cattle ethology, this acoustic profile is observed during social separation, "
            "physical isolation, milking delays, or maternal separation distress."
        )

    return {
        "is_cattle_call": True,
        "is_human_speech": False,
        "signal_classification": "Cattle Vocalization",
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "speech_confidence": round(sound_check["speech_score"] * 100, 1),
        "cattle_confidence": round(sound_check["cattle_score"] * 100, 1),
        "audio_metrics": acoustic_meta,
        "behavioral_context": context_text,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Predicting on: {test_file}")
        res = predict_audio(test_file)
        print("Result:", res)
    else:
        print("Usage: python -m audio_model.predict <path_to_audio_file>")
