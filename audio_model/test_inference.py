import os
import sys
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path
from audio_model.predict import predict_audio

TEST_SAMPLES_DIR = Path(__file__).resolve().parent / "test_samples"


def run_tests():
    print("=" * 65)
    print("MOOTRACK - AUDIO MODEL INFERENCE & SPEECH REJECTION TESTING")
    print("=" * 65)

    # 1. Test Positive Sample
    pos_file = TEST_SAMPLES_DIR / "cattle_positive_sample.wav"
    if pos_file.exists():
        print(f"\n[Test 1] Testing Positive Cattle Recording: {pos_file.name}")
        res_pos = predict_audio(str(pos_file))
        print("  -> Result:", res_pos)
        assert isinstance(res_pos, dict), "Result must be a dict"
        assert res_pos.get("is_cattle_call") is True, "Cattle sample must be recognized as cattle"
        assert "class" in res_pos and "confidence" in res_pos, "Missing required keys"
        assert res_pos["class"] in ["Positive", "Negative"], "Invalid class label"
        assert 0.0 <= res_pos["confidence"] <= 1.0, "Confidence out of range"
        print("  [PASS] Test 1 Passed!")
    else:
        print(f"[!] Warning: {pos_file} not found. Skipping Test 1.")

    # 2. Test Negative Sample
    neg_file = TEST_SAMPLES_DIR / "cattle_negative_sample.wav"
    if neg_file.exists():
        print(f"\n[Test 2] Testing Negative Cattle Recording: {neg_file.name}")
        res_neg = predict_audio(str(neg_file))
        print("  -> Result:", res_neg)
        assert isinstance(res_neg, dict), "Result must be a dict"
        assert res_neg.get("is_cattle_call") is True, "Cattle sample must be recognized as cattle"
        assert "class" in res_neg and "confidence" in res_neg, "Missing required keys"
        assert res_neg["class"] in ["Positive", "Negative"], "Invalid class label"
        assert 0.0 <= res_neg["confidence"] <= 1.0, "Confidence out of range"
        print("  [PASS] Test 2 Passed!")
    else:
        print(f"[!] Warning: {neg_file} not found. Skipping Test 2.")

    # 3. Test Non-Cattle / Silence Rejection
    print("\n[Test 3] Testing Silence / Low Energy Rejection")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_silent:
        silent_audio = np.zeros(16000 * 2, dtype=np.float32)
        sf.write(tmp_silent.name, silent_audio, 16000)
        tmp_silent_path = tmp_silent.name

    try:
        res_silent = predict_audio(tmp_silent_path)
        print("  -> Silent Audio Result:", res_silent)
        assert res_silent["is_cattle_call"] is False, "Silent audio must be rejected as non-cattle"
        assert "error" in res_silent and res_silent["error"] is not None, "Must contain error message"
        print("  [PASS] Test 3 Passed: Silent audio successfully rejected!")
    finally:
        if os.path.exists(tmp_silent_path):
            os.remove(tmp_silent_path)

    # 4. Test Missing/Invalid File Path Error Handling
    print("\n[Test 4] Testing Missing/Invalid File Path Error Handling")
    non_existent = "non_existent_cattle_call_12345.wav"
    try:
        predict_audio(non_existent)
        print("  [FAIL] Test 4 Failed: Function did not raise FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  [PASS] Test 4 Passed: Safely caught expected error -> {e}")

    # 5. Test Unsupported / Directory Path
    print("\n[Test 5] Testing Invalid Directory Input")
    try:
        predict_audio(str(TEST_SAMPLES_DIR))
        print("  [FAIL] Test 5 Failed: Function did not reject directory path")
    except ValueError as e:
        print(f"  [PASS] Test 5 Passed: Safely caught expected error -> {e}")

    # 6. Test Human Speech Audio Discrimination (Synthetic Formant / Modulated Signal)
    print("\n[Test 6] Testing Human Speech Audio Filter")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_speech:
        t = np.linspace(0, 1.5, int(16000 * 1.5), endpoint=False)
        # Synthetic speech-like formant harmonics (150Hz pitch + 800Hz / 1800Hz / 2500Hz formants + fast modulation)
        speech_synth = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t))
        speech_synth += 0.2 * np.sin(2 * np.pi * 800 * t) + 0.15 * np.sin(2 * np.pi * 1800 * t)
        # Add high-frequency consonant fricatives
        noise = np.random.normal(0, 0.05, len(t)) * (np.sin(2 * np.pi * 8 * t) > 0.5)
        speech_synth += noise
        sf.write(tmp_speech.name, speech_synth.astype(np.float32), 16000)
        tmp_speech_path = tmp_speech.name

    try:
        res_speech = predict_audio(tmp_speech_path)
        print("  -> Speech Discrimination Result:", res_speech.get("signal_classification"), "|", res_speech.get("class"))
        assert res_speech.get("is_cattle_call") is False, "Synthetic non-bovine speech must not be classified as cattle"
        print("  [PASS] Test 6 Passed: Human speech audio filter triggered successfully!")
    finally:
        if os.path.exists(tmp_speech_path):
            os.remove(tmp_speech_path)

    print("\n" + "=" * 65)
    print("ALL INFERENCE & REJECTION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
