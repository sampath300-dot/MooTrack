import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np

from behavior_model.predict import predict_behavior, predict_behavior_video
from behavior_model.config import BEHAVIOR_CLASSES, TEST_SAMPLES_DIR
from behavior_model.video_processor import generate_synthetic_test_video


def run_behavior_tests():
    print("=" * 65)
    print("MOOTRACK - CATTLE BEHAVIOUR RESNET18 INFERENCE VERIFICATION")
    print("=" * 65)

    # 1. Test Static Image Inference
    test_images = list(TEST_SAMPLES_DIR.glob("*.jpg")) + list(TEST_SAMPLES_DIR.glob("*.png"))
    
    if not test_images:
        print("[!] No test images in test_samples. Creating test images...")
        TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        for cls in BEHAVIOR_CLASSES:
            img_path = TEST_SAMPLES_DIR / f"sample_{cls}.jpg"
            arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)
            test_images.append(img_path)

    print(f"--- 1. Testing {len(test_images)} Cattle Behavior Images ---")

    for idx, img_path in enumerate(test_images, 1):
        print(f"[{idx}/{len(test_images)}] Testing Image: {img_path.name}")
        result = predict_behavior(str(img_path))
        print("  -> Result:", {k: v for k, v in result.items() if k != "probabilities"})

        assert isinstance(result, dict), "Result must be a dict"
        assert "class" in result and "confidence" in result, "Result must contain 'class' and 'confidence'"
        assert result["class"] in BEHAVIOR_CLASSES, f"Class must be one of {BEHAVIOR_CLASSES}"
        assert 0.0 <= result["confidence"] <= 1.0, "Confidence must be between 0.0 and 1.0"
        print(f"  [PASS] Successfully classified as: {result['class']} ({result['confidence']*100:.1f}% confidence)\n")

    # 2. Test Video Frame Extraction & Temporal Inference
    print("--- 2. Testing Video Frame Extraction & Temporal Inference ---")
    video_path = TEST_SAMPLES_DIR / "sample_cattle_video.mp4"
    if not video_path.exists():
        print("  Generating synthetic test video...")
        generate_synthetic_test_video(video_path)

    print(f"Testing Video: {video_path.name}")
    video_result = predict_behavior(str(video_path))

    assert isinstance(video_result, dict), "Video result must be a dict"
    assert "class" in video_result, "Video result must have dominant class"
    assert "activity_breakdown" in video_result, "Video result must have activity breakdown"
    assert "timeline" in video_result, "Video result must contain frame timeline"
    assert len(video_result["timeline"]) > 0, "Timeline must contain sampled frames"
    assert "video_metadata" in video_result, "Video result must have metadata"

    print(f"  [PASS] Dominant Video Class: {video_result['class']} (Confidence: {video_result['confidence']*100:.1f}%)")
    print(f"  [PASS] Frames Sampled: {video_result['total_frames_analyzed']} | Duration: {video_result['video_metadata']['duration_seconds']}s")
    print(f"  [PASS] Activity Breakdown: {video_result['activity_breakdown']}")

    # 3. Test Error Handling
    print("\n--- 3. Testing Error Handling ---")
    try:
        predict_behavior("non_existent_cattle_file_12345.jpg")
        print("  [FAIL] Did not raise FileNotFoundError")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  [PASS] Safely caught expected error -> {e}")

    print("\n" + "=" * 65)
    print("ALL CATTLE BEHAVIOUR RESNET18 & VIDEO TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_behavior_tests()
