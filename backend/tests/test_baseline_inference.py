"""
Acceptance Test for Phase 1 & Phase 2:
Validates exact model artifact loading, 28-feature dimensionality, quality check,
and actual baseline inference on sample nail images.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.model.baseline_model import BaselineNailAnemiaModel
from backend.preprocessing.feature_extraction import extract_features, FEATURE_NAMES
from backend.preprocessing.quality import assess_image_quality


def create_sample_nail_images(output_dir: Path):
    """
    Creates test photographic nail images:
    1. Healthy/vascularized nail (normal pinkish tone)
    2. Pale/anemic nail (high pallor, reduced red reflection)
    3. Low-quality blurry image (for quality filter validation)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Healthy nail photograph simulation (rich vascular pink nail bed with cuticle and skin surround)
    img_healthy = Image.new("RGB", (300, 300), color=(195, 145, 125))
    draw = ImageDraw.Draw(img_healthy)
    # Nail bed: healthy reddish-pink
    draw.ellipse([70, 50, 230, 250], fill=(215, 130, 135))
    # Lunula: whitish crescent
    draw.chord([100, 190, 200, 250], start=180, end=360, fill=(240, 220, 220))
    # Add subtle natural texture/noise
    arr = np.array(img_healthy).astype(float)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    healthy_path = output_dir / "sample_healthy_nail.png"
    Image.fromarray(arr).save(healthy_path)
    
    # 2. Anemic nail photograph simulation (pale, whitish-yellow, reduced hemoglobin redness)
    img_anemic = Image.new("RGB", (300, 300), color=(200, 175, 155))
    draw = ImageDraw.Draw(img_anemic)
    # Pale nail bed
    draw.ellipse([70, 50, 230, 250], fill=(235, 215, 210))
    # Prominent pallor
    draw.ellipse([90, 80, 210, 220], fill=(245, 235, 230))
    arr_anemic = np.array(img_anemic).astype(float)
    noise_anemic = np.random.normal(0, 5, arr_anemic.shape)
    arr_anemic = np.clip(arr_anemic + noise_anemic, 0, 255).astype(np.uint8)
    anemic_path = output_dir / "sample_anemic_nail.png"
    Image.fromarray(arr_anemic).save(anemic_path)
    
    # 3. Blurry image (low Laplacian variance)
    img_blur = Image.new("RGB", (300, 300), color=(180, 180, 180))
    blur_path = output_dir / "sample_blurry_fail.png"
    img_blur.save(blur_path)
    
    return healthy_path, anemic_path, blur_path


def run_acceptance_test():
    print("=" * 70)
    print("PHASE 1 & PHASE 2 ACCEPTANCE TEST: BASELINE MODEL INFERENCE")
    print("=" * 70)
    
    # Step 1: Initialize Model
    print("\n[1] Initializing BaselineNailAnemiaModel...")
    model = BaselineNailAnemiaModel()
    print(f"  - Model classes: {model.classes_}")
    print(f"  - Expected features: {model.n_features_in_}")
    print(f"  - Scaler features: {model.scaler.n_features_in_}")
    assert model.n_features_in_ == 28, f"Expected 28 features, got {model.n_features_in_}"
    print("  ✓ Model and Scaler loaded and verified (28 features).")
    
    # Step 2: Prepare Sample Images
    samples_dir = Path(__file__).resolve().parent.parent.parent / "data" / "samples"
    healthy_p, anemic_p, blur_p = create_sample_nail_images(samples_dir)
    print(f"\n[2] Created sample nail images in {samples_dir}:")
    print(f"  - Healthy sample: {healthy_p.name}")
    print(f"  - Anemic sample: {anemic_p.name}")
    print(f"  - Blurry reject sample: {blur_p.name}")
    
    # Step 3: Run Inference on Anemic Nail Image
    print("\n[3] Running Inference on Sample Anemic Nail Photograph...")
    img_anemic = Image.open(anemic_p)
    result_anemic = model.predict_image(img_anemic, threshold=0.10)
    print(f"  Result:")
    print(f"    - Success: {result_anemic['success']}")
    print(f"    - Diagnosis: {result_anemic['diagnosis']}")
    print(f"    - Model Score: {result_anemic['model_score']}")
    print(f"    - Is Calibrated: {result_anemic['is_calibrated']}")
    print(f"    - Confidence: {result_anemic['confidence']}")
    print(f"    - Threshold Used: {result_anemic['threshold_used']}")
    print(f"    - Explanation: {result_anemic['explanation']}")
    print(f"    - Feature Summary: {result_anemic['feature_summary']}")
    assert result_anemic["success"] is True
    
    # Step 4: Run Inference on Healthy Nail Image
    print("\n[4] Running Inference on Sample Healthy/Vascularized Nail Photograph...")
    img_healthy = Image.open(healthy_p)
    result_healthy = model.predict_image(img_healthy, threshold=0.10)
    print(f"  Result:")
    print(f"    - Success: {result_healthy['success']}")
    print(f"    - Diagnosis: {result_healthy['diagnosis']}")
    print(f"    - Model Score: {result_healthy['model_score']}")
    print(f"    - Is Calibrated: {result_healthy['is_calibrated']}")
    print(f"    - Confidence: {result_healthy['confidence']}")
    print(f"    - Threshold Used: {result_healthy['threshold_used']}")
    print(f"    - Explanation: {result_healthy['explanation']}")
    print(f"    - Feature Summary: {result_healthy['feature_summary']}")
    assert result_healthy["success"] is True
    
    # Step 5: Test Quality Gate (Blurry image rejection)
    print("\n[5] Testing Image Quality Gate on Low-Quality/Featureless Image...")
    img_blur = Image.open(blur_p)
    result_blur = model.predict_image(img_blur, threshold=0.10)
    print(f"  Result:")
    print(f"    - Success: {result_blur['success']}")
    print(f"    - Error: {result_blur.get('error')}")
    print(f"    - Message: {result_blur.get('message')}")
    print(f"    - Quality Metrics: {result_blur.get('quality_metrics')}")
    assert result_blur["success"] is False, "Quality filter should reject featureless/blurry images."
    print("  ✓ Quality gate correctly rejected poor quality image without fabricating a YES/NO diagnosis.")
    
    print("\n" + "=" * 70)
    print("ALL PHASE 1 & PHASE 2 ACCEPTANCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_acceptance_test()
