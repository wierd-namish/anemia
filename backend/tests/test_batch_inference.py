"""
Batch Inference Test & High-Precision Evaluation Script for JetX-GT Baseline Model.

Processes a directory of test images, saves original vs processed images for visual inspection,
performs high-precision probability and decision-function extraction, calculates pairwise
feature differences, checks for saturation, and outputs results.csv.
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.preprocessing.feature_extraction import extract_features, extract_feature_dict, FEATURE_NAMES
from backend.preprocessing.quality import assess_image_quality
from backend.config import BASELINE_MODEL_PATH, BASELINE_SCALER_PATH, BASELINE_METADATA_PATH


def run_batch_inference(input_dir: Path, output_csv: Path, debug_dir: Path):
    print("=" * 80)
    print("PHASE 2B: BATCH INFERENCE & HIGH-PRECISION MODEL VALIDATION")
    print("=" * 80)
    
    # 1. Load Pretrained Artifacts
    model = joblib.load(BASELINE_MODEL_PATH)
    scaler = joblib.load(BASELINE_SCALER_PATH)
    
    input_dir = Path(input_dir)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"\nFound {len(image_files)} test images in {input_dir}\n")
    
    results = []
    feature_vectors = {}
    
    for img_path in image_files:
        img_name = img_path.name
        stem = img_path.stem
        
        # Create debug visual directory for this image
        img_debug_dir = debug_dir / stem
        img_debug_dir.mkdir(parents=True, exist_ok=True)
        
        orig_img = Image.open(img_path)
        orig_img.save(img_debug_dir / "original.jpg")
        
        # Save processed 224x224 RGB image (exact input to feature extractor)
        proc_img = orig_img.convert("RGB").resize((224, 224))
        proc_img.save(img_debug_dir / "processed.jpg")
        
        # Quality check
        quality_passed, quality_reason, quality_metrics = assess_image_quality(orig_img)
        
        # Extract 28 features
        feat_vec = extract_features(orig_img)
        feature_vectors[img_name] = feat_vec
        
        # Scale
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))[0]
        
        # Raw forward pass
        z0 = np.dot(feat_scaled, model.coefs_[0]) + model.intercepts_[0]
        a0 = np.maximum(0, z0)
        z1 = np.dot(a0, model.coefs_[1]) + model.intercepts_[1]
        a1 = np.maximum(0, z1)
        raw_logit = float(np.dot(a1, model.coefs_[2])[0] + model.intercepts_[2][0])
        
        prob_class_1 = float(1.0 / (1.0 + np.exp(-raw_logit)))
        prob_class_0 = float(1.0 - prob_class_1)
        
        # Decision at thresholds
        thresh_balanced = 0.10
        thresh_high_sens = 0.255
        dec_balanced = "Anemia-associated pattern" if prob_class_1 >= thresh_balanced else "No anemia pattern detected"
        dec_high_sens = "Anemia-associated pattern" if prob_class_1 >= thresh_high_sens else "No anemia pattern detected"
        
        results.append({
            "filename": img_name,
            "raw_logit": f"{raw_logit:.8f}",
            "class_0_prob": f"{prob_class_0:.16f}",
            "class_1_prob": f"{prob_class_1:.16f}",
            "raw_score": f"{prob_class_1:.6f}",
            "decision_thresh_0_10": dec_balanced,
            "decision_thresh_0_255": dec_high_sens,
            "feat_min": f"{feat_vec.min():.4f}",
            "feat_max": f"{feat_vec.max():.4f}",
            "feat_mean": f"{feat_vec.mean():.4f}",
            "quality_passed": quality_passed,
            "quality_reason": quality_reason,
        })
        
        print(f"[{stem}]")
        print(f"  - Quality Passed: {quality_passed} ({quality_reason})")
        print(f"  - Feature Stats:  min={feat_vec.min():.4f}, max={feat_vec.max():.4f}, mean={feat_vec.mean():.4f}")
        print(f"  - Raw Logit (z2): {raw_logit:.12f}")
        print(f"  - Class 0 Prob:   {prob_class_0:.16f}")
        print(f"  - Class 1 Prob:   {prob_class_1:.16f}")
        print(f"  - Decision (0.10): {dec_balanced}")
        print()
        
    # Write CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "filename", "raw_logit", "class_0_prob", "class_1_prob", "raw_score",
        "decision_thresh_0_10", "decision_thresh_0_255", "feat_min", "feat_max",
        "feat_mean", "quality_passed", "quality_reason"
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Results successfully saved to: {output_csv}\n")
    
    # 2. Pairwise Feature Saturation Analysis
    print("=" * 80)
    print("PAIRWISE FEATURE SATURATION & DISTANCE ANALYSIS")
    print("=" * 80)
    
    names = list(feature_vectors.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            vec_a, vec_b = feature_vectors[name_a], feature_vectors[name_b]
            diff = np.abs(vec_a - vec_b)
            max_d = float(np.max(diff))
            mean_d = float(np.mean(diff))
            euc_d = float(np.linalg.norm(vec_a - vec_b))
            print(f"Distance [{name_a}] vs [{name_b}]:")
            print(f"  - Euclidean Distance:     {euc_d:.4f}")
            print(f"  - Mean Absolute Diff:     {mean_d:.4f}")
            print(f"  - Max Absolute Diff:      {max_d:.4f}")
            
    print("\n" + "=" * 80)
    print("MODEL EXECUTION VERIFIED: official model artifacts load and inference executes successfully.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/test_images")
    parser.add_argument("--output-csv", default="data/results.csv")
    parser.add_argument("--debug-dir", default="data/processed_debug")
    args = parser.parse_args()
    
    run_batch_inference(args.input_dir, args.output_csv, args.debug_dir)
