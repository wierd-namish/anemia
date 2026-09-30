"""
Model Checkpoint and Artifact Verification Script.
Audits checkpoint hashes, PyTorch state_dict tensors, and model manifests.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from anemia_ai.config.settings import get_settings
from anemia_ai.models.efficientnet import EfficientNetB0Model
from anemia_ai.models.jetx_nail import JetXNailModel
from anemia_ai.utils.hashing import compute_sha256


def verify_checkpoints() -> bool:
    settings = get_settings()
    print("=" * 70)
    print("ANEMIA AI — MODEL CHECKPOINT AUDIT & VERIFICATION")
    print("=" * 70)

    # 1. EfficientNet-B0 v002 Checkpoint
    v002_path = settings.v002_model_path
    print(f"\n[1] Checking Primary Vision Checkpoint: {v002_path}")
    if not v002_path.exists():
        print(f"  [FAIL] Checkpoint not found at: {v002_path}")
        return False

    v002_hash = compute_sha256(v002_path)
    size_mb = v002_path.stat().st_size / (1024 * 1024)
    print(f"  - File Size:   {size_mb:.2f} MB")
    print(f"  - SHA256:      {v002_hash}")

    model = EfficientNetB0Model()
    if not model.is_ready():
        print("  [FAIL] Model instantiation / state_dict loading failed.")
        return False
    print(f"  [OK] Primary Vision Model loaded successfully on {model.device}.")

    # 2. JetX-GT Artifacts
    print(f"\n[2] Checking Hugging Face JetX-GT Model Artifacts: {settings.jetx_cache_dir}")
    jetx = JetXNailModel()
    if not jetx.is_ready():
        print("  [FAIL] JetX-GT model artifacts failed verification.")
        return False
    for fname, sha in jetx.hashes.items():
        print(f"  - {fname:25}: SHA256 {sha[:16]}...")
    print("  [OK] JetX-GT model verified and ready.")

    # 3. Calibration & Fusion Artifacts
    print(f"\n[3] Checking Calibration & Ensemble Fusion Artifacts:")
    calib_v002 = settings.calibrator_v002_path
    calib_v003 = settings.calibrator_v003_path
    fusion_path = settings.fusion_model_path

    for label, p in [
        ("Calibrator v002", calib_v002),
        ("Calibrator v003", calib_v003),
        ("Fusion Model v003", fusion_path),
    ]:
        if p.exists():
            h = compute_sha256(p)
            print(f"  - {label:20}: SHA256 {h[:16]}... ({p.stat().st_size} bytes)")
        else:
            print(f"  - {label:20}: NOT FOUND at {p}")

    print("\n" + "=" * 70)
    print("CHECKPOINT & ARTIFACT VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_checkpoints()
    sys.exit(0 if success else 1)
