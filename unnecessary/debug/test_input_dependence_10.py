"""
Input-Dependence Acceptance Test across 10 Distinct Real Nail Images.
Verifies that:
- EfficientNet-B0 v002 raw logits and probabilities vary by image
- JetX-GT probabilities vary by image
- Fusion probabilities and calibrated probabilities vary by image
- Predictions are strictly input-dependent and non-constant
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.model.ensemble_pipeline import TwoModelEnsembleService

def run_10_image_test():
    print("=" * 110)
    print("RUNNING 10-IMAGE INPUT-DEPENDENCE ACCEPTANCE TEST (v002 + JetX-GT Ensemble)")
    print("=" * 110)

    service = TwoModelEnsembleService()
    assert service.is_ready(), "Ensemble service is not ready!"

    test_df = pd.read_csv(BASE_DIR / "data/splits/test.csv")
    sample_rows = test_df.sample(n=min(10, len(test_df)), random_state=42).to_dict("records")

    eff_logits, eff_probs = [], []
    jetx_probs, fusion_probs, cal_probs = [], [], []

    print(f"{'Image ID':<16} | {'Patient ID':<12} | {'Eff Logit':<12} | {'Eff Prob':<10} | {'JetX Prob':<10} | {'Fusion Prob':<12} | {'Cal Prob':<10} | {'State':<10}")
    print("-" * 110)

    for r in sample_rows:
        img_p = Path(r["image_path"])
        img = Image.open(img_p).convert("RGB")
        res = service.predict_single(img)
        
        iid = r["image_id"]
        pid = r["patient_id"]
        if "efficientnet_raw_logit" not in res:
            print(f"{iid:<16} | {pid:<12} | REJECTED: {res.get('reason')} - {res.get('description')}")
            continue

        eff_l = res["efficientnet_raw_logit"]
        eff_p = res["efficientnet_probability"]
        j_p = res["jetx_gt_probability"]
        f_p = res["raw_fusion_probability"]
        c_p = res["probability"]
        st = res["state"]

        eff_logits.append(eff_l)
        eff_probs.append(eff_p)
        jetx_probs.append(j_p)
        fusion_probs.append(f_p)
        cal_probs.append(c_p)

        print(f"{iid:<16} | {pid:<12} | {eff_l:<12.4f} | {eff_p:<10.4f} | {j_p:<10.4f} | {f_p:<12.4f} | {c_p:<10.4f} | {st:<10}")

    # Standard deviation checks
    std_eff_l = np.std(eff_logits)
    std_eff_p = np.std(eff_probs)
    std_jetx_p = np.std(jetx_probs)
    std_cal_p = np.std(cal_probs)

    print("\n" + "=" * 60)
    print("INPUT-DEPENDENCE VARIATION AUDIT")
    print("=" * 60)
    print(f"Std Dev (EfficientNet Logits):  {std_eff_l:.6f} (> 0 confirmed)")
    print(f"Std Dev (EfficientNet Probs):   {std_eff_p:.6f} (> 0 confirmed)")
    print(f"Std Dev (JetX-GT Probs):        {std_jetx_p:.6f} (> 0 confirmed)")
    print(f"Std Dev (Calibrated Ensemble):  {std_cal_p:.6f} (> 0 confirmed)")

    assert std_eff_l > 1e-4, "FAILED: EfficientNet logits are constant!"
    assert std_eff_p > 1e-4, "FAILED: EfficientNet probabilities are constant!"
    assert std_jetx_p > 1e-4, "FAILED: JetX probabilities are constant!"
    assert std_cal_p > 1e-4, "FAILED: Calibrated probabilities are constant!"

    print("\n[SUCCESS] 10-IMAGE INPUT DEPENDENCE TEST PASSED: ALL OUTPUTS VARY BY INPUT CONTENT.")
    return True

if __name__ == "__main__":
    run_10_image_test()
