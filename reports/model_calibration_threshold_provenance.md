# Model → Calibrator → Threshold Provenance Graph

**Date:** 2026-09-30  
**Status:** `CALIBRATION_PROVENANCE_MISMATCH_DETECTED`  
**Required Action:** `RETRAIN_REQUIRED`

---

## 1. Provenance Architecture Graph

```mermaid
graph TD
    A["Training Split (70%)<br>data/splits/train.csv"] -->|Simulated in Phase 4| B["Uninitialized EfficientNet-B0<br>experiments/efficientnet_b0_v001/best_model.pth<br>SHA-256: ca7af0c344aa..."]
    C["Calibration Split (10%)<br>data/splits/calibration.csv"] -->|Simulated Scores (rng.normal)| D["Isotonic Calibrator v001<br>configs/calibrator_isotonic.joblib<br>SHA-256: 1448dfb4aabb..."]
    E["Validation Split (10%)<br>data/splits/val.csv"] -->|Selected on Simulated Scores| F["Locked Threshold tau=0.48<br>configs/clinical_protocol_version.json<br>SHA-256: a6c96aa9e097..."]
    
    B -->|Actual Runtime Inference| G["Collapsed Feature Output (~10^-15)"]
    G -->|Linear Layer Bias| H["Constant Logit: 0.011419"]
    H -->|Sigmoid| I["Constant Probability: 0.502855"]
    I -->|Passed to Calibrator| D
    D -->|Deterministic Piecewise Mapping| J["Constant Calibrated: 0.538342"]
    J -->|Compared to Threshold| F
    F -->|0.538 >= 0.48| K["Constant State: ANEMIA"]
```

---

## 2. Component Digest & Provenance Breakdown

| Component | File Path | Version | SHA-256 | Source / Origin | Integrity Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Model Weights Checkpoint** | `experiments/efficientnet_b0_v001/best_model.pth` | `efficientnet_b0_v001` | `ca7af0c344aa35df9192ce4c508b76d26a7252b2d07b7d309cafc0fd4abd58c6` | `scripts/run_phase_4_complete.py` (`pretrained=False`) | **INVALID FOR INFERENCE** |
| **Calibration Artifact** | `configs/calibrator_isotonic.joblib` | `isotonic_regression_v001` | `1448dfb4aabbecfee53528b78807d9b9d7efd4f7dc8cba4772dc3dae1cdeeb39` | Fitted on synthetic `rng.normal` distribution | **INVALID FOR CURRENT MODEL** |
| **Operating Threshold** | `configs/clinical_protocol_version.json` | `locked_tau_0.48` | `a6c96aa9e0970d110cb105658e23485d45084ce11749da3cbf3be5e95c05244c` | Derived on synthetic validation scores | **INVALID FOR CURRENT MODEL** |

---

## 3. Immediate Technical Requirements for Retraining (v002)

1. **Model Retraining:** Train a genuine `EfficientNet-B0` binary classifier starting from torchvision pretrained ImageNet weights (`EfficientNet_B0_Weights.DEFAULT`) using binary cross-entropy loss with AdamW on `data/splits/train.csv`.
2. **Artifact Versioning:**
   - Model checkpoint: `experiments/efficientnet_b0_v002/best_model.pth`
   - Calibrator: `configs/calibrator_isotonic_v002.joblib` (`isotonic_regression_v002`)
   - Threshold: `locked_tau_<value>` derived on true validation predictions for $\ge 90\%$ sensitivity.
3. **Independent Untouched Evaluation:** Evaluate final performance on the frozen untouched test set `data/splits/test.csv`.
