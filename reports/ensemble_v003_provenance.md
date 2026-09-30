# Provenance & Architectural Specification: Two-Model Ensemble (v003)

**Ensemble Identifier:** `ensemble_v003`  
**Primary Model:** `efficientnet_b0_v002` (Deep Vision Backbone)  
**Secondary Model:** `JetX-GT/nail-anemia-detector` (Hugging Face, 27 Color Features + MLP)  
**Fusion Model:** Empirical Logistic Regression Fusion on Development Partition  
**Calibrator:** `isotonic_regression_v003`  
**Locked Threshold:** $\tau_{\text{v003}} = 0.9000$  

---

## 1. Dual-Model Architecture & Pipeline Flow

```text
                     SAME NAIL IMAGE / ROI
                               |
                +--------------+--------------+
                |                             |
                v                             v
       EfficientNet-B0 v002        JetX-GT nail model (HF)
       (224x224 RGB Tensor)       (27 Handcrafted Color Features)
                |                             |
                v                             v
          Raw Logit / Prob              Raw Probability
                |                             |
                +--------------+--------------+
                               |
                               v
                         Fusion Model
                     (Logistic Regression)
                               |
                               v
                    Isotonic Calibrator v003
                               |
                               v
                     Locked Threshold v003
                               |
                               v
                        Final Assessment
```

---

## 2. Artifact Verification & Hashes

| Artifact Component | File Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Primary Model** | `experiments/efficientnet_b0_v002/best_model.pth` | `a5c0ad6ca54f0d954dc4aad008f79bb3090f47e7059521ec34809cbb272f6f56` |
| **Secondary MLP** | `models/jetx_gt/mlp_model.joblib` | `d81d660e46fadc642e31a90edd8649e6a07d169af2591da21570ac6d1d6b9c94` |
| **Secondary Scaler** | `models/jetx_gt/feature_scaler.joblib` | `2089f87388ab2ab40e59985ec54053d3bbf77be220ea20cc5352db8974e2c79d` |
| **Secondary Metadata** | `models/jetx_gt/model_metadata.json` | `dac21daf75d75e642a9e11b77a4adcd8f606da18d08303e1b46e5d90aba9cc0b` |
| **Fusion Model** | `configs/ensemble_fusion_v003.joblib` | `6642b82274d5a3a0d0bd5d97b3d5ac43960eec593cca4efb303d86f947b1b272` |
| **Calibrator v003** | `configs/calibrator_isotonic_v003.joblib` | `823c91e4bf78422dae139c265f299ee86830f886a0da7e487d8f418c00130ebc` |
| **Locked Threshold** | `configs/locked_tau_v003.json` | `a71e15e05c882cd1fdbe9006a01bde314da0d0c668f81fac40acd0ad16b8a9bf` |

---

## 3. Data Leakage Control & Provenance Verification

- **Hugging Face Model Card Disclosure:** The published `JetX-GT/nail-anemia-detector` model was trained on the Ghanaian cohort development images.
- **Evaluation Rule:** To avoid optimistic bias, JetX performance was audited strictly against our locked patient-partitioned validation and test sets.
- **Fusion Training:** The fusion weights were fitted **only** on the development training split ($N = 2,993$ images, zero patient overlap with validation, calibration, or test sets).
