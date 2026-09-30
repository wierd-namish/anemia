# Full Fingernail Anemia Project Verification

**Project Title:** Fingernail-Based Anemia Screening Decision-Support System  
**Evaluation Type:** Comprehensive Software, Experimental, Statistical, and Runtime Audit  
**Execution Timestamp:** 2026-09-30 03:55:40  
**Target Architecture:** Dual-Model Multimodal Ensemble (`efficientnet_b0_v002` + `JetX-GT` Ensemble `v003`)  

---

## 1. Environment Snapshot
* **Operating System:** Windows 11 (AMD64)
* **Python Runtime:** `3.13.7`
* **PyTorch Version:** `2.6.0+cu124`
* **Torchvision Version:** `0.21.0+cu124`
* **CUDA Hardware Acceleration:** ACTIVE (`NVIDIA GeForce RTX 3070 Laptop GPU`, CUDA 12.4)
* **Dataset Manifest:** `data/final_manifest.csv` (4,260 images across 554 patients)

---

## 2. Current Architecture
```
RAW FINGERNAIL IMAGE (Camera / Upload)
  |
  v
INPUT & QUALITY/OOD GATING (assess_image_quality)
  |
  v
NAIL CONTOUR ROI EXTRACTION (NailDetector -> 224x224 RGB)
  |
  +---------------------------------+---------------------------------+
  |                                                                   |
  v                                                                   v
EFFICIENTNET-B0 v002 (CUDA FP16)                   JETX-GT COLORIMETRIC MLP
(Deep 224x224 Convolutions)                       (28 Handcrafted Color Features)
  |                                                                   |
  +---------------------------------+---------------------------------+
                                    |
                                    v
               LOGISTIC FUSION v003 (w_eff*z_eff + w_jetx*z_jetx + b)
                                    |
                                    v
               ISOTONIC REGRESSION v003 (Fitted strictly on Calibration Partition)
                                    |
                                    v
               LOCKED THRESHOLD DECISION ($\tau = 0.9000$)
                                    |
                                    v
               ANEMIA / NO ANEMIA / INCONCLUSIVE
```

---

## 3. Model Artifacts & Configurations
| Component | File Path | File Size | SHA-256 Hash |
| :--- | :--- | :--- | :--- |
| **Primary CNN** | `experiments/efficientnet_b0_v002/best_model.pth` | 16,334,822 B | `a5c0ad6ca54f0d954dc4aad008f79bb3090f47e7059521ec34809cbb272f6f56` |
| **Secondary MLP** | `models/jetx_gt/mlp_model.joblib` | 155,172 B | `d81d660e46fadc642e31a90edd8649e6a07d169af2591da21570ac6d1d6b9c94` |
| **Feature Scaler** | `models/jetx_gt/feature_scaler.joblib` | 1,287 B | `2089f87388ab2ab40e59985ec54053d3bbf77be220ea20cc5352db8974e2c79d` |
| **Logistic Fusion** | `configs/ensemble_fusion_v003.joblib` | 895 B | `6642b82274d5a3a0d0bd5d97b3d5ac43960eec593cca4efb303d86f947b1b272` |
| **Calibrator** | `configs/calibrator_isotonic_v003.joblib` | 742 B | `823c91e4bf78422dae139c265f299ee86830f886a0da7e487d8f418c00130ebc` |
| **Locked Threshold**| `configs/locked_tau_v003.json` | 613 B | `a71e15e05c882cd1fdbe9006a01bde314da0d0c668f81fac40acd0ad16b8a9bf` |

---

## 4. Runtime Verification
* **FastAPI Service:** Online on `http://127.0.0.1:8000`
* **`GET /health`:** HTTP 200 (`healthy`, `model_loaded: true`, `device: cuda`)
* **`GET /model-info`:** HTTP 200 (Complete metadata with investigational disclaimer)
* **`POST /predict`:** HTTP 200 (Single nail inference across Camera and File Upload)
* **`POST /predict-multiple`:** HTTP 200 (Patient-level multi-nail aggregation)
* **Deterministic Inference:** Bitwise identical outputs on repeated input ($f(A) \equiv f(A)$)

---

## 5. Software Tests
* **Total Automated Tests:** 43
* **Tests Passed:** 43 (100%)
* **Failures:** 0
* **Errors:** 0

---

## 6. Experimental Results Matrix (EXP-01 through EXP-15)

| Experiment | Status | Actual Result | Evidence File | Fix Needed | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01: Cross-Split Leakage** | **PASS** | 0 exact, 0 near-duplicates (Max SSIM = 0.8864) | [`leakage_audit_summary.md`](file:///c:/Users/Asus/MYPASS/reports/leakage_audit_summary.md) | None | Visual and metadata independence confirmed across all splits. |
| **EXP-02: Patient Bootstrap** | **PASS** | Sens: 94.3% [86.1%-100%], Spec: 100% | [`patient_bootstrap_summary.md`](file:///c:/Users/Asus/MYPASS/reports/patient_bootstrap_summary.md) | None | Non-parametric patient-stratified variance bounds established. |
| **EXP-03: Multi-Finger Ablation**| **PASS** | 1-finger: 96.4% Acc $\to$ 4-finger: 98.2% Acc | [`multifinger_ablation.md`](file:///c:/Users/Asus/MYPASS/reports/multifinger_ablation.md) | None | Multi-finger mean aggregation stabilizes per-patient predictions. |
| **EXP-04: Calibration Audit** | **PASS** | ECE = 0.0283 (Isotonic), Brier = 0.0094 | [`calibration_comparison.csv`](file:///c:/Users/Asus/MYPASS/reports/calibration_comparison.csv) | None | Reliability diagram confirms smooth calibration on held-out test. |
| **EXP-05: Decision Curve** | **PASS** | Net benefit positive for $p_t \in [0.1, 0.9]$ | [`decision_curve_analysis.csv`](file:///c:/Users/Asus/MYPASS/reports/decision_curve_analysis.csv) | None | Demonstrates clinical triage value over "Screen All" baseline. |
| **EXP-06: 5-Fold Cross-Val** | **PASS** | Mean AUC = 1.0000 across 5 patient folds | [`cross_validation_results.csv`](file:///c:/Users/Asus/MYPASS/reports/cross_validation_results.csv) | None | Confirms algorithmic convergence stability across patient folds. |
| **EXP-07: Grad-CAM Saliency** | **PASS** | Attention localized in nail bed vascular zone | [`gradcam_summary.md`](file:///c:/Users/Asus/MYPASS/reports/gradcam_summary.md) | None | Biological plausibility of final conv feature representations. |
| **EXP-08: Pigmentation Audit** | **NOT RUN** | Demographic skin subgroups not documented | N/A | External demographic data | Subgroup fairness cannot be evaluated without true skin tone codes. |
| **EXP-09: Illumination Stress**| **PASS** | Probability delta $< 0.08$ across 2700K-6500K | [`illumination_stress_test.csv`](file:///c:/Users/Asus/MYPASS/reports/illumination_stress_test.csv) | None | Model is resilient to moderate color temperature shifts. |
| **EXP-10: Physiological OOD** | **PARTIAL** | Non-biological gated; medical nail OOD pending | [`ood_rejection_report.md`](file:///c:/Users/Asus/MYPASS/reports/ood_rejection_report.md) | Curated pathology cohort | Non-biological distractors rejected; onychomycosis pending. |
| **EXP-11: JetX Feature SHAP** | **PASS** | Top: pink_ratio (0.0513), spatial_center_diff | [`jetx_feature_importance.csv`](file:///c:/Users/Asus/MYPASS/reports/jetx_feature_importance.csv) | None | Colorimetric feature ranking verified against physical pallor indices. |
| **EXP-12: Continuous Hb Reg** | **NOT RUN** | Continuous laboratory Hb (g/dL) unavailable | N/A | Reference lab values | Binary dataset cannot support continuous Hb regression modeling. |
| **EXP-13: External Validation** | **NOT RUN** | Independent multi-center cohort unavailable | N/A | External clinical trial | Single-center Ghanaian dataset cannot prove cross-country validity. |
| **EXP-14: JPEG Degradation** | **PASS** | Predictions stable down to Quality 50 | [`quality_degradation.csv`](file:///c:/Users/Asus/MYPASS/reports/quality_degradation.csv) | None | Mobile upload specification set to minimum JPEG quality 60. |
| **EXP-15: Edge Quantization** | **PASS** | FP32: 26.0 ms (15.6 MB) $\to$ INT8: 26.8 ms (6.5 MB)| [`edge_quantization_profile.csv`](file:///c:/Users/Asus/MYPASS/reports/edge_quantization_profile.csv) | None | Desktop CPU dynamic quantization profile completed. |

---

## 7. Fix Log Summary
* **FIX-001:** Backward-compatible baseline configuration paths added to `backend/config.py`.
* **FIX-002:** CUDA-accelerated pairwise SSIM matrix pre-caching in `scripts/fast_ssim_leakage_audit.py`.
* **FIX-003:** Stratified class sampling in Platt calibration fitting within `scripts/run_all_experiments_exp02_to_exp15.py`.
* **FIX-004:** Feature extraction dimension alignment (28 dimensions) in `scripts/run_all_experiments_exp02_to_exp15.py`.
* All details preserved in [`reports/fix_log.md`](file:///c:/Users/Asus/MYPASS/reports/fix_log.md).

---

## 8. Final Multi-Category Evaluation Gate

| Category | Evaluation Status | Evidentiary Basis |
| :--- | :--- | :--- |
| **A. Software Verification** | **VERIFIED (PASS)** | 43/43 automated tests passing, clean FastAPI endpoints, zero mock/hardcoded outputs. |
| **B. Real Inference Verification** | **VERIFIED (PASS)** | Dynamic non-constant inference on RTX 3070 (Logit std: 0.6192, Determinism: True). |
| **C. Dataset Integrity** | **VERIFIED (PASS)** | Zero patient ID overlap, zero cross-split exact duplicates, zero near-duplicates (EXP-01). |
| **D. Statistical Validation** | **VERIFIED (PASS)** | Patient-level bootstrap $95\%$ CI: Sensitivity $[86.1\%, 100.0\%]$, Specificity $100.0\%$ (EXP-02). |
| **E. Robustness** | **VERIFIED (PASS)** | Resilient to 2700K-6500K illumination shifts (EXP-09) and JPEG compression $\ge 50$ (EXP-14). |
| **F. Explainability** | **VERIFIED (PASS)** | Grad-CAM confirms attention energy inside anatomical nail bed vascular zone (EXP-07). |
| **G. External Generalization** | **NOT PROVEN (LIMITATION)** | Single-center retrospective Ghanaian cohort; requires external multi-ethnic cohort. |
| **H. Medical/Clinical Validation**| **NOT PROVEN (LIMITATION)** | Investigational research prototype only; has not undergone prospective clinical trials. |
