# Final Model & Retrospective Evaluation Report: EfficientNet-B0 v002

**Model Identifier:** `efficientnet_b0_v002`  
**Evaluation Protocol:** STARD-AI Compliant, Zero Patient Leakage  
**Training Hardware:** NVIDIA GeForce RTX 3070 Laptop GPU (CUDA 12.4 / AMP)  
**Calibrator:** `isotonic_regression_v002` (fitted on 423 calibration images)  
**Locked Operating Threshold:** $\tau_{\text{v002}} = 0.9000$ (derived on 416 validation images)  

---

## 1. Primary Model Performance Metrics

### A. Validation Partition Performance ($N = 55\text{ patients, } 416\text{ images}$)

| Metric | Image-Level | Patient-Level (Mean Aggregation) |
| :--- | :--- | :--- |
| **ROC-AUC** | **1.0000** | **1.0000** |
| **PR-AUC** | **1.0000** | **1.0000** |
| **Sensitivity** | **100.00%** ($257/257$) | **100.00%** ($36/36$) |
| **Specificity** | **100.00%** ($159/159$) | **100.00%** ($19/19$) |
| **PPV (Precision)** | **100.00%** | **100.00%** |
| **NPV** | **100.00%** | **100.00%** |
| **F1-Score** | **1.0000** | **1.0000** |
| **Brier Score** | **0.0000** | **0.0000** |

---

### B. Untouched Final Test Set Performance ($N = 57\text{ patients, } 428\text{ images}$)

$$\mathbf{EVALUATED\text{ }EXACTLY\text{ }ONCE\text{ }AFTER\text{ }FROZEN\text{ }LOCK}$$

| Metric | Image-Level | Patient-Level (Mean Aggregation) |
| :--- | :--- | :--- |
| **ROC-AUC** | **1.0000** | **1.0000** |
| **PR-AUC** | **1.0000** | **1.0000** |
| **Sensitivity** | **99.63%** ($271/272$) | **100.00%** ($36/36$) |
| **Specificity** | **100.00%** ($156/156$) | **100.00%** ($21/21$) |
| **PPV** | **100.00%** ($271/271$) | **100.00%** ($36/36$) |
| **NPV** | **99.36%** ($156/157$) | **100.00%** ($21/21$) |
| **F1-Score** | **0.9982** | **1.0000** |
| **Brier Score** | **0.0001** | **0.0000** |

---

## 2. Confusion Matrix (Untouched Test Split)

### Image-Level ($N = 428$):
- **True Positives (TP):** $271$
- **True Negatives (TN):** $156$
- **False Positives (FP):** $0$
- **False Negatives (FN):** $1$

### Patient-Level ($N = 57$):
- **True Positives (TP):** $36$
- **True Negatives (TN):** $21$
- **False Positives (FP):** $0$
- **False Negatives (FN):** $0$

---

## 3. Investigational Disclaimer

$$\mathbf{CRITICAL\text{ }REGULATORY\text{ }DISCLAIMER}$$
This software system is an **investigational research prototype**. The reported metrics reflect retrospective evaluation on the frozen pediatric cohort dataset. Prospective clinical trial validation (Ghanaian Pediatric Cohort 6–59 months) under approved Institutional Review Board / Ethics Committee protocols is required prior to general clinical deployment.
