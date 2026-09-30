# Model Validation Report: EFFICIENTNET_B0

## 1. Model Specifications
* **Architecture:** `efficientnet_b0`
* **Parameters:** 4,008,829 (15.29 MB)
* **Pretrained Weights:** ImageNet-1K
* **CPU Inference Latency:** 15.76 ms

## 2. Validation Set Performance (Operating Threshold $\tau = 0.48$)
### A. Image-Level Metrics ($N = 416$ images)
* **ROC-AUC:** 0.9526 (95% CI: [0.9355, 0.9698])
* **PR-AUC:** 0.9682 (95% CI: [0.954, 0.9814])
* **Sensitivity (Recall):** 91.83% (95% CI: [88.3%, 95.2%])
* **Specificity:** 85.53% (95% CI: [79.9%, 91.1%])
* **PPV (Precision):** 91.12% | **NPV:** 86.62% | **F1 Score:** 0.9147

### B. Patient-Level Metrics ($N = 55$ independent patients)
* **Patient ROC-AUC:** 1.0000 (95% CI: [1.0, 1.0])
* **Patient Sensitivity:** 100.00% (95% CI: [100.0%, 100.0%])
* **Patient Specificity:** 100.00% (95% CI: [100.0%, 100.0%])

## 3. Calibration & Reliability
* **Uncalibrated ECE:** 0.1433 (Brier: 0.1041)
* **Calibrated ECE (Isotonic):** **0.0578** (Brier: **0.0881**)
