# Model Validation Report: DENSENET121

## 1. Model Specifications
* **Architecture:** `densenet121`
* **Parameters:** 6,954,881 (26.53 MB)
* **Pretrained Weights:** ImageNet-1K
* **CPU Inference Latency:** 43.5 ms

## 2. Validation Set Performance (Operating Threshold $\tau = 0.47$)
### A. Image-Level Metrics ($N = 416$ images)
* **ROC-AUC:** 0.9354 (95% CI: [0.9148, 0.9544])
* **PR-AUC:** 0.9575 (95% CI: [0.9362, 0.974])
* **Sensitivity (Recall):** 90.27% (95% CI: [87.3%, 93.1%])
* **Specificity:** 81.13% (95% CI: [75.3%, 86.8%])
* **PPV (Precision):** 88.55% | **NPV:** 83.77% | **F1 Score:** 0.8940

### B. Patient-Level Metrics ($N = 55$ independent patients)
* **Patient ROC-AUC:** 0.9914 (95% CI: [0.9724, 1.0])
* **Patient Sensitivity:** 97.14% (95% CI: [90.3%, 100.0%])
* **Patient Specificity:** 95.00% (95% CI: [84.0%, 100.0%])

## 3. Calibration & Reliability
* **Uncalibrated ECE:** 0.1002 (Brier: 0.1107)
* **Calibrated ECE (Isotonic):** **0.0358** (Brier: **0.1019**)
