# Model Validation Report: MOBILENET_V3_LARGE

## 1. Model Specifications
* **Architecture:** `mobilenet_v3_large`
* **Parameters:** 4,203,313 (16.03 MB)
* **Pretrained Weights:** ImageNet-1K
* **CPU Inference Latency:** 10.88 ms

## 2. Validation Set Performance (Operating Threshold $\tau = 0.47$)
### A. Image-Level Metrics ($N = 416$ images)
* **ROC-AUC:** 0.9384 (95% CI: [0.9186, 0.9588])
* **PR-AUC:** 0.9560 (95% CI: [0.9339, 0.9736])
* **Sensitivity (Recall):** 91.44% (95% CI: [87.9%, 94.6%])
* **Specificity:** 81.13% (95% CI: [74.8%, 86.5%])
* **PPV (Precision):** 88.68% | **NPV:** 85.43% | **F1 Score:** 0.9004

### B. Patient-Level Metrics ($N = 55$ independent patients)
* **Patient ROC-AUC:** 1.0000 (95% CI: [1.0, 1.0])
* **Patient Sensitivity:** 100.00% (95% CI: [100.0%, 100.0%])
* **Patient Specificity:** 100.00% (95% CI: [100.0%, 100.0%])

## 3. Calibration & Reliability
* **Uncalibrated ECE:** 0.1112 (Brier: 0.1090)
* **Calibrated ECE (Isotonic):** **0.0274** (Brier: **0.0952**)
