# Model Validation Report: RESNET50

## 1. Model Specifications
* **Architecture:** `resnet50`
* **Parameters:** 23,510,081 (89.68 MB)
* **Pretrained Weights:** ImageNet-1K
* **CPU Inference Latency:** 43.37 ms

## 2. Validation Set Performance (Operating Threshold $\tau = 0.46$)
### A. Image-Level Metrics ($N = 416$ images)
* **ROC-AUC:** 0.9434 (95% CI: [0.9232, 0.9624])
* **PR-AUC:** 0.9631 (95% CI: [0.9471, 0.9785])
* **Sensitivity (Recall):** 91.05% (95% CI: [87.3%, 94.2%])
* **Specificity:** 83.65% (95% CI: [78.2%, 88.9%])
* **PPV (Precision):** 90.00% | **NPV:** 85.26% | **F1 Score:** 0.9052

### B. Patient-Level Metrics ($N = 55$ independent patients)
* **Patient ROC-AUC:** 1.0000 (95% CI: [1.0, 1.0])
* **Patient Sensitivity:** 100.00% (95% CI: [100.0%, 100.0%])
* **Patient Specificity:** 100.00% (95% CI: [100.0%, 100.0%])

## 3. Calibration & Reliability
* **Uncalibrated ECE:** 0.1282 (Brier: 0.1077)
* **Calibrated ECE (Isotonic):** **0.0314** (Brier: **0.0930**)
