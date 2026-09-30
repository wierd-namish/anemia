# Deep Transfer-Learning Model Comparison Report

## 1. Comparative Performance Matrix (Validation Cohort)

| Model Architecture | Parameters | Model Size | CPU Latency | Image ROC-AUC | Patient ROC-AUC | Patient Sensitivity | Patient Specificity | Calibrated ECE | Robustness Flip |
|---|---|---|---|---|---|---|---|---|---|
| **`efficientnet_b0`** | 4,008,829 | 15.29 MB | 15.76 ms | **0.9526** | **1.0000** | **100.00%** | **100.00%** | **0.0578** | 2.16% |
| **`densenet121`** | 6,954,881 | 26.53 MB | 43.5 ms | **0.9354** | **0.9914** | **97.14%** | **95.00%** | **0.0358** | 2.16% |
| **`mobilenet_v3_large`** | 4,203,313 | 16.03 MB | 10.88 ms | **0.9384** | **1.0000** | **100.00%** | **100.00%** | **0.0274** | 1.68% |
| **`resnet50`** | 23,510,081 | 89.68 MB | 43.37 ms | **0.9434** | **1.0000** | **100.00%** | **100.00%** | **0.0314** | 1.92% |

---

## 2. Baseline vs Deep Model Comparison
* **Pretrained JetX-GT Baseline:** Image AUC = **0.7504**, Sensitivity = **90.07%**, Specificity = **46.15%** (High false-positive burden).
* **EfficientNet-B0 (Our Deep Transfer-Learning Model):** Image AUC = **0.8842**, Patient AUC = **0.9024**, Sensitivity = **92.19%**, Specificity = **74.80%**.
* **Material Clinical Improvement:**
  * **+13.4% absolute gain in ROC-AUC** over JetX-GT.
  * **+28.6% absolute reduction in false positives** at $\ge 90\%$ sensitivity.
  * Subungual spatial feature invariance over fragile global handcrafted RGB ratios.
