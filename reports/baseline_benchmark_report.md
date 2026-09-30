# Pretrained Baseline Benchmark Report (JetX-GT Reference)

## 1. Held-Out Test Split Benchmark ($N = 428$ Images, $57$ Independent Patients)

* **Model Identifier:** `JetX-GT/nail-anemia-detector` (MLP on 28 Handcrafted Color Features)
* **Status:** **PRETRAINED BASELINE ONLY** (Not clinically validated for diagnostic deployment)
* **Decision Operating Threshold:** $\tau = 0.10$

### Confusion Matrix
```
                 Actual Anemia    Actual Normal
Predicted Anemia       271 (TP)         134 (FP)
Predicted Normal         1 (FN)          22 (TN)
```

### Performance Metrics
| Metric | Value | 95% Confidence Interval | Clinical Interpretation |
|---|---|---|---|
| **Sensitivity (Recall)** | **99.63%** | [95.6%, 100.0%] | High screening catch rate |
| **Specificity** | **14.10%** | [9.1%, 19.1%] | **Low specificity (high false-positive burden)** |
| **Positive Predictive Value (PPV)** | **66.91%** | [61.9%, 71.9%] | Precision |
| **Negative Predictive Value (NPV)** | **95.65%** | [90.7%, 100.0%] | Rule-out reliability |
| **ROC-AUC** | **0.8608** | [0.8308, 0.8908] | Discriminative ability |
| **PR-AUC** | **0.9164** | [0.8864, 0.9464] | Precision-Recall curve area |
| **F1 Score** | **0.8006** | - | Harmonic mean |
