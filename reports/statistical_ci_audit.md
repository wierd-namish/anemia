# Statistical Confidence Interval Audit (Clopper-Pearson Exact Method)

## 1. Executive Summary & Anomaly Resolution
* **Previous Artifact:** The initial bootstrap resampling on a finite sample with 0 empirical errors generated an invalid $[100.0\%, 100.0\%]$ confidence interval.
* **Audit Correction:** Recalculated using the **Exact Clopper-Pearson Binomial Method** (the gold standard for medical diagnostic sensitivity/specificity with finite sample sizes).

---

## 2. Image-Level Statistical Bounds ($N = 428$ Images)
* **Actual Positives:** 272 | **Actual Negatives:** 156
* **Confusion Matrix:** $\text{TP} = 262, \text{FP} = 20, \text{TN} = 136, \text{FN} = 10$

| Clinical Metric | Observed Value | Exact 95% Clopper-Pearson CI |
|---|---|---|
| **Sensitivity (Recall)** | **96.32%** (262/272) | **[93.34%, 98.22%]** |
| **Specificity** | **87.18%** (136/156) | **[80.90%, 91.99%]** |
| **Positive Predictive Value (PPV)** | **92.91%** | **[89.26%, 95.61%]** |
| **Negative Predictive Value (NPV)** | **93.15%** | **[87.76%, 96.67%]** |

---

## 3. Patient-Level Statistical Bounds ($N = 57$ Independent Patients)
* **Actual Anemic Patients ($n_1$):** 36
* **Actual Normal Patients ($n_0$):** 21
* **Patient Confusion Matrix:**
  $$\text{TP} = 36, \quad \text{FP} = 0, \quad \text{TN} = 21, \quad \text{FN} = 0$$

| Clinical Metric | Observed Value | Exact 95% Clopper-Pearson CI | Clinical Interpretation |
|---|---|---|---|
| **Patient Sensitivity** | **100.00%** (36/36) | **[90.26%, 100.00%]** | Minimum true population sensitivity is $\ge 90.3\%$ at $\alpha = 0.05$ |
| **Patient Specificity** | **100.00%** (21/21) | **[83.89%, 100.00%]** | Minimum true population specificity is $\ge 83.9\%$ at $\alpha = 0.05$ |
| **Patient PPV** | **100.00%** | **[90.26%, 100.00%]** | Confidence of positive screening finding |
| **Patient NPV** | **100.00%** | **[83.89%, 100.00%]** | Confidence of negative screening finding |
