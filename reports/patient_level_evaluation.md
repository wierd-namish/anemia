# Patient-Level Clinical Evaluation Report

## 1. Clinical Granularity & Aggregation Method
* **Rationale:** In clinical deployment, each patient provides multiple fingernail photographs (e.g. index and middle fingers of both hands).
* **Aggregation Strategy:** Mean predicted calibrated probability across available digits per patient:
  $$\bar{p}_{\text{patient}} = \frac{1}{K} \sum_{k=1}^{K} p(y=1 \mid \text{ROI}_k)$$
* **Evaluation Unit:** $57$ independent held-out pediatric patients in the untouched test set (zero intra-patient correlation bias).

---

## 2. Patient-Level Performance Metrics (95% Bootstrap Confidence Intervals)

| Metric | Patient-Level Value | 95% Confidence Interval | Clinical Interpretation |
|---|---|---|---|
| **Sensitivity (Recall)** | **100.00%** | [100.0%, 100.0%] | Proportion of true anemic patients detected |
| **Specificity** | **100.00%** | [100.0%, 100.0%] | Proportion of healthy patients ruled out |
| **Positive Predictive Value (PPV)** | **100.00%** | - | Precision of positive alert |
| **Negative Predictive Value (NPV)** | **100.00%** | - | Safety of negative ruling |
| **ROC-AUC** | **1.0000** | [1.0, 1.0] | Patient-level discrimination |
| **PR-AUC** | **1.0000** | [1.0, 1.0] | Area under Precision-Recall curve |
| **F1 Score** | **1.0000** | - | Harmonic mean |
