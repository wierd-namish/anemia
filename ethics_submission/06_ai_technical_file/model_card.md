# Model Card: EfficientNet-B0 Subungual Nail Anemia Classifier

**Model Name:** `EfficientNet-B0` (Pediatric Nail Anemia Classifier)  
**Model Version:** `efficientnet_b0_v001`  
**Model Date:** 2026-09-30  
**Model Type:** Deep Convolutional Neural Network (Transfer Learning from ImageNet Backbone)  
**Calibration:** Monotonic Isotonic Regression (`isotonic_regression_v001`)  
**Operating Threshold:** $\tau = 0.48$ (Locked)  
**Intended Task:** Binary assessment of pediatric anemia probability from isolated fingernail ROI photographs.  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Intended Use & Target Population

* **Primary Intended Use:** Mobile-based non-invasive triage screening aid for evaluating pediatric anemia probability in primary healthcare, triage, and community screening settings.
* **Intended Population:** Children aged **6 to 59 months** in Ghana and West Africa.
* **Out-of-Scope / Prohibited Uses:**
  - Standalone medical diagnosis without confirmatory blood testing.
  - Adult patient screening (NOT validated).
  - Infants aged 0–5 months (NOT validated in primary diagnostic claim).
  - Generalization to non-African Fitzpatrick skin phototypes (I–III) without independent prospective validation.
  - Screening on toes, skin patches, mucous membranes, or non-biological surfaces.

---

## 2. Model Architecture & Pipeline Specifications

* **Backbone:** `EfficientNet-B0` (5.3M parameters, 15.29 MB uncompressed footprint).
* **Classifier Head:** Dropout ($p=0.3$) $\to$ Linear ($1280 \to 1$).
* **Input Specifications:** $224 \times 224$ RGB image tensor, normalized by ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).
* **Zero Metadata Input:** The model consumes strictly isolated fingernail pixels; zero clinical variables (Hb, CBC, age, sex, symptoms) are accepted.

---

## 3. Retrospective Development & Evaluation Dataset

* **Cohort:** Retrospective Ghanaian pediatric dataset ($N = 552$ normalized unique patients, $4,260$ total fingernail images).
* **Ground Truth Source:** Venous/capillary blood hemoglobin measured at the time of photograph acquisition.
* **4-Way Zero-Leakage Patient-Level Split:**
  - **Train Partition (70%):** $387\text{ patients}$ ($2,986\text{ images}$)
  - **Validation Partition (10%):** $55\text{ patients}$ ($424\text{ images}$) — used for early stopping and threshold selection.
  - **Calibration Partition (10%):** $55\text{ patients}$ ($423\text{ images}$) — used strictly to fit Isotonic Calibrator.
  - **Locked Test Partition (10%):** $57\text{ patients}$ ($427\text{ images}$) — locked independent holdout.

---

## 4. Retrospective Benchmark Performance (Internal Test Set)

> [!NOTE]
> **Retrospective vs. Prospective Performance Distinction:**  
> The metrics below reflect retrospective internal evaluation on the frozen Ghanaian development split. True real-world performance will be established in the Phase 6 prospective clinical validation study.

| Metric | Image-Level Metric (427 Images) | Patient-Level Metric (57 Independent Children) |
|:---|:---|:---|
| **ROC-AUC** | $0.9425$ ($95\%\text{ CI: } [0.918, 0.967]$) | **$1.0000$** ($95\%\text{ CI: } [0.937, 1.000]$) |
| **PR-AUC** | $0.9632$ | **$1.0000$** |
| **Sensitivity** | $91.83\%$ ($95\%\text{ CI: } [88.3\%, 95.2\%]$) | **$100.0\%$** (Exact $95\%\text{ Clopper-Pearson: } [90.5\%, 100.0\%]$) |
| **Specificity** | $85.53\%$ ($95\%\text{ CI: } [79.9\%, 91.1\%]$) | **$100.0\%$** (Exact $95\%\text{ Clopper-Pearson: } [83.2\%, 100.0\%]$) |
| **Brier Score** | $0.0881$ (Calibrated) | — |
| **Expected Calibration Error (ECE)** | **$0.0384$** (Down from $0.1420$ uncalibrated) | — |
| **Inference Latency (CPU)** | **$15.76\text{ ms}$** | — |

---

## 5. Quality Gates & Out-of-Distribution (OOD) Defenses

1. **Blur Gate:** Laplacian variance threshold $< 5.0 \implies$ `INCONCLUSIVE` (`severe_blur`).
2. **Exposure Gate:** Mean luminance $< 40.0 \implies$ `underexposed`; $> 230.0 \implies$ `overexposed`.
3. **Glare Gate:** $> 20.0\%$ saturated pixels $\implies$ `excessive_glare`.
4. **Nail Polish Gate:** Non-physiological high-saturation hues $> 15.0\% \implies$ `nail_polish_detected`.
5. **Physiological ROI Saliency Guard:** Rejects flat wood backgrounds (historical JetX regression immunity), uniform skin, and textiles with `100% INCONCLUSIVE` assignment.

---

## 6. Known Failure Modes & Limitations

1. **Severely Deformed or Missing Nails:** Onychogryphosis or complete nail loss prevents contour localization.
2. **Dense Non-Removable Pigments:** Traditional henna or opaque enamel covering subungual capillary beds cannot be penetrated.
3. **Severe Peripheral Vasoconstriction:** Cold extremities, Raynaud's phenomenon, or hypovolemic shock alter subungual capillary filling.
