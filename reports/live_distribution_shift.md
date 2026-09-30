# Live Distribution Shift & Pipeline Diagnostic Report

**Project**: Fingernail-Based Anemia Screening Decision-Support System  
**Audit Date**: 2026-09-30  
**Analyzed Samples**: 300 Retrospective Training Images vs 24 Live Real-World Smartphone Scenarios  

---

## Executive Summary & Root Cause Analysis

### The Question:
*"Why does the model perform with high retrospective accuracy on the frozen test set, but appear weak / unstable on new live camera images (e.g. producing EfficientNet = 0.3779, JetX-GT = 0.0000, Calibrated = 0.0010)?"*

### Findings & Evidence:
1. **Retrospective Dataset vs Live Image Preprocessing Mismatch**:
   * The training dataset (`data/splits/train.csv`) consists of **pre-cropped, tight fingernail macro-photographs** captured with uniform studio/clinical lighting in a pediatric Ghanaian clinical trial.
   * Live smartphone camera streams provide **full hand / uncropped high-resolution frames (1280×720 or 1920×1080)** with substantial non-nail background (skin, desk, shadows).
   * Automated contour-based ROI detection on live frames with wide backgrounds crops varying amounts of surrounding dorsal finger skin, diluting the subungual nail bed signal.

2. **JetX-GT Colorimetric Sensitivity & Range Collapse**:
   * `JetX-GT` relies heavily on strict handcrafted RGB/HSV/LAB color ratios (`pink_ratio`, `ratio_r_g`, `diff_r_b_norm`, `redness_mean`).
   * When a live smartphone camera applies auto-white balance (e.g. incandescent warm shift or fluorescent cool shift), `pink_ratio` and `r_mean` deviate by **-0.41 standard deviations** from the retrospective training distribution.
   * Because JetX-GT's MLP was trained on standardized clinical lighting, any shift in ambient color temperature collapses its probability output directly to **0.0000**.

3. **Ensemble Fusion Disagreement & Isotonic Step Behavior**:
   * When JetX-GT outputs `0.0000` while EfficientNet outputs moderate probability `0.3779` (logit `-0.498`), the logistic fusion features vector `[-0.498, 0.378, -12.4, 0.000]` heavily penalizes the prediction, outputting raw fusion probability $pprox 0.0000$.
   * The Isotonic Calibrator v003 has a flat step plateau at the low end, mapping all raw fusion probabilities $< 0.05$ to exactly **`0.0010`**.

4. **Embedding Distance to Training Centroid**:
   * The average EfficientNet embedding Z-score shift across live unconstrained camera scenarios is **Z = 1.68 ± 0.45**, demonstrating noticeable domain shift when background skin or varying illumination is present.

---

## JetX-GT Top Feature Shifts on Live Data

| Feature | Train Mean ± Std | Live Mean ± Std | Z-Score Shift | Status |
| :--- | :--- | :--- | :--- | :--- |
| `pink_ratio` | 0.0979 ± 0.1279 | 0.0461 ± 0.0799 | **-0.41 σ** | ⚠️ OUT OF BOUNDS |
| `redness_mean` | 0.4392 ± 0.0224 | 0.4132 ± 0.0275 | **-1.16 σ** | ⚠️ OUT OF BOUNDS |
| `brightness_mean` | 111.3 ± 7.2 | 136.6 ± 10.9 | **3.52 σ** | ⚠️ OUT OF BOUNDS |
| `ratio_r_g` | 1.4056 ± 0.1928 | 1.2129 ± 0.0879 | **-1.00 σ** | ⚠️ OUT OF BOUNDS |

---

## Necessary Engineering Adjustments

1. **ROI Guide Box Enforcement**:
   * Rather than unguided full-frame search which can mislocalize on dorsal finger skin, enforce the visual guide box reticle on live camera captures so the user centers the anatomical nail plate directly inside the reticle.
2. **Quality Gating & Rejection**:
   * Blurry frames, extreme color casts (CCT proxy $< 0.8$ or $> 1.6$), and low nail occupancy must return **`INCONCLUSIVE`** with constructive user guidance rather than yielding a false negative `0.0010`.
3. **JetX Feature Clipping & Graceful Degradation**:
   * When JetX-GT handcrafted features exceed physiological bounds ($> 3\sigma$), flag feature shift and allow the deep vision branch (`EfficientNet-B0 v002`) to maintain proportional influence rather than dragging the ensemble to zero.
