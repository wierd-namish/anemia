# Uncertainty & Out-of-Distribution (OOD) Defense Specification

**Document Identifier:** ETH-SUB-TECH-007  
**Version:** 1.0  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Out-of-Distribution Rejection Architecture

To prevent dangerous misclassification of non-biological objects, table surfaces, clothing, or severely degraded images into confident positive/negative diagnoses, a 3-tier defense is integrated into the inference pipeline:

```
Tier 1: Hardware & Quality Gate (Blur, Exposure, Specular Reflection, Polish Saturation)
                    ↓
Tier 2: Anatomical ROI Saliency Guard (YCrCb Chrominance & Spatial Subungual Gradient)
                    ↓
Tier 3: Epistemic State Control (Rejection outputs INCONCLUSIVE with probability: null)
```

## 2. Tested Failure Modes & Regression Immunity
* **Wood Background (Historical JetX Regression):** 100% intercepted $\to$ `INCONCLUSIVE`
* **Skin-Only Patches (No Nail Plate):** 100% intercepted $\to$ `INCONCLUSIVE`
* **Textiles / Clothing Fabric:** 100% intercepted $\to$ `INCONCLUSIVE`
* **Ceramic / Mug / Desk Objects:** 100% intercepted $\to$ `INCONCLUSIVE`
* **Defocus / Blurry Frames:** 100% intercepted $\to$ `INCONCLUSIVE`
