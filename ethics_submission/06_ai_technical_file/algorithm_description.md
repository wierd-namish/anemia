# Algorithm Description & Theoretical Basis

**Document Identifier:** ETH-SUB-TECH-001  
**Investigational System:** EfficientNet-B0 Nail Anemia Assessment Algorithm  
**Version:** 1.0  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Biological & Physiological Rationale

Anemia is characterized by a reduction in total circulating red blood cell mass and sub-physiological hemoglobin concentration, leading to impaired tissue oxygenation. 

In the human body, the subungual capillary bed located beneath the translucent fingernail plate is an accessible, unpigmented microvascular bed where microcirculatory perfusion and pallor can be visually appraised. Under anemic conditions, reduced intravascular oxyhemoglobin absorption manifests as subungual pallor with distinct spectroscopic and chromatic attenuation in the erythema/red-absorption spectral bands.

---

## 2. End-to-End Computational Pipeline

```
┌────────────────────────────────────────────────────────┐
│ 1. INPUT ACQUISITION (Live Mobile Viewfinder)           │
│ Raw RGB camera frame via navigator.mediaDevices        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. IMAGE QUALITY ASSESSMENT GATE                       │
│ Blur (Laplacian var ≥ 5.0), Exposure (40–230 lum),     │
│ Specular Glare (≤ 20%), Polish hue saturation (≤ 15%)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. NAIL ROI CONTOUR DETECTION & EXTRACTION             │
│ YCrCb/HSV subungual contrast contour localization      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. PHYSIOLOGICAL ROI VALIDATION                        │
│ Checks Cr ∈ [130, 180], Cb ∈ [75, 135], Cr_std ≥ 2.5   │
│ Enforces INCONCLUSIVE on non-biological/desk surfaces  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. NORMALIZATION & TENSOR RESIZING                     │
│ 224x224 RGB, ImageNet mean/std standardization         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 6. DEEP CNN FEATURE EXTRACTION & CLASSIFICATION        │
│ EfficientNet-B0 Backbone (Mobile Inverted Bottleneck)   │
│ Linear Classification Head -> Scalar Logit (z)         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 7. MONOTONIC EMPIRICAL PROBABILITY CALIBRATION         │
│ Isotonic Regression: p_cal = Isotonic(Sigmoid(z))      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 8. PATIENT MULTI-IMAGE AGGREGATION                     │
│ P_mean = (1/k) * sum(p_cal_i) across valid digits (k≥1)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 9. LOCKED DIAGNOSTIC THRESHOLD DECISION                │
│ If P_mean ≥ 0.48 -> ANEMIA                             │
│ If P_mean < 0.48 -> NO_ANEMIA                          │
│ If k == 0        -> INCONCLUSIVE                       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

1. **Backbone Feature Mapping:**  
   Given preprocessed nail ROI tensor $X \in \mathbb{R}^{3 \times 224 \times 224}$, the feature extractor produces embedding vector $f(X) \in \mathbb{R}^{1280}$.
2. **Logit Generation:**  
   $$z = W^T f(X) + b \quad (W \in \mathbb{R}^{1280}, b \in \mathbb{R})$$
   $$p_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
3. **Isotonic Calibration:**  
   Given non-parametric isotonic function $g: [0, 1] \to [0, 1]$ fitted on the independent calibration partition:
   $$p_{\text{cal}} = g(p_{\text{raw}})$$
4. **Patient Aggregation & Decision:**  
   $$\bar{P} = \frac{1}{k} \sum_{i=1}^k p_{\text{cal}, i}$$
   $$\text{Output State} = \begin{cases}
   \text{ANEMIA}, & \bar{P} \ge 0.48 \\
   \text{NO\_ANEMIA}, & \bar{P} < 0.48 \\
   \text{INCONCLUSIVE}, & k = 0
   \end{cases}$$
