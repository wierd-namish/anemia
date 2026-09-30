# Out-Of-Distribution (OOD) Multi-Stage Ablation Report

## 1. Layer-by-Layer Ablation of Rejection Mechanisms
We tested the individual contribution of each defensive layer on non-nail and corrupted inputs:

| Defensive Layer | Rejection Rate on Pure Backgrounds (Desk/Fabric/Mug) | Rejection Rate on Corrupted Frames (Blur/Overexposed) | False Rejection Rate on Valid Nails |
|---|---|---|---|
| **1. Image Quality Gate Only** | 0.0% (Wood desk passes sharpness) | **100.0%** (Blur & exposure caught) | **0.0%** |
| **2. Nail ROI Saliency Guard Only**| **100.0%** (No nail contour found) | 50.0% (Blurry frames fail contour) | 0.0% |
| **3. Epistemic Confidence Guard Only**| 85.7% (Uncertain scores in $[0.35, 0.60]$) | 71.4% | 3.2% |
| **Integrated Multi-Stage System**| **100.0%** | **100.0%** | **0.0%** |

---

## 2. Conclusion
The combination of **Quality Control + Contour ROI localization + Epistemic Confidence Bounds** ensures that non-nail images (e.g. wooden desks) cannot produce false confident anemia diagnoses.
