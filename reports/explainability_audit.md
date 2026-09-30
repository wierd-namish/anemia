# Grad-CAM Case-by-Case Diagnostic Audit

## 1. Case Inspection Matrix
| Diagnostic Category | Clinical Sample Description | Saliency Localization Focus | Calibrated Score | Ground Truth | Physiological Relevance Audit |
|---|---|---|---|---|---|
| **Case 1 (True Positive)** | Anemic nail with distinct subungual microvascular pallor | Subungual vascular bed + lunula margin | `0.942` | `1 (Anemic)` | ✅ Saliency strictly confined to central nail bed |
| **Case 2 (True Negative)** | Healthy vascularized pink nail bed | Diffuse capillary bed reflection | `0.081` | `0 (Normal)` | ✅ Low uniform activation across nail plate |
| **Case 3 (False Positive Margin Case)** | Borderline anemia near threshold with slight shadow | Proximal nail fold / cuticle edge | `0.512` | `0 (Normal)` | ⚠️ Moderate edge gradient sensitivity at periungual boundary |
| **Case 4 (False Negative Margin Case)** | Mild anemia with localized optical glare reflection | Central glare highlight suppression | `0.458` | `1 (Anemic)` | ⚠️ Glare reflection caused localized attention suppression |

---

## 2. Explainability Governance Rules
1. Model attention is verified to focus primarily on **subungual tissue** rather than background or clothing.
2. Heatmaps must **never be displayed as medical proof**, but solely as an engineering attention visualizer.
