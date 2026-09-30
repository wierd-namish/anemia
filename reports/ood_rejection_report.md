# Out-Of-Distribution (OOD) Rejection & Uncertainty Report

## 1. OOD Testing Summary ($N = 7$ Challenging Non-Nail & Degraded Inputs)
* **Goal:** Verify that non-nail images (wood desk, flat skin, clothing fabric, coffee mug, blurry/washed-out frames) **never produce confident positive/negative predictions**.
* **Rejection Mechanism:** Integrated Image Quality Gate + Contour ROI Saliency Guard.
* **Overall OOD Rejection Rate:** **71.4%**

### Detailed Sample Breakdown:
| Test Input Image | Image Quality Gate | ROI Localization Method | Final Pipeline Action | Safe Rejection? |
|---|---|---|---|---|
| `ood_blurred_surface.jpg` | Rejected (Quality Gate) | `contour_nail_detector` | **REJECT / INCONCLUSIVE** | ✅ Yes |
| `ood_clothing_fabric.jpg` | Rejected (Quality Gate) | `contour_nail_detector` | **REJECT / INCONCLUSIVE** | ✅ Yes |
| `ood_overexposed_image.jpg` | Rejected (Quality Gate) | `contour_nail_detector` | **REJECT / INCONCLUSIVE** | ✅ Yes |
| `ood_random_object.jpg` | Pass | `contour_nail_detector` | **PROVISIONAL_INCONCLUSIVE** | ⚠️ Guard Triggered |
| `ood_skin_only.jpg` | Rejected (Quality Gate) | `center_fallback` | **REJECT / INCONCLUSIVE** | ✅ Yes |
| `ood_underexposed_image.jpg` | Rejected (Quality Gate) | `contour_nail_detector` | **REJECT / INCONCLUSIVE** | ✅ Yes |
| `ood_wood_desk.jpg` | Pass | `contour_nail_detector` | **PROVISIONAL_INCONCLUSIVE** | ⚠️ Guard Triggered |

---

## 2. Conclusion on Background Leakage
Unlike the baseline JetX-GT model (which assigned 99.99% anemia probability to a wooden desk), our deep multi-stage pipeline **intercepts and rejects invalid non-nail images**, enforcing an **`INCONCLUSIVE / UNSUITABLE`** status.
