# Diagnostic Threshold Specification

**Document Identifier:** ETH-SUB-TECH-006  
**Version:** `locked_tau_0.48`  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Operating Threshold Selection

The operating threshold was selected on the independent validation partition ($N = 55$ patients, $424$ images) by optimizing specificity under the clinical safety constraint:

$$\text{Maximize } \text{Specificity}(\tau) \quad \text{subject to } \text{Sensitivity}(\tau) \ge 90.0\%$$

* **Selected Locked Threshold:** **$\tau = 0.48$**
* **Operating Point (Validation Set):** Sensitivity $= 91.83\%$, Specificity $= 85.53\%$, Accuracy $= 89.42\%$.
* **Ordering Rule:** The threshold is strictly applied to **Calibrated Probabilities** ($p_{\text{cal}}$), never raw CNN logits.
* **Artifact Path:** `configs/diagnostic_threshold.json`
