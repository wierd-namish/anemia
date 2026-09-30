# Probability Calibration Specification

**Document Identifier:** ETH-SUB-TECH-005  
**Version:** `isotonic_regression_v001`  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Mathematical Formulation

Raw sigmoid outputs from deep convolutional networks frequently exhibit overconfidence or non-linear distortion. To transform neural network scores into empirically reliable probabilities, non-parametric **Isotonic Regression** was fitted on the held-out calibration partition ($N = 423$ images, $55$ patients):

$$\min_{g \in \mathcal{M}} \sum_{i=1}^M \left( y_i - g(p_{\text{raw}, i}) \right)^2$$

where $\mathcal{M}$ is the set of all non-decreasing piecewise constant monotonic functions on $[0, 1]$.

## 2. Reliability Gains
* **Uncalibrated Expected Calibration Error (ECE):** $0.1420$ (Brier: $0.1041$)
* **Calibrated Expected Calibration Error (ECE):** **$0.0384$** (Brier: **$0.0881$**)
* **Artifact Path:** `configs/calibrator_isotonic.joblib`
