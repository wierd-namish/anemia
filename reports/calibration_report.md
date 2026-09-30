# Probability Calibration & Reliability Report

## 1. Calibration Strategy
* **Calibration Cohort:** Independent $10\%$ calibration partition ($N = 423$ images, $55$ patients).
* **Fitted Calibrator:** Isotonic Regression (`IsotonicRegression(out_of_bounds='clip')`).

---

## 2. Calibration Metrics Comparison
| Model State | Brier Score (Lower is Better) | Expected Calibration Error (ECE) | Reliability Curve Alignment |
|---|---|---|---|
| **Uncalibrated Raw Model** | 0.1041 | 0.1433 | Moderate overconfidence in extremes |
| **Calibrated Model (Isotonic)** | **0.0881** | **0.0578** | **Empirically aligned to true observed frequencies** |
