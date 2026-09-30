# Calibration & Threshold Sequence Audit

## 1. Sequence Verification (Pipeline Architecture B Locked)
To prevent threshold drift after non-linear isotonic probability mapping, the system strictly implements **Pipeline Architecture B**:

```
1. Train Deep CNN (Backbone + Head on Train Split)
        │
        ▼
2. Fit Probability Calibrator on CALIBRATION Split (Isotonic Regression)
        │
        ▼
3. Transform Validation Set to Calibrated Probabilities
        │
        ▼
4. Select Operating Threshold (	au = 0.48) on CALIBRATED Validation Probabilities
        │
        ▼
5. LOCK Model, Calibrator, and Threshold Simultaneously
        │
        ▼
6. Evaluate Untouched Final TEST Split
```

* **Consistency Guarantee:** The threshold $\tau = 0.48$ was selected **directly on calibrated probabilities**, eliminating post-calibration threshold mismatch.
