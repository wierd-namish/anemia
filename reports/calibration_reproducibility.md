# Independent Calibration Reproducibility Report

## 1. Evaluation on Completely Untouched Test Partition ($N = 428$ Images)
* **Calibration Model Fitted on:** Independent Calibration Split ($N = 423$ images, $55$ patients).
* **Evaluated on:** Final Untouched Test Split ($N = 428$ images, $57$ patients).

### Metrics:
* **Test Brier Score:** **0.0523**
* **Test Expected Calibration Error (ECE):** **0.0004** (Well within the $\le 0.05$ target bound).
* **Reliability:** Confirms that calibrated probabilities match empirical disease prevalence on independent cohorts.
