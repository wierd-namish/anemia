# Patient-Specific Acquisition Shift & Covariate Balance Audit

## 1. Partition Balance & Metadata Audit
Evaluated across Train ($N=2,993$), Validation ($N=416$), Calibration ($N=423$), and Test ($N=428$):
* **Class Ratio Balance:** Anemia prevalence is tightly conserved ($63.3\% \pm 0.4\%$) across all partitions.
* **Aspect Ratio & Resolution:** $100\%$ of images across splits share identical native acquisition format.
* **Camera Sensor & Flash:** Standardized hospital camera setup (spotlights/flash turned off).
* **Demographic Isolation:** Strict patient-level separation verified (0 patient ID or image hash overlap).
