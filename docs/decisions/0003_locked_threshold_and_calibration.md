# ADR 0003: Locked Diagnostic Thresholds & Probability Calibration

## Status
Accepted

## Context
Raw deep learning logits or sigmoid outputs are uncalibrated overconfident scores rather than true empirical probabilities. Furthermore, applying an ad-hoc 0.50 threshold in medical screening leads to suboptimal sensitivity for high-stakes clinical triaging.

## Decision
1. **Independent Calibration Partition**: Fit non-parametric Isotonic Regression strictly on an independent calibration split (`data/splits/calibration.csv`), completely separated from training and test data.
2. **Locked Threshold Optimization**: Derive decision thresholds on the validation partition targeting high clinical sensitivity (&ge; 90%) to prioritize patient safety in primary screening.
3. **Immutable Configuration Tracking**: Lock the derived thresholds in JSON files (`locked_tau_v002.json`, `locked_tau_v003.json`) with corresponding SHA256 manifests. At runtime, the application loads these locked values and does not compute dynamic thresholds.

## Consequences
- Reliable probabilistic outputs that reflect real empirical risk.
- High screening sensitivity maintained consistently at runtime.
- Full provenance and regulatory traceability for clinical audits.
