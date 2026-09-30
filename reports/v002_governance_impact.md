# Clinical Governance & Dossier Impact Report: Model Version Transition (v001 → v002)

**Audit Date:** 2026-09-30  
**Transition Notice:** Investigational Model Version Update from `efficientnet_b0_v001` to `efficientnet_b0_v002`

---

## 1. Context & Rationale

During software runtime auditing, `efficientnet_b0_v001` was identified as an uninitialized checkpoint that collapsed activations to a constant value. 

In accordance with good machine learning practices (GMLP) and medical device software lifecycle standards (IEC 62304 / ISO 13485):
- **The old version `v001` remains archived for complete traceability and forensic integrity.**
- **A genuine transfer-learning model `efficientnet_b0_v002` was trained from frozen patient-partitioned nail images.**
- **All calibration, operating threshold, and technical documentation must be tracked under `v002`.**

---

## 2. Governance Documents Impacted & Revision Requirements

The following frozen regulatory and ethical dossier documents reference `v001` artifacts and must be formally amended prior to prospective clinical deployment:

| Dossier Section | Document File | Current v001 Reference | Required Revision for v002 |
| :--- | :--- | :--- | :--- |
| **01 Protocol** | `ethics_submission/01_protocol/phase_6_clinical_protocol_v2.1.md` | Model: `efficientnet_b0_v001`, Threshold: `0.48` | Update to `efficientnet_b0_v002`, `locked_tau_v002`, `isotonic_regression_v002` |
| **02 Statistical Plan** | `ethics_submission/02_statistical_plan/statistical_analysis_plan_v2.1.md` | Threshold: `0.48` derived from Phase 4 | Document newly derived operating threshold $\tau_{\text{v002}}$ |
| **06 Technical File** | `ethics_submission/06_ai_technical_file/model_version_manifest.json` | Checkpoint: `experiments/efficientnet_b0_v001/best_model.pth` | Update SHA-256 hash to `experiments/efficientnet_b0_v002/best_model.pth` |
| **06 Technical File** | `ethics_submission/06_ai_technical_file/traceability_matrix.csv` | Model version: `efficientnet_b0_v001` | Add `efficientnet_b0_v002` row with new verification dates |
| **06 Technical File** | `ethics_submission/06_ai_technical_file/calibration_specification.md` | Calibrator: `isotonic_regression_v001` | Update to `isotonic_regression_v002` with new calibration curve |
| **07 Risk Assessment** | `ethics_submission/07_risk_management/risk_management_report.md` | Risk Mitigation R-04: Frozen Checkpoint v001 | Update verification record to v002 |

---

## 3. Critical Regulatory Disclaimer

$$\mathbf{IMPORTANT}$$
`efficientnet_b0_v002` is a **NEW INVESTIGATIONAL MODEL VERSION**. It cannot automatically inherit clinical validation claims from historical simulations. Prospective clinical trial validation (Ghanaian Pediatric Cohort 6–59 months) must be conducted using `efficientnet_b0_v002`.
