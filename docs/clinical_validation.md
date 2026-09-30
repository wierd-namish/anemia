# Clinical Validation & Ethical Governance

## Investigational Status & Regulatory Scope

> [!IMPORTANT]
> **Investigational Device Status**: Anemia AI is currently an **investigational medical software algorithm**. It is intended strictly for clinical trial research, observational screening studies, and algorithm development. It is **not** cleared for independent medical diagnosis by any national regulatory body.

## Reference Standard & Gold Standard Comparison

All developmental evaluations utilize gold-standard clinical laboratory hematology:
- **Reference Assay**: Complete Blood Count (CBC) automated analyzer or HemoCue photometer.
- **Reference Metric**: Whole-blood Hemoglobin concentration (Hb in g/dL).
- **Clinical Cutoffs (WHO Pediatric Guidelines)**:
  - Anemia: Hb < 11.0 g/dL (children 6–59 months)
  - Severe Anemia: Hb < 7.0 g/dL

## STARD-AI & Good Clinical Practice (GCP) Compliance

Clinical evaluations and prospective cohort studies follow the **STARD-AI 2020** (Standards for Reporting of Diagnostic Accuracy Studies for AI) reporting checklist:
- Patient-level split validation (zero cross-partition subject leakage).
- Reporting of both image-level and patient-level metrics.
- Full reporting of 95% confidence intervals across Sensitivity, Specificity, PPV, NPV, and ROC-AUC.
- Complete documentation of out-of-distribution exclusion and image rejection rates.

## Complete Ethics Protocol Documentation

Full clinical protocol and ethics documentation for prospective trial submission are catalogued in `ethics_submission/`:
- `01_protocol/`: Clinical trial protocol (v2.1)
- `02_statistical_plan/`: Sample size justification and SAP
- `03_reference_standard/`: Standard operating procedure (SOP) for laboratory hemoglobin measurement
- `04_participant_documents/`: Informed consent and information sheets (English, Ga, Twi)
- `06_ai_technical_file/`: Full AI technical file, algorithm specification, and traceability matrix
