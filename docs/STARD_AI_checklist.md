# STARD-AI (2025) Reporting and Validation Audit Checklist

**Study Title:** Prospective Multi-Center Clinical Validation of an AI Fingernail Assessment System for Pediatric Anemia Screening in Primary Healthcare Settings  
**Target Document:** Phase 6 Clinical Validation Protocol & Evaluation Report  
**Framework:** STARD-AI (Standards for Reporting Diagnostic Accuracy Studies for Artificial Intelligence, 2025 Edition)  
**Date of Audit:** 2026-09-30  
**Audit Status:** PROTOCOL-LOCKED PRE-ENROLLMENT AUDIT  

---

## 1. STARD-AI Item Mapping & Verification Table

| STARD-AI Item | Section / Topic | Protocol Section / Specification | Compliance / Audit Verification |
|:---|:---|:---|:---|
| **1. Title** | AI Identification | Identifies investigational technology as deep convolutional network (`EfficientNet-B0`) assessing anemia from mobile fingernail photographs. | **VERIFIED** — Stated in Title & Protocol Header. |
| **2. Abstract** | Structured Summary | Background, Intended Use, Reference Standard, Metrics (Sensitivity, Specificity with 95% CIs), Intended Population (6–59m). | **VERIFIED** — Protocol Summary Section 11. |
| **3. Clinical Problem** | Intended Purpose | Pediatric anemia screening in low-resource community triage and outpatient settings in sub-Saharan Africa. | **VERIFIED** — Protocol Section 1.1. |
| **4. Intended Use** | SaMD Role | Point-of-care, non-invasive mobile screening triage aid; not a standalone confirmatory test. | **VERIFIED** — Protocol Section 1.1. |
| **5. Target Population** | Demographics | Children aged **6 to 59 months** in Ghana / West Africa. Adults and infants < 6m explicitly excluded. | **VERIFIED** — Protocol Section 1.2. |
| **6. Study Design** | Methodology | Multi-center, prospective, cross-sectional, stratified/enriched diagnostic accuracy study across 3 sites. | **VERIFIED** — Protocol Section 3.1 & Sample Size Justification. |
| **7. Participants** | Eligibility Criteria | Inclusion: Age 6–59m, clinical blood test indication, parental consent. Exclusion: Polish on all digits, severe trauma. | **VERIFIED** — Protocol Section 4. |
| **8. Reference Standard** | Primary Clinical Reference Standard | **Primary:** Laboratory Automated Hematology Analyzer (Sysmex XN-350 / Sysmex XN-550) on Venous EDTA blood. **Secondary:** HemoCue Hb 301. | **VERIFIED** — Protocol Section 5.1. |
| **9. Diagnostic Thresholds** | Ground Truth Cutoffs | **WHO 2024 Pediatric Guidelines:**<br>• 6–23m: $\text{Hb} < 10.5\text{ g/dL}$<br>• 24–59m: $\text{Hb} < 11.0\text{ g/dL}$ | **VERIFIED** — Pre-frozen in Protocol Section 5.2. |
| **10. Index Test** | AI System Specs | `EfficientNet-B0` (`efficientnet_b0_v001`), Isotonic Regression Calibrator, Operating Threshold $\tau = 0.48$. | **VERIFIED** — Frozen in `configs/diagnostic_threshold.json`. |
| **11. Inference Inputs** | Feature Isolation | **Nail photographs only.** Zero clinical metadata (Hb, age, sex, symptoms) accepted by model. | **VERIFIED** — Frozen in `backend/model/inference_pipeline.py`. |
| **12. Image Acquisition** | Standardization | 3–4 nail photos per child, $10\text{--}15\text{ cm}$ distance, flash disabled, multi-device tiers (Samsung, Redmi, iPhone). | **VERIFIED** — Protocol Section 7. |
| **13. Preprocessing** | Image Processing | Image Quality Gate (blur, exposure, glare, polish) $\to$ Nail ROI Contour Extraction $\to$ $224\times 224$ RGB Tensor Normalization. | **VERIFIED** — Protocol Section 8. |
| **14. Unit of Analysis** | Statistical Unit | **Independent Child / Patient** ($N = 300$). Multiple images aggregated via mean calibrated probability. | **VERIFIED** — Protocol Section 8 & 9. |
| **15. Blinding** | Bias Control | AI Operator blinded to reference blood test. Laboratory technologist blinded to AI prediction. | **VERIFIED** — Protocol Section 7.3. |
| **16. Missing / Inconclusive** | Inconclusive Policy | Captured images failing quality gates yield `INCONCLUSIVE`. Formal secondary worst-case sensitivity analysis planned. | **VERIFIED** — Protocol Section 8.3 & SAP. |
| **17. Primary Endpoints** | Predefined Endpoints | **Primary:** Patient Sensitivity ($95\%$ Clopper-Pearson CI). **Co-Primary:** Patient Specificity ($95\%$ Clopper-Pearson CI). | **VERIFIED** — Protocol Section 2.1 & 9.2. |
| **18. Secondary Endpoints** | Multi-metric Evaluation | PPV, NPV, ROC-AUC, PR-AUC, Brier Score, Expected Calibration Error (ECE), Rejection Percentage. | **VERIFIED** — Protocol Section 9.2. |
| **19. Sample Size** | Power & Precision | $N = 350$ enrolled ($N \ge 300$ evaluable; $150$ pos, $150$ neg), powered for $S_e \ge 90\%$ (lower bound $\ge 82\%$) and $S_p \ge 80\%$. | **VERIFIED** — Detailed in `reports/phase_6_sample_size_justification.md`. |
| **20. Subgroup Analyses** | Pre-specified Strata | Age ($6\text{--}23\text{m}$ vs $24\text{--}59\text{m}$), Sex, Anemia Severity, Smartphone Tier, Clinical Site, Malaria/Sickle status. | **VERIFIED** — Protocol Section 9.3. |
| **21. Calibration** | Probability Alignment | Empirical Isotonic Regression calibrated on held-out partition; ECE and reliability diagrams evaluated. | **VERIFIED** — Protocol Section 8.2 & `backend/model/calibration.py`. |
| **22. Explainability** | Saliency Validation | Grad-CAM attention mapped exclusively to subungual capillary nail bed tissue. | **VERIFIED** — Verified in `reports/explainability_audit.md`. |
| **23. Human-AI Interface** | Result Display | Standardized screening states (`ANEMIA`, `NO ANEMIA`, `INCONCLUSIVE`) with `"Model-estimated probability"`. | **VERIFIED** — `frontend/index.html` & `app.js`. |
| **24. Privacy & Ethics** | Data Protection | In-memory processing, anonymized UUIDs, encrypted storage, GHS-ERC ethical review package. | **VERIFIED** — `docs/ethics_submission_checklist.md`. |
| **25. External Generalizability** | Scope of Claims | Validated strictly for pediatric Ghanaian cohort. Adult and non-African claims explicitly withheld. | **VERIFIED** — Protocol Section 1.2 & Persistent Disclaimers. |

---

## 2. Audit Conclusion

The Phase 6 Clinical Protocol meets the 2025 STARD-AI reporting and architectural standards. The protocol is frozen and ready for ethical review submission.
