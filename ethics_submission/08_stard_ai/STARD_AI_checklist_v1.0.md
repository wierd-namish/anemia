# STARD-AI (2025) Reporting Checklist (Submission Master)

**Document Identifier:** ETH-SUB-STD-001  
**Version:** 1.0  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

| Item No. | STARD-AI Item | Section / Description | Compliance Verification |
|:---|:---|:---|:---|
| **1** | **Title** | Identifies algorithm as deep CNN (`EfficientNet-B0`) assessing anemia from nail photos. | **VERIFIED** |
| **2** | **Abstract** | Structured summary including primary sensitivity/specificity with 95% CIs. | **VERIFIED** |
| **3** | **Clinical Problem** | Pediatric anemia screening in low-resource community triage and outpatient settings. | **VERIFIED** |
| **4** | **Intended Use** | Point-of-care mobile screening triage aid; non-invasive triage tool. | **VERIFIED** |
| **5** | **Target Population** | Children aged **6 to 59 months** in Ghana / West Africa (0–5m exploratory segregated). | **VERIFIED** |
| **6** | **Study Design** | Multi-center prospective stratified/enriched diagnostic accuracy study ($N=350$). | **VERIFIED** |
| **7** | **Participants** | Inclusion: Age 6–59m, clinical blood test indication, parental consent. | **VERIFIED** |
| **8** | **Reference Standard** | **Primary:** Laboratory Automated Hematology Analyzer (Sysmex XN-350 / XN-550) on Venous EDTA blood. **Secondary:** HemoCue Hb 301. | **VERIFIED** |
| **9** | **Diagnostic Thresholds** | WHO 2024 Guidelines: 6–23m ($<10.5\text{ g/dL}$), 24–59m ($<11.0\text{ g/dL}$). | **VERIFIED** |
| **10** | **Index Test** | `EfficientNet-B0` (`efficientnet_b0_v001`), Isotonic Calibrator, Threshold $\tau = 0.48$. | **VERIFIED** |
| **11** | **Inference Inputs** | Isolated fingernail photographs only; zero clinical metadata. | **VERIFIED** |
| **12** | **Image Acquisition** | 3–4 nail photos per child, $10\text{--}15\text{ cm}$ distance, flash disabled, multi-device tiers. | **VERIFIED** |
| **13** | **Preprocessing** | Quality gate $\to$ YCrCb contour segmentation $\to 224\times 224$ RGB normalization. | **VERIFIED** |
| **14** | **Unit of Analysis** | **Child / Patient** ($N = 300$). Multi-digit mean calibrated probability. | **VERIFIED** |
| **15** | **Blinding** | AI operator blinded to reference lab result; laboratory blinded to AI score. | **VERIFIED** |
| **16** | **Inconclusive Handling** | Inconclusive policy predefined; primary evaluable and ITD worst-case bounding. | **VERIFIED** |
| **17** | **Primary Endpoints** | Patient Sensitivity ($95\%$ Clopper-Pearson CI) and Specificity ($95\%$ CI). | **VERIFIED** |
| **18** | **Secondary Endpoints** | PPV, NPV, ROC-AUC (DeLong), PR-AUC, Brier score, ECE, rejection percentage. | **VERIFIED** |
| **19** | **Sample Size** | $N = 350$ enrolled ($N \ge 300$ evaluable; $150$ pos, $150$ neg), powered for $S_e \ge 90\%$. | **VERIFIED** |
| **20** | **Subgroup Analyses** | Age (6–23m vs 24–59m), Anemia severity, Sex, Smartphone tier, Study site. | **VERIFIED** |
| **21** | **Calibration** | Isotonic Regression calibration on held-out partition; ECE evaluated. | **VERIFIED** |
| **22** | **Explainability** | Grad-CAM subungual capillary localization verification. | **VERIFIED** |
| **23** | **Human-AI Interface** | Controlled screening states (`ANEMIA`, `NO ANEMIA`, `INCONCLUSIVE`). | **VERIFIED** |
| **24** | **Data Governance** | Ghana Data Protection Act 2012 (Act 843) compliance, in-memory processing. | **VERIFIED** |
| **25** | **Generalizability** | Scope limited to pediatric Ghanaian cohort. Adult/non-African claims excluded. | **VERIFIED** |
