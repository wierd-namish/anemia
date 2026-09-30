# Institutional Ethics Review Submission Package Checklist

**Target Ethics Committees:**
1. Ghana Health Service Ethics Review Committee (GHS-ERC), Research and Development Division, Accra.
2. Institutional Review Board (IRB), Princess Marie Louise Children's Hospital / Korle Bu Teaching Hospital.
3. Ga West Municipal Hospital Research & Ethics Committee.

**Study Protocol:** Prospective Multi-Center Clinical Validation of an AI Fingernail Assessment System for Pediatric Anemia Screening in Primary Healthcare Settings (Protocol Version 2.1)  
**Date:** 2026-09-30  
**Status:** Pre-Submission Complete Package  

---

## 1. Required Submission Documents Checklist

| Document Item | Document Title / Description | Status / Reference File |
|:---|:---|:---|
| **[X] Item 1** | **Formal Protocol Submission Cover Letter**<br>Addressed to the Chairperson, GHS-ERC. | `docs/ethics/01_ghs_cover_letter.pdf` |
| **[X] Item 2** | **Full Clinical Investigation Protocol (Version 2.1)**<br>Includes study rationale, stratified enrichment sample size justification ($N=350$), reference standard hierarchy (Sysmex vs. HemoCue), WHO 2024 cutoffs, and Statistical Analysis Plan. | [docs/phase_6_clinical_protocol.md](file:///c:/Users/Asus/MYPASS/docs/phase_6_clinical_protocol.md) |
| **[X] Item 3** | **Sample Size & Statistical Precision Plan**<br>Exact Clopper-Pearson binomial power calculations, enrollment inflation formulas, and Bayes' prevalence-adjusted predictive value modeling. | [reports/phase_6_sample_size_justification.md](file:///c:/Users/Asus/MYPASS/reports/phase_6_sample_size_justification.md) |
| **[X] Item 4** | **Parental Informed Consent Forms (ICF)**<br>Written informed consent documents in English, Twi (Akan), and Ga, specifying non-invasive nail photography and standard-of-care venous blood draw. | `docs/ethics/04_informed_consent_forms.md` |
| **[X] Item 5** | **Investigator's Brochure & Software Technical File**<br>Detailed architecture specifications (`EfficientNet-B0`), frozen threshold ($\tau = 0.48$), calibration methodology, quality control gates, and OOD non-nail defense. | [reports/phase_5_acceptance_test.md](file:///c:/Users/Asus/MYPASS/reports/phase_5_acceptance_test.md) |
| **[X] Item 6** | **STARD-AI 2025 Reporting Alignment Audit**<br>Pre-study mapping against the international STARD-AI 2025 diagnostic reporting standard. | [docs/STARD_AI_checklist.md](file:///c:/Users/Asus/MYPASS/docs/STARD_AI_checklist.md) |
| **[X] Item 7** | **Data Protection & Cybersecurity Governance Plan**<br>De-identification protocol, zero facial/biometric storage, in-memory image processing, TLS 1.3 encryption, and compliance with Ghana Data Protection Act 2012 (Act 843). | `docs/ethics/07_data_protection_plan.md` |
| **[X] Item 8** | **Risk-Benefit Analysis Statement**<br>Minimal risk classification: non-invasive smartphone photography; venous phlebotomy performed as part of standard-of-care or under standard clinical safety precautions with no experimental drug/device administration. | Protocol Section 10 & ICF |
| **[X] Item 9** | **Investigator Credentials & Good Clinical Practice (GCP)**<br>Curriculum Vitae and valid CITI / NIH Good Clinical Practice (GCP) certificates of Principal Investigators and Study Coordinators. | Institutional Dossier |
| **[X] Item 10** | **Letters of Site Institutional Support**<br>Endorsement letters from Medical Superintendents of Princess Marie Louise Children's Hospital, Ga West Municipal Hospital, and Central Region Health Directorate. | Institutional Dossier |

---

## 2. Ethical Safeguards & Vulnerable Population Protection

1. **Pediatric Safeguards:** Research involving children (6–59 months) is ethically justified because pediatric anemia in low-resource settings causes irreversible cognitive and developmental deficits, and point-of-care non-invasive screening directly addresses this vulnerable group's healthcare access barriers.
2. **Direct Clinical Benefit:** Any child identified with anemia by the primary laboratory reference standard will immediately receive standard pediatric hematinics, iron supplementation, or clinical referral per Ghana Ministry of Health Pediatric Clinical Management Guidelines, regardless of AI output.
3. **No Diagnostic Delay:** AI imaging takes $< 60\text{ seconds}$ and does not delay clinical blood collection, laboratory evaluation, or medical management.
4. **Voluntary Participation:** Parents/guardians are explicitly informed that refusal to participate will in no way affect their child's standard medical care or treatment.
