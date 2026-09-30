# Prospective Clinical Validation Protocol (Phase 6) — Final Lock

**Protocol Title:** Prospective Multi-Center Clinical Validation of an AI Fingernail Assessment System for Pediatric Anemia Screening in Primary Healthcare Settings  
**Protocol Version:** 2.1 (Protocol Locked for Ethical Review Submission)  
**Date:** 2026-09-30  
**Study Phase:** Phase 6 Prospective Clinical Validation  
**Investigational System:** EfficientNet-B0 Subungual Nail Anemia Diagnostic System (`efficientnet_b0_v001`, Operating Threshold $\tau = 0.48$, Isotonic Calibration)  
**Intended Regulatory Pathway:** Software as a Medical Device (SaMD) / In Vitro Diagnostic (IVD) Triage Aid  

---

## 1. Intended Use and Target Population

### 1.1 Intended Use Statement
The Investigational AI System is intended for use by trained healthcare workers and community health officers as a non-invasive, point-of-care mobile screening tool to evaluate the probability of anemia from mobile phone fingernail photographs in pediatric patients presenting to outpatient clinics, community health posts, and triage centers.

The software outputs a **Model-Estimated Probability** and a categorical screening state:
* `ANEMIA` (High probability of anemia, prompting confirmatory laboratory blood testing)
* `NO_ANEMIA` (Low probability of anemia)
* `INCONCLUSIVE` (Image quality or anatomical ROI unsuitable for diagnostic assessment)

### 1.2 Primary Target Population
* **Age Group:** **Children aged 6 to 59 months** (infants aged 6–23 months and preschool children aged 24–59 months).
* **Clinical Setting:** Pediatric outpatient departments, community health posts (CHPS compounds), nutrition assessment clinics, and triage centers.
* **Geographic / Demographic Scope:** Ghanaian / West African pediatric population presenting at designated study sites.

> [!IMPORTANT]
> **Definitive Validation Boundaries & Non-Claims:**
> * **Adult Performance:** Adult diagnostic validity is **NOT validated**.
> * **0–5 Month Infants:** Performance in infants under 6 months is **NOT validated** in the primary clinical study.
> * **Unstudied Populations:** Populations outside the enrolled 6–59 month pediatric cohort, and non-African Fitzpatrick skin phototypes (I–III), are **NOT validated**.

---

## 2. Study Objectives & Hypotheses

### 2.1 Primary Objective
To prospectively evaluate the diagnostic sensitivity of the locked AI fingernail assessment system against a contemporaneous primary laboratory automated hematology reference standard in children aged **6 to 59 months**.

### 2.2 Co-Primary Objective
To prospectively evaluate the diagnostic specificity of the locked AI system at the locked patient-level threshold ($\tau = 0.48$) in children aged **6 to 59 months**.

### 2.3 Secondary Objectives
1. To evaluate concordance between the Primary Reference Standard (Laboratory Automated Hematology Analyzer) and the Secondary Comparator (Point-of-Care Hemoglobinometer).
2. To evaluate AI sensitivity across anemia severity tiers (mild, moderate, severe).
3. To determine image quality rejection rates across smartphone hardware tiers.
4. To evaluate Positive and Negative Likelihood Ratios ($\text{LR}^+, \text{LR}^-$) and Diagnostic Odds Ratios (DOR).

### 2.4 Statistical Hypotheses
* **Primary Null Hypothesis ($H_0$):** True patient-level sensitivity $\le 80.0\%$
* **Primary Alternative Hypothesis ($H_1$):** True patient-level sensitivity $> 80.0\%$ (Target $\ge 90.0\%$, lower bound of two-sided $95\%$ CI $\ge 82.0\%$)
* **Co-Primary Specificity Target:** Specificity $\ge 80.0\%$ (lower bound of two-sided $95\%$ CI $\ge 70.0\%$)

---

## 3. Study Design & Diagnostic Comparison Architecture

### 3.1 Design Classification: Stratified / Enriched Diagnostic Accuracy
The study employs a **Stratified / Enriched Prospective Diagnostic Accuracy Design** across three diverse clinical sites. Enrollment is managed to ensure approximately equal representation across ground-truth categories ($150$ confirmed anemic and $150$ confirmed non-anemic evaluable participants), guaranteeing statistical power for both sensitivity and specificity endpoints without requiring an excessively large unstratified cohort.

```
                 ENROLLED PARTICIPANT (6–59 Months)
                                 ↓
    ┌────────────────────────────┴────────────────────────────┐
    ↓                                                         ↓
[ AI IMAGING OPERATOR ]                                [ PHLEBOTOMIST / LAB ]
Trained Study Nurse                                   Certified Technologist
(Blinded to Clinical & Lab Data)                      (Blinded to AI Predictions)
    ↓                                                         ↓
Standardized Mobile Nail Capture                      Venous Blood Draw (EDTA tube)
(3–4 Nail Photographs)                                [Primary Specimen]
    ↓                                                 + Optional Capillary Finger-prick
Image Quality Gate & ROI Isolation                    [Secondary Specimen]
    ↓                                                         ↓
EfficientNet-B0 Calibrated Inference                  PRIMARY REFERENCE STANDARD:
    ↓                                                 Laboratory Automated Hematology Analyzer
Locked Mean Patient Probability                       (Sysmex XN-350 / XN-550)
    ↓                                                         +
Diagnostic State (τ = 0.48)                           SECONDARY COMPARATOR:
(ANEMIA / NO_ANEMIA / INCONCLUSIVE)                   Point-of-Care Hemoglobinometer
                                                      (HemoCue Hb 301)
                                                              ↓
                                                      WHO 2024 Age-Specific Anemia Truth:
                                                      • 6–23m:  Hb < 10.5 g/dL
                                                      • 24–59m: Hb < 11.0 g/dL
    └────────────────────────────┬────────────────────────────┘
                                 ↓
           [ INDEPENDENT DATA MANAGEMENT / STATISTICIAN ]
                   Blinded Statistical Audit & SAP
```

### 3.2 Clinical Study Sites
1. **Site A (Urban Tertiary Pediatric Center):** Princess Marie Louise Children's Hospital / Korle Bu Teaching Hospital, Accra.
2. **Site B (Peri-Urban District Hospital):** Ga West Municipal Hospital, Amasaman.
3. **Site C (Rural Primary Care / CHPS Compound):** Central Region Community Health Center.

---

## 4. Participant Eligibility & Recruitment

### 4.1 Primary Inclusion Criteria
1. Child aged between **6 months (exact 180 days) and 59 months** at the date of visit.
2. Presenting to outpatient clinic, triage, or community nutrition screening.
3. Clinical indication for routine or diagnostic blood hemoglobin evaluation.
4. Written informed consent signed by parent or legally authorized representative (LAR).

### 4.2 Exclusion Criteria
1. Age $< 6\text{ months}$ or $\ge 60\text{ months}$ (infants $< 6\text{ months}$ eligible only for the separate exploratory cohort).
2. Opaque nail polish, henna, or artificial coverings on all accessible fingernails.
3. Active severe trauma, burns, or extensive fungal onychomycosis obliterating nail beds.
4. Blood transfusion received within the preceding 14 days.
5. Hemodynamic instability or hypovolemic shock requiring emergency resuscitation.
6. Failure to obtain venous reference blood draw within 30 minutes of nail photography.

---

## 5. Clinical Reference Standard Hierarchy & Labeling Rules

### 5.1 Reference Standard Classification

| Hierarchy Level | Technology / Instrument | Sample Type | Operating Procedure | Role in Study |
|:---|:---|:---|:---|:---|
| **PRIMARY REFERENCE STANDARD** | **Laboratory Automated Hematology Analyzer** (Sysmex XN-350 / Sysmex XN-550) | Venous Blood in EDTA Microtainer ($\ge 0.5\text{ mL}$) | Run within 2 hours of phlebotomy in hospital central laboratory; daily 3-level commercial quality control. | **Primary ground truth** for all primary sensitivity and specificity calculations. |
| **SECONDARY COMPARATOR** | **Point-of-Care Hemoglobinometer** (HemoCue Hb 301 System) | Capillary Blood (microcuvette via standardized finger-prick) | Immediate point-of-care analysis in clinic; calibrated using manufacturer optical reference daily. | **Secondary comparator** to evaluate POC capillary concordance and microcuvette variance. |

### 5.2 Handling of Diagnostic Discordance
* If discordance occurs between the Primary Reference Standard (Sysmex venous Hb) and Secondary Comparator (HemoCue capillary Hb), the **Primary Reference Standard (Sysmex venous Hb) is the sole definitive ground truth** for primary efficacy evaluation.
* Secondary analyses will quantify Sysmex-vs-HemoCue discordance using Bland-Altman limits of agreement and Cohen's Kappa statistic.

### 5.3 Pre-Frozen WHO 2024 Age-Specific Diagnostic Cutoffs

Ground-truth anemia labels are determined strictly by **WHO 2024 pediatric hemoglobin guidelines**:

$$\text{Primary Ground Truth Label} = \begin{cases} 
1 \text{ (Anemia)}, & \text{if } \text{Age } \mathbf{6\text{ to }23\text{ months}} \text{ and } \text{Venous Hb} < \mathbf{10.5\text{ g/dL}} \ (105\text{ g/L}) \\
1 \text{ (Anemia)}, & \text{if } \text{Age } \mathbf{24\text{ to }59\text{ months}} \text{ and } \text{Venous Hb} < \mathbf{11.0\text{ g/dL}} \ (110\text{ g/L}) \\
0 \text{ (No Anemia)}, & \text{otherwise}
\end{cases}$$

#### WHO 2024 Severity Stratification:
* **Children 6–23 Months:**
  - **Mild Anemia:** $9.5 \le \text{Hb} < 10.5\text{ g/dL}$
  - **Moderate Anemia:** $7.0 \le \text{Hb} < 9.5\text{ g/dL}$
  - **Severe Anemia:** $\text{Hb} < 7.0\text{ g/dL}$
* **Children 24–59 Months:**
  - **Mild Anemia:** $10.0 \le \text{Hb} < 11.0\text{ g/dL}$
  - **Moderate Anemia:** $7.0 \le \text{Hb} < 10.0\text{ g/dL}$
  - **Severe Anemia:** $\text{Hb} < 7.0\text{ g/dL}$

---

## 6. Separate 0–5 Month Exploratory Cohort (Non-Primary)

> [!NOTE]
> **Exploratory Protocol Section:**  
> Infants aged **0 to 5 months** ($< 180\text{ days}$) are enrolled under a distinct, non-interventional exploratory arm. Because the WHO 2024 guideline does not provide an equivalent universal threshold for this age window due to physiological infant hemoglobin fluctuations, this cohort:
> 1. Is **completely segregated** from the primary prospective statistical endpoint.
> 2. Will be evaluated using local pediatric reference intervals ($< 10.0\text{ g/dL}$ exploratory boundary).
> 3. Does not contribute to regulatory claims of sensitivity/specificity for the primary 6–59 month intended use.

---

## 7. Image Acquisition Protocol & Precise Blinding Controls

### 7.1 Multi-Device Smartphone Allocation
* **Tier 1 (Mid-Range Android):** Samsung Galaxy A14 / A15 (50MP sensor)
* **Tier 2 (Entry-Level Android):** Xiaomi Redmi 12C / Tecno Spark 10 (13–50MP sensor)
* **Tier 3 (iOS Benchmark):** Apple iPhone 11 / 12 (12MP sensor)

### 7.2 Image Capture Schedule
* Exactly **$N = 4$ nail photographs** per participant (Index, Middle, Ring of dominant hand; Thumb/Index of non-dominant hand).
* Positioned inside the UI guide box at $10\text{--}15\text{ cm}$ distance under ambient lighting with flash disabled.
* Metadata recorded for robustness auditing only: `device_model`, `os_version`, `lighting_lux`, `finger_digit`, `capture_timestamp`.

### 7.3 Precise Blinding Protocol
1. **AI Operator Blinding:** The study nurse operating the camera is blinded to the participant's reference blood hemoglobin level, CBC parameters, clinical diagnoses, and previous medical history.
2. **Laboratory Blinding:** Laboratory technicians performing automated hematology analysis and HemoCue testing are blinded to the AI image capture, AI probability, and categorical output.
3. **AI Input Isolation:** The AI inference system receives **strictly fingernail photographs** with zero clinical metadata.
4. **Independent Reference Production:** The clinical reference standard is generated and verified independently in the central laboratory before data linkage.

---

## 8. Multi-Image Aggregation & Image Quality Gate

### 8.1 Image Quality Gate
Every image is automatically evaluated by the quality control module:
* **Optical Sharpness:** Laplacian variance $\ge 5.0$
* **Luminance Exposure:** $40 \le \text{Mean Luminance} \le 230$
* **Specular Glare:** Saturated pixel ratio $\le 20\%$
* **Physiological Nail Bed Chrominance:** YCrCb chrominance verification ($Cr \in [130, 180], Cb \in [75, 135], Cr_{std} \ge 2.5$)

### 8.2 Unit of Analysis & Aggregation Policy
* **Independent Statistical Unit:** **Child / Patient ($N = 300$)**, never individual photographs.
* **Locked Aggregation Rule:**
  $$\bar{P}_{\text{patient}} = \frac{1}{k} \sum_{i=1}^k p_i \quad (k \ge 1 \text{ valid nail images})$$
* **Patient Decision Rule:**
  $$\text{Final State} = \begin{cases}
  \text{ANEMIA}, & \bar{P}_{\text{patient}} \ge 0.48 \\
  \text{NO\_ANEMIA}, & \bar{P}_{\text{patient}} < 0.48 \\
  \text{INCONCLUSIVE}, & k = 0 \text{ (all captures rejected by quality gate)}
  \end{cases}$$

### 8.3 Inconclusive Results Handling Policy
* **Primary Analysis (Evaluable Cohort):** Sensitivity and specificity are computed on participants with $\ge 1$ usable photograph ($k \ge 1$).
* **Secondary Sensitivity Analyses (Intention-to-Diagnose):**
  1. **Worst-Case Screening Analysis:** All `INCONCLUSIVE` cases among confirmed anemic children treated as False Negatives ($FN$), and among confirmed normal children treated as False Positives ($FP$).
  2. **Best-Case / Imputed Analysis:** Multiple imputation based on baseline clinical covariates.
* **Reporting Requirement:** Full STARD flow diagram reporting total enrolled, evaluable, inconclusive, and specific quality gate rejection reasons.

---

## 9. Statistical Analysis Plan (SAP)

### 9.1 Sample Size Justification
* Designed as a **Stratified / Enriched Diagnostic Accuracy Study**:
  - **Required Evaluable Anemic Children ($N_{\text{pos}}$):** $\ge 150$ (ensures lower $95\%$ CI bound $\ge 84.0\%$ for $\hat{S}_e = 90.0\%$)
  - **Required Evaluable Non-Anemic Children ($N_{\text{neg}}$):** $\ge 150$ (ensures lower $95\%$ CI bound $\ge 72.7\%$ for $\hat{S}_p = 80.0\%$)
  - **Evaluable Cohort:** $N \ge 300$ independent children ($1,200$ nail images)
  - **Enrollment Inflation ($10\%$ inconclusive + $4\%$ phlebotomy loss):** **$N = 350$ enrolled participants**.

### 9.2 Confidence Interval Methodology
* All patient-level proportions (Sensitivity, Specificity, PPV, NPV) reported with **exact two-sided $95\%$ Clopper-Pearson Binomial Confidence Intervals**.
* Multiple images from the same child are NEVER treated as independent observations.

### 9.3 Prevalence-Adjusted Predictive Value Reporting
Because the cohort is stratified ($1:1$), raw study PPV/NPV reflect $50\%$ study prevalence. The final report will emphasize prevalence-independent metrics ($\text{LR}^+, \text{LR}^-$, DOR) alongside a Bayes' modeled PPV/NPV table across natural pediatric prevalences ($20\%\text{ to }65\%$).

### 9.4 Pre-Specified Subgroup Analyses
1. **Age Groups:** $6\text{--}23\text{ months}$ vs. $24\text{--}59\text{ months}$.
2. **Anemia Severity:** Mild ($10.0\text{--}10.9\text{ g/dL}$ for 24–59m, $9.5\text{--}10.4\text{ g/dL}$ for 6–23m), Moderate ($7.0\text{--}9.9\text{ g/dL}$), and Severe ($< 7.0\text{ g/dL}$).
3. **Biological Sex:** Male vs. Female.
4. **Smartphone Hardware Tier:** Tier 1 vs. Tier 2 vs. Tier 3.
5. **Study Site:** Tertiary Hospital vs. District Hospital vs. Rural Community Health Post.
6. **Comparator Concordance:** Primary Sysmex Analyzer vs. Secondary HemoCue 301.

---

## 10. Protocol Version History & Sign-Off

| Version | Date | Status | Key Changes |
|:---|:---|:---|:---|
| 1.0 | 2026-09-30 | Draft | Initial prospective multi-center proposal. |
| 2.0 | 2026-09-30 | Revised | Updated to 6–59m primary age, WHO 2024 cutoffs, separated 0–5m cohort, Sysmex vs. HemoCue hierarchy. |
| **2.1** | **2026-09-30** | **Protocol Locked** | **1. Formalized Stratified/Enriched design terminology.**<br>**2. Locked exact Clopper-Pearson precision plan ($N=350$).**<br>**3. Precise blinding language.**<br>**4. Formal Inconclusive handling & discordance policy.**<br>**5. Complete STARD-AI 2025 and Ethics package alignment.** |
