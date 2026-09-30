# Statistical Analysis Plan (SAP) — Phase 6 Prospective Clinical Validation

**Document Title:** Pre-Specified Statistical Analysis Plan for the Phase 6 Clinical Validation Study  
**Document Identifier:** ETH-SUB-SAP-001  
**Version:** 2.1 (Locked Pre-Study Statistical Plan)  
**Date of Lock:** 2026-09-30  
**Study Phase:** Phase 6 Multi-Center Prospective Clinical Validation  
**Target Population:** Children aged 6 to 59 months presenting in Ghanaian primary/triage settings  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Study Objectives & Primary Estimands

### 1.1 Primary Objective
To prospectively evaluate the diagnostic sensitivity of the locked `EfficientNet-B0` algorithm in classifying anemia against a contemporaneous central laboratory automated hematology reference standard in children aged **6 to 59 months**.

### 1.2 Co-Primary Objective
To prospectively evaluate the diagnostic specificity of the locked `EfficientNet-B0` algorithm at the pre-specified locked threshold ($\tau = 0.48$).

---

## 2. Analysis Populations

1. **Enrolled Population ($N = 350$):** All pediatric participants whose parents/guardians provided written informed consent.
2. **Evaluable Paired Cohort (Primary Analysis Population, $N \ge 300$):** All enrolled participants with:
   - At least one usable fingernail photograph meeting Image Quality Gate standards ($k \ge 1$), AND
   - A valid contemporaneous Primary Reference Standard venous blood hemoglobin result ($< 30\text{ min}$ interval).
3. **Intention-to-Diagnose (ITD) Population ($N = 350$):** All enrolled participants, incorporating all `INCONCLUSIVE` cases into pre-specified worst-case sensitivity bounding analyses.
4. **Exploratory Infant Cohort ($N \approx 50$):** Children aged 0–5 months evaluated separately under local reference intervals, with zero contribution to primary regulatory efficacy endpoints.

---

## 3. Unit of Analysis & Aggregation Algorithm

* **Independent Statistical Unit:** **Child / Patient**.
* Individual fingernail images from the same child ($3\text{ to }4$ images) are NEVER treated as independent statistical samples.
* **Locked Image Preprocessing & Forward Pipeline:**
  $$\text{Nail Photo} \to \text{Quality Gate} \to \text{ROI Contour} \to 224\times 224\text{ RGB} \to \text{EfficientNet-B0} \to \text{Logit} \to \text{Isotonic Calibrator} \to p_i$$
* **Patient-Level Probability Aggregation:**
  $$\bar{P}_{\text{patient}} = \frac{1}{k} \sum_{i=1}^k p_i \quad (k \ge 1)$$
* **Decision Rule at Locked Threshold ($\tau = 0.48$):**
  $$\hat{Y}_{\text{patient}} = \begin{cases}
  \text{ANEMIA (1)}, & \bar{P}_{\text{patient}} \ge 0.48 \\
  \text{NO\_ANEMIA (0)}, & \bar{P}_{\text{patient}} < 0.48 \\
  \text{INCONCLUSIVE}, & k = 0 \text{ (all photographs rejected)}
  \end{cases}$$

---

## 4. Ground-Truth Anemia Labeling (WHO 2024 Guidelines)

$$\text{Reference Anemia Ground Truth } (Y_i) = \begin{cases}
1 \text{ (Anemia)}, & \text{if Age } 6\text{--}23\text{m and Venous Hb} < 10.5\text{ g/dL} \ (105\text{ g/L}) \\
1 \text{ (Anemia)}, & \text{if Age } 24\text{--}59\text{m and Venous Hb} < 11.0\text{ g/dL} \ (110\text{ g/L}) \\
0 \text{ (No Anemia)}, & \text{otherwise}
\end{cases}$$

---

## 5. Statistical Estimation Methods & Confidence Intervals

### 5.1 Primary Estimators
* **Sensitivity ($S_e$):** $\frac{TP}{TP + FN} = \frac{\sum I(\hat{Y}_i = 1 \text{ and } Y_i = 1)}{\sum I(Y_i = 1)}$
* **Specificity ($S_p$):** $\frac{TN}{TN + FP} = \frac{\sum I(\hat{Y}_i = 0 \text{ and } Y_i = 0)}{\sum I(Y_i = 0)}$

### 5.2 Exact Binomial Confidence Intervals (Clopper-Pearson)
Because normal asymptotic approximations fail near boundary conditions and bootstrap intervals can degenerate in low-error partitions, all patient-level proportions ($S_e, S_p, \text{PPV}, \text{NPV}$) will be calculated using exact two-sided $95\%$ Clopper-Pearson Binomial Confidence Intervals:

$$\text{CI}_{95\%} = \left[ B\left(0.025; x, n - x + 1\right), B\left(0.975; x + 1, n - x\right) \right]$$

### 5.3 Secondary Estimators
* **Positive Likelihood Ratio ($\text{LR}^+$):** $\frac{S_e}{1 - S_p}$ with log-method $95\%$ CI.
* **Negative Likelihood Ratio ($\text{LR}^-$):** $\frac{1 - S_e}{S_p}$ with log-method $95\%$ CI.
* **Diagnostic Odds Ratio (DOR):** $\frac{\text{LR}^+}{\text{LR}^-}$.
* **Receiver Operating Characteristic (ROC-AUC):** Empirical trapezoidal rule with non-parametric DeLong $95\%$ CI.
* **Expected Calibration Error (ECE) & Brier Score:** Evaluated with 10 uniform probability bins on calibrated probabilities.

---

## 6. Sample Size Justification & Stratified Enrichment

* **Design Type:** **Stratified / Enriched Diagnostic Accuracy Design**
* **Target Positive Cohort ($N_{\text{pos}}$):** $\ge 150$ evaluable anemic children ($S_e \ge 90.0\%$, guarantees lower $95\%$ CI bound $\ge 84.0\%$).
* **Target Negative Cohort ($N_{\text{neg}}$):** $\ge 150$ evaluable non-anemic children ($S_p \ge 80.0\%$, guarantees lower $95\%$ CI bound $\ge 72.7\%$).
* **Inflation Adjustment ($10\%$ inconclusive + $4\%$ missing reference):** Total enrollment target = **$N = 350$ participants**.

---

## 7. Prevalence-Adjusted Predictive Value Modeling

Because the study is stratified ($1:1$, $\text{study prevalence} \approx 50.0\%$), raw sample PPV/NPV do not reflect natural clinical prevalence. The final report will include a Bayes' theorem modeled predictive value table across realistic pediatric clinic prevalences ($\pi \in [20\%, 65\%]$):

$$\text{PPV}(\pi) = \frac{S_e \cdot \pi}{S_e \cdot \pi + (1 - S_p) \cdot (1 - \pi)}, \quad \text{NPV}(\pi) = \frac{S_p \cdot (1 - \pi)}{S_p \cdot (1 - \pi) + (1 - S_e) \cdot \pi}$$

---

## 8. Handling of Inconclusive Results & Missing Data

1. **Primary Reporting:** STARD flow diagram reporting Total Enrolled, Evaluable, Inconclusive, and Missing Phlebotomy.
2. **Worst-Case Sensitivity Analysis:**
   - Inconclusive cases in reference-positive children treated as $FN$.
   - Inconclusive cases in reference-negative children treated as $FP$.
3. **Missing Phlebotomy Reference:** Excluded from primary efficacy analysis; evaluated in sensitivity dropout analysis.

---

## 9. Pre-Specified Subgroup Analyses

1. **Age Cohorts:** $6\text{--}23\text{ months}$ vs. $24\text{--}59\text{ months}$.
2. **Anemia Severity:** Mild ($10.0\text{--}10.9\text{ g/dL}$ for 24–59m, $9.5\text{--}10.4\text{ g/dL}$ for 6–23m), Moderate ($7.0\text{--}9.9\text{ g/dL}$), and Severe ($< 7.0\text{ g/dL}$).
3. **Biological Sex:** Male vs. Female.
4. **Smartphone Hardware Tier:** Tier 1 (Samsung) vs. Tier 2 (Redmi/Tecno) vs. Tier 3 (iPhone).
5. **Study Site:** Tertiary (Korle Bu / PML) vs. District (Ga West) vs. Rural (Central Region).
6. **Analyzer Concordance:** Primary Sysmex Automated Analyzer vs. Secondary HemoCue Hb 301.
