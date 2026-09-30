# Phase 6 Sample Size & Statistical Precision Justification

**Document Title:** Prospective Diagnostic Accuracy Sample Size Justification  
**Study Phase:** Phase 6 Clinical Validation  
**Target Population:** Children aged 6 to 59 months  
**Design Type:** **Stratified / Enriched Prospective Diagnostic Accuracy Design**  
**Operating Threshold:** $\tau = 0.48$ (Locked on Isotonic Calibrated Probabilities)  
**Primary Reference Standard:** Laboratory Automated Hematology Analyzer (Venous EDTA Whole Blood)  
**Date:** 2026-09-30  
**Version:** 1.0 (Locked Pre-Study Statistical Plan)  

---

## 1. Study Design Framework: Stratified Enrichment

In a natural unselected primary care setting, the point prevalence of pediatric anemia can vary widely ($30\%\text{ to }70\%$) depending on local seasonality, malaria transmission intensity, and nutritional factors. 

To ensure statistical adequacy for **both** primary diagnostic sensitivity and co-primary specificity without inflating the total sample size unnecessarily, the Phase 6 study adopts a **Stratified / Enriched Diagnostic Accuracy Design**. Under this design:
1. Clinical sites enroll consecutive eligible children aged 6–59 months presenting for pediatric evaluation.
2. Enrollment is stratified to ensure approximately equal representation across reference ground-truth categories:
   * **Target Anemia-Positive Cohort ($N_{\text{pos}}$):** $\ge 150$ evaluable participants
   * **Target Non-Anemic Cohort ($N_{\text{neg}}$):** $\ge 150$ evaluable participants
3. Total evaluable participant target: **$N_{\text{evaluable}} = 300$ independent children**.

---

## 2. Statistical Precision & Exact Binomial Calculations

Sample size calculations are based on exact binomial confidence interval width (Clopper-Pearson method) and pre-specified lower confidence interval bounds, rather than asymptotic normal approximations.

### 2.1 Primary Endpoint: Diagnostic Sensitivity
* **Target Sensitivity ($\hat{S}_e$):** $90.0\%$ ($0.90$)
* **Minimum Acceptable Lower Bound of 2-Sided $95\%$ CI ($L_{95\%}$):** $\ge 82.0\%$ ($0.82$)
* **Significance Level ($\alpha$):** $0.05$ (two-sided)
* **Statistical Power ($1 - \beta$):** $\ge 85\%$

The exact two-sided $100(1-\alpha)\%$ Clopper-Pearson confidence interval for $x$ successes in $n$ trials is defined by:

$$\text{Lower Bound} = B\left(\frac{\alpha}{2}; x, n - x + 1\right) = \frac{1}{1 + \frac{n - x + 1}{x F_{1-\alpha/2}(2(n-x+1), 2x)}}$$

$$\text{Upper Bound} = B\left(1 - \frac{\alpha}{2}; x + 1, n - x\right) = \frac{\frac{x + 1}{n - x} F_{1-\alpha/2}(2(x+1), 2(n-x))}{1 + \frac{x + 1}{n - x} F_{1-\alpha/2}(2(x+1), 2(n-x))}$$

Where $B(p; a, b)$ is the beta distribution quantile and $F$ is the Snedecor-Fisher F-distribution quantile.

#### Sensitivity Precision Table for $\hat{S}_e = 90.0\%$:
| Anemic Sample Size ($N_{\text{pos}}$) | Expected True Positives ($TP$) | Exact $95\%$ Clopper-Pearson CI | Lower Bound Margin ($\Delta$) |
|:---|:---|:---|:---|
| $n = 100$ | $90$ | $[82.38\%, 95.10\%]$ | $+7.62\%$ |
| $n = 125$ | $113$ | $[83.94\%, 94.99\%]$ | $+6.06\%$ |
| **$n = 150$ (Selected)** | **$135$** | **$[84.00\%, 94.30\%]$** | **$+6.00\%$ (Exceeds $\ge 82.0\%$ target)** |
| $n = 175$ | $158$ | $[84.81\%, 94.09\%]$ | $+5.19\%$ |

$\implies$ A sample size of **$N_{\text{pos}} = 150$ evaluable anemic children** guarantees that if the true sensitivity is $90\%$, the lower $95\%$ confidence limit will be $84.0\%$, comfortably exceeding the pre-specified regulatory hurdle of $82.0\%$.

---

### 2.2 Co-Primary Endpoint: Diagnostic Specificity
* **Target Specificity ($\hat{S}_p$):** $80.0\%$ ($0.80$)
* **Minimum Acceptable Lower Bound of 2-Sided $95\%$ CI ($L_{95\%}$):** $\ge 70.0\%$ ($0.70$)
* **Significance Level ($\alpha$):** $0.05$ (two-sided)

#### Specificity Precision Table for $\hat{S}_p = 80.0\%$:
| Non-Anemic Sample Size ($N_{\text{neg}}$) | Expected True Negatives ($TN$) | Exact $95\%$ Clopper-Pearson CI | Lower Bound Margin ($\Delta$) |
|:---|:---|:---|:---|
| $n = 100$ | $80$ | $[70.82\%, 87.33\%]$ | $+9.18\%$ |
| $n = 125$ | $100$ | $[71.95\%, 86.64\%]$ | $+8.05\%$ |
| **$n = 150$ (Selected)** | **$120$** | **$[72.70\%, 86.07\%]$** | **$+7.30\%$ (Exceeds $\ge 70.0\%$ target)** |
| $n = 175$ | $140$ | $[73.37\%, 85.57\%]$ | $+6.63\%$ |

$\implies$ A sample size of **$N_{\text{neg}} = 150$ evaluable non-anemic children** guarantees that if the true specificity is $80\%$, the lower $95\%$ confidence limit will be $72.7\%$, exceeding the pre-specified hurdle of $70.0\%$.

---

## 3. Enrollment Inflation Accounting for Rejections & Missing Data

To achieve $N_{\text{evaluable}} = 300$ fully evaluable participants with complete paired index test (AI nail image) and reference standard (automated CBC hemoglobin), enrollment must be inflated to account for two distinct potential loss mechanisms:

1. **AI Image Quality / Inconclusive Rejection Rate ($r_{\text{inconcl}}$):** Estimated at **$10.0\%$** based on Phase 4/5 pediatric field testing (severe motion blur, off-center placement, unremovable polish).
2. **Missing Reference Blood Standard / Phlebotomy Failure ($r_{\text{miss\_ref}}$):** Estimated at **$4.0\%$** (difficult pediatric venous access, clotted microtainer specimen).

### Inflation Formula:

$$N_{\text{enrolled}} = \frac{N_{\text{evaluable}}}{(1 - r_{\text{inconcl}}) \times (1 - r_{\text{miss\_ref}})}$$

$$N_{\text{enrolled}} = \frac{300}{(1 - 0.10) \times (1 - 0.04)} = \frac{300}{0.90 \times 0.96} = \frac{300}{0.864} = 347.2 \implies \mathbf{350\text{ Enrolled Participants}}$$

### Summary Breakdown:
* **Total Target Enrolled:** **$N = 350$ children**
* **Expected Inconclusive / Quality Rejection ($10\%$):** $\approx 35$ participants
* **Expected Missing Reference / Clotted Tube ($4\%$):** $\approx 15$ participants
* **Net Evaluable Paired Cohort:** **$N \ge 300$ participants** ($150$ anemic, $150$ non-anemic)
* **Total Fingernail Image Library ($4$ images/child):** **$1,400$ standardized nail photographs**

---

## 4. Clinical Prevalence & Predictive Value Interpretation

> [!WARNING]
> **Prevalence Bias Warning in Enriched Diagnostic Studies:**  
> Because the cohort is intentionally stratified/enriched to achieve approximately equal numbers of anemic and non-anemic children ($\text{study prevalence} \approx 50.0\%$), the raw unadjusted Positive Predictive Value (PPV) and Negative Predictive Value (NPV) observed in this study will reflect the $50\%$ study design prevalence, rather than the variable natural prevalence in primary care clinics.

### Pre-Specified Reporting Strategy:
1. **Primary Reporting:** The clinical validation report will primarily report prevalence-independent diagnostic metrics:
   * **Sensitivity ($S_e$)**
   * **Specificity ($S_p$)**
   * **Positive Likelihood Ratio ($\text{LR}^+$):** $\frac{S_e}{1 - S_p}$
   * **Negative Likelihood Ratio ($\text{LR}^-$):** $\frac{1 - S_e}{S_p}$
   * **Diagnostic Odds Ratio (DOR):** $\frac{\text{LR}^+}{\text{LR}^-}$

2. **Prevalence-Adjusted Predictive Value Table:**  
   Bayes' Theorem will be applied across a clinically realistic range of pediatric anemia prevalences ($\pi \in [20\%, 70\%]$):

$$\text{PPV}(\pi) = \frac{S_e \cdot \pi}{S_e \cdot \pi + (1 - S_p) \cdot (1 - \pi)}$$

$$\text{NPV}(\pi) = \frac{S_p \cdot (1 - \pi)}{S_p \cdot (1 - \pi) + (1 - S_e) \cdot \pi}$$

#### Modeled Predictive Values for Point Estimates ($S_e = 90.0\%, S_p = 80.0\%$):
| Clinical Setting | Setting Description | Assumed Prevalence ($\pi$) | Modeled PPV | Modeled NPV |
|:---|:---|:---|:---|:---|
| **Low-Risk Screening** | Well-child clinic / urban routine check | $20.0\%$ | **$52.9\%$** | **$96.9\%$** |
| **Moderate-Risk Triage** | General pediatric outpatient | $35.0\%$ | **$70.8\%$** | **$93.8\%$** |
| **Study Enriched Cohort** | Stratified study sample ($1:1$) | **$50.0\%$** | **$81.8\%$** | **$88.9\%$** |
| **High-Risk Triage** | High malaria endemic / malnutrition post | $65.0\%$ | **$89.3\%$** | **$81.1\%$** |

---

## 5. Statistical Freeze Verification

* **Unit of Analysis:** Independent child ($N = 300$), never pooled individual photographs.
* **Locked Aggregation Rule:** Arithmetic mean calibrated probability across valid photographed digits.
* **Frozen Operating Threshold:** $\tau = 0.48$.
* **Confidence Interval Method:** Exact Clopper-Pearson binomial confidence intervals.
