# Phase 6 Sample Size & Statistical Precision Justification (Submission Master)

**Document Identifier:** ETH-SUB-SAP-002  
**Version:** 1.0 (Locked Pre-Study Protocol)  
**Date:** 2026-09-30  
**Design Type:** **Stratified / Enriched Prospective Diagnostic Accuracy Design**  
**Target Population:** Children aged 6 to 59 months  
**Operating Threshold:** $\tau = 0.48$  
**Primary Reference Standard:** Sysmex XN-350 / Sysmex XN-550 Automated Hematology Analyzers (Venous EDTA Whole Blood)  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Study Design Framework: Stratified Enrichment

In a natural unselected primary care setting, the point prevalence of pediatric anemia can vary widely ($30\%\text{ to }70\%$) depending on local seasonality, malaria transmission intensity, and nutritional factors. 

To ensure statistical adequacy for **both** primary diagnostic sensitivity and co-primary specificity without inflating the total sample size unnecessarily, the Phase 6 study adopts a **Stratified / Enriched Diagnostic Accuracy Design**:
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

$$N_{\text{enrolled}} = \frac{N_{\text{evaluable}}}{(1 - r_{\text{inconcl}}) \times (1 - r_{\text{miss\_ref}})} = \frac{300}{(1 - 0.10) \times (1 - 0.04)} = \frac{300}{0.864} = 347.2 \implies \mathbf{350\text{ Enrolled Participants}}$$

* **Target Enrolled Cohort:** **$N = 350$ participants**
* **Expected Inconclusive Rate ($10\%$):** $\approx 35$ participants
* **Expected Phlebotomy Failure ($4\%$):** $\approx 15$ participants
* **Net Evaluable Paired Cohort:** **$N \ge 300$ children** ($150$ positive, $150$ negative)
* **Total Nail Image Archive ($4$ images/child):** **$1,400$ standardized nail photographs**
