# Clinical Ground-Truth Label Definition & Reference Standards

## 1. Clinical Reference Standards (WHO 2024 Guidelines)

Ground-truth binary classification labels ($y \in \{0, 1\}$) are established strictly by applying validated international hematological reference criteria to quantitative blood hemoglobin ($\text{Hb}$) measurements.

### WHO Hemoglobin Diagnostic Cutoffs (at Sea Level)

$$\text{Anemia Ground Truth } y = \begin{cases} 1 & \text{if } \text{Hb} < \tau_{\text{pop}} \\ 0 & \text{if } \text{Hb} \ge \tau_{\text{pop}} \end{cases}$$

| Population Category | Age / Condition | Cutoff ($\tau_{\text{pop}}$) | Mild Anemia | Moderate Anemia | Severe Anemia |
|---|---|---|---|---|---|
| **Pediatric (Under 5)** | 6 to 59 months | **$< 11.0\text{ g/dL}$** | $10.0 - 10.9\text{ g/dL}$ | $7.0 - 9.9\text{ g/dL}$ | $< 7.0\text{ g/dL}$ |
| **Pediatric (5–11 y)** | 5 to 11 years | **$< 11.5\text{ g/dL}$** | $11.0 - 11.4\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |
| **Adolescent (12–14 y)** | 12 to 14 years | **$< 12.0\text{ g/dL}$** | $11.0 - 11.9\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |
| **Non-Pregnant Women** | $\ge 15$ years | **$< 12.0\text{ g/dL}$** | $11.0 - 11.9\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |
| **Pregnant Women** | Any trimester | **$< 11.0\text{ g/dL}$** | $10.0 - 10.9\text{ g/dL}$ | $7.0 - 9.9\text{ g/dL}$ | $< 7.0\text{ g/dL}$ |
| **Adult Men** | $\ge 15$ years | **$< 13.0\text{ g/dL}$** | $11.0 - 12.9\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |

*Source: World Health Organization (WHO, 2024). Guideline on haemoglobin cutoffs to define anaemia in individuals and populations.*

---

## 2. Target Dataset Reference Mapping

For our locked pediatric training cohort (Ghana Dataset, Asare et al., 2022):
* **Target Cohort:** Children aged $\le 5$ years.
* **Applied Cutoff:** $\text{Hb} < 11.0\text{ g/dL}$.
* **Label Mapping:**
  * `1` = **Anemic** ($\text{Hb} < 11.0\text{ g/dL}$)
  * `0` = **Non-Anemic / Healthy** ($\text{Hb} \ge 11.0\text{ g/dL}$)
* **Preservation Rule:** The dataset's clinical ground-truth labels are mapped from laboratory verification and preserved without heuristic or visual post-hoc re-labeling.

---

## 3. Handling Missing or Ambiguous Demographic Data

* **Strict Non-Inference Principle:** If a dataset record lacks demographic metadata (e.g., missing exact age or sex in an adult record), the pipeline **MUST NOT** guess or default to an arbitrary cutoff.
* **Flagging & Exclusion:** Such records are explicitly flagged in `hb_audit.csv` and excluded from threshold-sensitive clinical ground-truth derivation.

---

## 4. Inference Isolation Guarantee

> [!IMPORTANT]
> The clinical variables ($\text{Hb}$, CBC, age, sex, clinical diagnosis) exist **exclusively as ground truth targets during dataset curation, model training, loss calculation, calibration, and evaluation**. At inference time, the deployed model accepts **strictly nail photographs** and requires zero clinical metadata.
