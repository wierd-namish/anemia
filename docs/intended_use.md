# Medical AI System: Intended Use & Target Condition Specification

## 1. Regulatory & Clinical Purpose Summary

* **Device / Software Name:** Non-Invasive Fingernail Anemia Assessment System (NF-AAS)
* **Intended Purpose:** To assess the presence or absence of anemia in individuals by analyzing digital photographic images of fingernails captured via smartphone or digital camera.
* **Clinical Classification:** Software as a Medical Device (SaMD) / Computer-Aided Screening & Triage Tool.

---

## 2. Input and Output Constraints at Inference Time

### A. Inference Input (Zero Clinical Metadata at Test Time)
At inference and deployment time, the **ONLY** permissible input to the system is:
```
ONE OR MORE DIGITAL FINGERNAIL PHOTOGRAPHS
```

The end user / patient **MUST NOT** be asked or required to enter:
* Hemoglobin (Hb) value or complete blood count (CBC)
* Clinical symptoms (fatigue, pallor, dizziness, dyspnea)
* Age or age group
* Sex / Gender
* Medical history or comorbidities (e.g., chronic kidney disease, sickle cell disease, thalassemia)
* Iron biomarkers (ferritin, serum iron, transferrin saturation)
* Any other laboratory or demographic metadata

The inference model is strictly an **image-only vision system**.

### B. Training / Calibration Ground Truth vs Inference Role
All clinical metadata (Hb concentration, CBC, sex, age, pregnancy status) are utilized **strictly during dataset construction, supervised training, probability calibration, and subgroup validation as reference ground truth labels**. Under no circumstances are clinical variables fed as feature inputs to the deployed inference network.

### C. Inference Output Specification
1. **Diagnostic Decision:** Binary classification:
   * `Anemia-associated pattern detected` (Positive)
   * `No anemia pattern detected` (Negative)
2. **Probability / Score Interpretation:**
   * **Validated Probability ($0.0\% - 100.0\%$):** Displayed only if empirical probability calibration (e.g., Isotonic Regression / Platt Scaling) has been performed and validated on an independent calibration cohort with documented Brier and ECE scores.
   * **Model Screening Score:** If probability calibration is pending or unverified, the numerical value is explicitly labeled as `Model Screening Score (Uncalibrated)`.
3. **Confidence Level:** Categorized as `HIGH`, `MODERATE`, or `LOW` based on the statistical margin between the calibrated score and the clinically tuned decision threshold ($\tau$).
4. **Transparent Human-Readable Explanation:** Explains the visual characteristics (e.g., degree of microvascular bed pallor, redness chromaticity, optical absorption contrast) that drove the prediction.
5. **Standard Clinical Disclaimer:** Directs the user to confirm all screening findings with a laboratory blood test (venous/capillary CBC/Hb).

---

## 3. Medical Target Definition & Reference Standards

### A. Primary Target Condition
* **Target Condition:** Presence of anemia defined by a quantitative blood hemoglobin (Hb) concentration below standard clinical thresholds established by the World Health Organization (WHO) or specific regional guideline panels.

### B. WHO Clinical Reference Standard Cutoffs (Sea Level)
| Population Subgroup | Age / Physiological State | Non-Anemic Hb Threshold | Mild Anemia | Moderate Anemia | Severe Anemia |
|---|---|---|---|---|---|
| **Children** | 6 to 59 months (0.5 – 4.9 y) | $\ge 11.0\text{ g/dL}$ | $10.0 - 10.9\text{ g/dL}$ | $7.0 - 9.9\text{ g/dL}$ | $< 7.0\text{ g/dL}$ |
| **Children** | 5 to 11 years | $\ge 11.5\text{ g/dL}$ | $11.0 - 11.4\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |
| **Children** | 12 to 14 years | $\ge 12.0\text{ g/dL}$ | $11.0 - 11.9\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |
| **Non-Pregnant Adult Women** | $\ge 15$ years | $\ge 12.0\text{ g/dL}$ | $11.0 - 11.9\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |
| **Pregnant Women** | Any gestation age | $\ge 11.0\text{ g/dL}$ | $10.0 - 10.9\text{ g/dL}$ | $7.0 - 9.9\text{ g/dL}$ | $< 7.0\text{ g/dL}$ |
| **Adult Men** | $\ge 15$ years | $\ge 13.0\text{ g/dL}$ | $11.0 - 12.9\text{ g/dL}$ | $8.0 - 10.9\text{ g/dL}$ | $< 8.0\text{ g/dL}$ |

*Source: WHO (2011/2024). Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity.*

### C. Confounding Subtypes & Exclusions
The primary model targets **overall anemia (reduced Hb oxygen-carrying capacity)** as manifested in subungual capillary pallor. The model does **NOT** distinguish underlying etiology (e.g., iron-deficiency vs thalassemia vs aplastic anemia vs anemia of chronic disease). Etio-pathological subtype determination requires secondary clinical workup.

---

## 4. Population Subgroup & Stratification Constraints

Model development and validation require explicit documentation of target cohorts. The model cannot claim generalized diagnostic performance unless evaluated across:
1. **Age Distributions:** Pediatric ($<5$y, $5-14$y) vs Adult ($15-64$y) vs Geriatric ($\ge 65$y).
2. **Skin Phototypes (Fitzpatrick Scale I–VI):** Melanin pigmentation in surrounding periungual tissue must not bias subungual nail-bed analysis.
3. **Biological Sex & Pregnancy Status:** Baseline normal hemoglobin ranges differ between sexes and during gestational hemodilution.
4. **Environmental Factors:** Altitude adjustments ($\Delta\text{Hb}$ elevation corrections) and ambient lighting lux/color temperatures.

---

## 5. Contraindications & Image Exclusion Criteria

The AI system is contraindicated and will refuse to process an image under the following conditions:
* **Nail Surface Coverage:** Artificial nails, opaque or dark nail polish, henna, or heavy cosmetic coatings obscuring the nail bed.
* **Onychopathology / Trauma:** Severe subungual hematoma, severe onychomycosis, active periungual infection (paronychia), or crushing trauma.
* **Image Quality Failure:** Severe optical blur, low resolution ($<128\times 128$), extreme underexposure / overexposure, or severe specular flash glare.
