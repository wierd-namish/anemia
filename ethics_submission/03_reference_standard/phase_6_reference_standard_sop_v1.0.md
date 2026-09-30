# Standard Operating Procedure (SOP): Clinical Hemoglobin Reference Standard and Point-of-Care Comparator (Submission Master)

**Document Identifier:** ETH-SUB-SOP-001  
**Version:** 1.0 (Locked Pre-Study Protocol)  
**Effective Date:** 2026-09-30  
**Study Phase:** Phase 6 Clinical Validation  
**Document Purpose:** Standardized laboratory and point-of-care hemoglobin measurement procedures for the Phase 6 prospective validation study.  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. Reference Standard Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PRIMARY CLINICAL REFERENCE STANDARD                                     │
│ Laboratory Automated Hematology Analyzer (Sysmex XN-350 / XN-550)       │
│ Specimen: Venous Whole Blood (K2-EDTA Anticoagulant)                    │
│ Role: Sole ground truth for Primary Sensitivity & Specificity Endpoints │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ SECONDARY POINT-OF-CARE COMPARATOR                                      │
│ HemoCue Hb 301 Microcuvette System                                      │
│ Specimen: Capillary Whole Blood (Standardized Finger-Prick)             │
│ Role: Secondary comparator to assess POC microcuvette field concordance │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Primary Clinical Reference Standard (Laboratory Automated Analyzer)

### 2.1 Specified Instrument Hardware
* **Primary Analyzer Models:** **Sysmex XN-350 / Sysmex XN-550 Automated Hematology Analyzers** (Sysmex Corporation, Kobe, Japan).
* **Measurement Principle:** Sodium Lauryl Sulfate (SLS) Hemoglobin Method (cyanide-free spectrophotometric absorption measurement at $555\text{ nm}$).
* **Site Allocation:**
  - **Site A (Princess Marie Louise / Korle Bu):** Sysmex XN-550
  - **Site B (Ga West Municipal Hospital):** Sysmex XN-350
  - **Site C (Central Region Health Post):** Cold-chain transit to Ga West central laboratory (Sysmex XN-350)

### 2.2 Specimen Collection & Handling
1. **Specimen Type:** Venous whole blood.
2. **Collection Tube:** Pediatric $K_2\text{EDTA}$ micro-collection tube (BD Microtainer, lavender top).
3. **Minimum Sample Volume:** $0.5\text{ mL}$ (minimum aspiration volume required for analyzer micro-sampling: $25\text{ }\mu\text{L}$).
4. **Phlebotomy Procedure:**
   - Venipuncture performed using a 23G or 25G pediatric butterfly needle.
   - Gently invert the collection tube $8\text{ to }10\text{ times}$ immediately following draw to ensure complete anticoagulant mixing.
   - Do **NOT** shake aggressively to avoid mechanical in vitro hemolysis.
5. **Specimen Transport & Timing:**
   - Specimen stored at ambient temperature ($20\text{--}25^\circ\text{C}$) and processed within **$\le 2\text{ hours}$** of phlebotomy.
   - In rural health post settings (Site C), blood tubes are maintained in temperature-monitored cold boxes ($4\text{--}8^\circ\text{C}$) and transported to the central laboratory within $4\text{ hours}$.

### 2.3 Quality Control & Calibration Process
* **Commercial Quality Control:** Daily three-level commercial control material (`Sysmex e-CHECK`: Low, Normal, High) analyzed every morning prior to clinical study sample processing.
* **Acceptance Criteria:** Control values must fall within $\pm 2\text{ Standard Deviations (SD)}$ of the manufacturer's target mean value ($CV < 1.5\%$).
* **Reagent Lot Verification:** Whenever a new reagent lot is opened, a 3-level calibration verification run is executed and logged.

### 2.4 Handling of Specimen Errors & Instrument Failures
1. **Clotted Specimen:** The tube is visually inspected before aspiration. If macro-clots or micro-clots are identified, the specimen is rejected. An immediate redraw request is issued if the participant is still on-site ($< 30\text{ min}$). If a redraw is impossible, the participant is classified as `MISSING REFERENCE STANDARD` and excluded from primary efficacy analyses.
2. **Severe In Vitro Hemolysis:** If plasma supernate exhibits gross hemolysis ($\text{free Hb} > 2.0\text{ g/dL}$), the sample is flagged and excluded.
3. **Analyzer Hardware Failure:** If the primary Sysmex analyzer malfunctions, specimens are transferred to the designated secondary backup analyzer (Sysmex XP-300 / XN-350) within the same accredited facility, following emergency cross-calibration verification.

### 2.5 Site-to-Site Harmonization Protocol
* **Monthly Inter-Laboratory Split-Sample Cross-Check:** 5 split venous EDTA aliquots are distributed across sites on the first Monday of each recruitment month.
* **Harmonization Threshold:** Inter-site mean bias must be $< 0.3\text{ g/dL}$ with Bland-Altman $95\%$ limits of agreement within $\pm 0.5\text{ g/dL}$.

---

## 3. Secondary Point-of-Care Comparator (HemoCue Hb 301 System)

### 3.1 Specified Instrument Hardware
* **Instrument:** **HemoCue Hb 301 System** (HemoCue AB, Ängelholm, Sweden).
* **Measurement Principle:** Double-wavelength spectrophotometry ($506\text{ nm}$ and $880\text{ nm}$) measuring non-lysed whole blood in single-use microcuvettes.

### 3.2 Capillary Collection Protocol
1. **Patient Site Selection:** Middle or ring finger (palmar surface of distal phalanx, lateral side). Heel prick used for infants $< 12\text{ months}$ if fingertip access is inadequate.
2. **Skin Asepsis & Puncture:** Clean skin with $70\%$ isopropanol swab; allow to dry completely. Puncture using a single-use pediatric safety lancet ($1.5\text{--}1.8\text{ mm}$ depth).
3. **Drop Management:**
   - **Wipe away the first 2 large drops** of blood with sterile gauze to eliminate interstitial fluid contamination.
   - Collect the **third drop** of blood into the microcuvette in one continuous filling motion.
4. **Visual Inspection:** Inspect the microcuvette cavity. If air bubbles or incomplete filling are observed, discard the microcuvette immediately and fill a new one from a fresh drop.
5. **Execution:** Place microcuvette into holder and slide into the analyzer within $< 40\text{ seconds}$ of filling.

### 3.3 Quality Control & Calibration
* **Daily Optical Cleaner Check:** Measure the HemoCue factory calibration optic check cuvette every morning before testing.
* **Weekly Liquid Control:** Analyze low, normal, and high HemoCue liquid controls weekly.
* **Storage:** Microcuvettes stored in tightly sealed original desiccant vials at $10\text{--}40^\circ\text{C}$.

### 3.4 Handling of Discrepancies
* Primary efficacy endpoints (Sensitivity, Specificity, ROC-AUC) are evaluated **strictly against the Primary Reference Standard (Sysmex venous Hb)**.
* Any numerical discrepancy between the Sysmex venous result and HemoCue capillary reading will be analyzed as a pre-specified secondary endpoint (Bland-Altman bias, capillary-vs-venous regression, microcuvette variance).
