# Research Dataset Inventory: Fingernail Anemia Assessment

This inventory documents legitimate, publicly available research datasets featuring digital photographs of fingernails paired with clinical hemoglobin (Hb) measurements or clinically established anemia labels.

---

## 1. Ghana Fingernail Anemia Dataset (Asare et al., 2022)

* **Dataset Name:** Detection of Anemia using Colour of the Fingernails Image Datasets from Ghana
* **Source / DOI:** [Mendeley Data, V1, doi:10.17632/2xx4j3kjg2.1](https://data.mendeley.com/datasets/2xx4j3kjg2/1)
* **Citation:** Asare, Justice Williams; Appiahene, Peter; Donkoh, Emmanuel Timmy (2022). "Detection of Anemia using Colour of the Fingernails Image Datasets from Ghana", Mendeley Data, V1.
* **License / Access Requirements:** Creative Commons Attribution 4.0 International (CC BY 4.0) — Permitted for academic and commercial research with citation.
* **Country / Origin:** Ghana (Sunyani Municipal Hospital & regional clinics).
* **Population:** Pediatric cohort aged five years and below ($0 - 5$ years old).
* **Demographic Representation:** West African / Ghanaian population (Fitzpatrick skin phototypes V and VI).
* **Sample Size:**
  * **Patient Count:** $552$ unique pediatric patients.
  * **Image Count:** $4,260$ total fingernail photographs ($\sim 8$ images per patient across different fingers).
* **Sex Distribution:** Both male and female pediatric patients represented.
* **Clinical Ground Truth:** Blood samples analyzed by laboratory technicians to establish clinical anemia status.
* **Anemia Label Definition:** Binary label (`1 = Anemic`, `0 = Non-anemic`) based on hospital laboratory reference cutoff ($\text{Hb} < 11.0\text{ g/dL}$ for children $<5\text{y}$).
* **Image Acquisition Setup:**
  * Camera resolution $\ge 12\text{ MP}$.
  * Camera flash/spotlight explicitly turned off to eliminate specular glare reflections.
  * Laboratory officers held the pediatric fingers in standard extension.
* **Nail ROI / Bounding Boxes:** Raw images contain finger with background; ROI extraction via threshold triangle segmentation.
* **Patient ID Availability:** Available (`patient_id` prefixes such as `Anemic-Fin-013`, `Non-Anemic-Fin-045` enabling patient-level splitting).

---

## 2. Dataset of Human Skin and Fingernails Images (Yakimov et al., 2024)

* **Dataset Name:** Dataset of human skin and fingernails images for non-invasive haemoglobin level assessment
* **Source / DOI:** [Figshare / Nature Scientific Data, doi:10.6084/m9.figshare.25867432](https://doi.org/10.6084/m9.figshare.25867432)
* **License / Access Requirements:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Country / Origin:** Multicenter clinical study.
* **Population:** Adults ($18 - 75$ years old) presenting for routine or diagnostic blood work.
* **Sample Size:**
  * **Patient Count:** $250$ patients.
  * **Image Count:** $2,000+$ images across multiple fingers and dorsal skin regions.
* **Demographic Metadata:** Gender, exact age, and Fitzpatrick skin tone category recorded.
* **Clinical Ground Truth:** Quantitative venous blood complete blood count (CBC) with laboratory-measured $\text{Hb}$ in $\text{g/dL}$ (range: $6.2 - 17.8\text{ g/dL}$).
* **Anemia Label Definition:** Continuous $\text{Hb}$ values provided; allows flexible thresholding according to WHO criteria ($<12.0\text{ g/dL}$ for females, $<13.0\text{ g/dL}$ for males).
* **Image Acquisition Setup:** High-resolution smartphone cameras with standardized distance and lighting calibration targets.
* **Nail ROI / Bounding Boxes:** Included (annotated bounding box coordinates for nail bed and dorsal skin patches).
* **Patient ID Availability:** Full anonymous `patient_id` available.

---

## 3. Mannino / Emory Smartphone Fingernail Anemia Cohort (Mannino et al., 2018)

* **Dataset Name:** Smartphone App for Non-Invasive Detection of Anemia via Fingernail Bed Pallor
* **Source / DOI:** [Nature Communications 9, 4924 (2018), doi:10.1038/s41467-018-07262-4](https://doi.org/10.1038/s41467-018-07262-4)
* **Citation:** Mannino, R. G., Myers, D. R., Tyburski, E. A., et al. (2018). "Smartphone app for non-invasive detection of anemia using fingernail photos". *Nature Communications*.
* **License / Access Requirements:** Research access upon institutional application / dataset terms.
* **Country / Origin:** United States (Children's Healthcare of Atlanta / Emory University Hospital).
* **Population:** Children and adult outpatients ($N = 337$ subjects, including individuals with sickle cell disease and healthy controls).
* **Sample Size:** $337$ subjects.
* **Clinical Ground Truth:** Complete blood count (CBC) hemoglobin measured via automated hematology analyzer (Sysmex / Beckman Coulter) within 1 hour of photo capture.
* **Key Findings:** Demonstrated that optical nail bed color features (R, G, B chromaticity after color calibration) correlate strongly with blood $\text{Hb}$ ($r = 0.82$).

---

## 4. Summary Matrix: Dataset Comparison for Model Training & Validation

| Parameter | Ghana Dataset (Asare 2022) | Figshare Dataset (Yakimov 2024) | Emory Cohort (Mannino 2018) |
|---|---|---|---|
| **Primary Population** | Pediatric ($<5\text{y}$) | Adults ($18-75\text{y}$) | Pediatric + Adult |
| **Skin Phototypes** | Fitzpatrick V–VI | Fitzpatrick I–IV | Fitzpatrick I–VI |
| **Patient Count** | $552$ | $250$ | $337$ |
| **Image Count** | $4,260$ | $2,000+$ | $1,500+$ |
| **Hb Value Provided** | Binary Label ($\text{Hb}<11$) | Quantitative $\text{Hb}$ ($\text{g/dL}$) | Quantitative $\text{Hb}$ ($\text{g/dL}$) |
| **Bounding Boxes** | Computed via ROI / YOLO | Pre-annotated XML/JSON | App-guided ROI |
| **Patient-Level Split** | Yes (explicit `patient_id`) | Yes (`patient_id`) | Yes (`patient_id`) |
| **Target Role** | Training & Internal Test | Multi-center External Validation | Reference Clinical Benchmark |
