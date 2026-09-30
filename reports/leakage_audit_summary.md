# EXP-01 Cross-Split Visual Leakage Audit

**Audit Execution Timestamp:** 2026-09-30 03:51:13  
**Total Wall-Clock Elapsed:** 13.58 seconds  
**Investigational Scope:** Complete verification of patient-level isolation, exact SHA-256 duplicates, 64-bit DCT perceptual hash screening, and Structural Similarity (SSIM) across all cross-split partitions.

---

## 1. Dataset & Manifest Verification
* **Manifest File:** `data/final_manifest.csv`
* **Total Manifest Records:** 4,260
* **Valid Image Files Resolved:** 4,260 (100.0% resolved, 0 missing, 0 unreadable)
* **Total Unique Patients:** 554
* **Partition Allocation:**
  - `train`: 2,993 images across 387 patients
  - `validation`: 416 images across 55 patients
  - `calibration`: 423 images across 55 patients
  - `test`: 428 images across 57 patients

---

## 2. Patient Overlap Audit
Every pairwise set intersection across the four partitions was computed:

| Partition Pair | Intersection Count | Status |
| :--- | :--- | :--- |
| **TRAIN ∩ VALIDATION** | 0 patients | **EMPTY SET ($\emptyset$)** |
| **TRAIN ∩ CALIBRATION** | 0 patients | **EMPTY SET ($\emptyset$)** |
| **TRAIN ∩ TEST** | 0 patients | **EMPTY SET ($\emptyset$)** |
| **VALIDATION ∩ CALIBRATION** | 0 patients | **EMPTY SET ($\emptyset$)** |
| **VALIDATION ∩ TEST** | 0 patients | **EMPTY SET ($\emptyset$)** |
| **CALIBRATION ∩ TEST** | 0 patients | **EMPTY SET ($\emptyset$)** |

* **Finding:** Zero patient ID overlap exists across any partition.

---

## 3. Exact Duplicate Results (SHA-256)
* **Total Unique SHA-256 Hashes:** 4,260 / 4,260
* **Cross-Split Exact Duplicates (LEAKAGE):** **0 (Zero)**
* **Finding:** No identical image files appear across different partitions.

---

## 4. pHash Results (64-bit DCT Hamming Distance)
* **Algorithm:** 64-bit Frequency-Domain DCT Perceptual Hashing (Median Binarization)
* **Candidate Pairs with Hamming Distance $\le 3$:** 32,012 pairs
* **Candidate CSV Output:** [`reports/audit_leakage_phash_candidates.csv`](file:///c:/Users/Asus/MYPASS/reports/audit_leakage_phash_candidates.csv)

---

## 5. SSIM Results (Structural Similarity)
* **Confirmation Threshold:** $\text{SSIM} \ge 0.95$
* **Total Candidates Evaluated on GPU:** 32,012 pairs
* **Maximum Observed Cross-Split SSIM:** 0.8864
* **Confirmed Near-Duplicate Pairs ($\text{SSIM} \ge 0.95$):** 0
* **Results CSV Output:** [`reports/audit_leakage_phash_ssim_results.csv`](file:///c:/Users/Asus/MYPASS/reports/audit_leakage_phash_ssim_results.csv)

---

## 6. Suspicious Pairs & Visual Examples
* **Total Confirmed Suspicious Pairs:** 0
* **Side-by-Side Artifacts Saved in:** `reports/leakage_examples/` (0 generated).

---

## 7. Burst-Capture & Intra-Patient Patterns
* Intra-patient captures (multiple fingers from the same subject) remain 100% confined to their assigned split.
* No burst captures or repeated angles crossed train, validation, calibration, or test boundaries.

---

## 8. Processing Errors
* **Missing Files:** 0
* **Corrupt Images:** 0
* **Unreadable Files:** 0
* **Set Intersection Failures:** 0

---

## 9. Final Classification

```
====================================================================================================
                             NO CROSS-SPLIT DUPLICATES DETECTED
====================================================================================================
```

> **Defensible Scientific Conclusion:**  
> No detectable cross-split exact or near-duplicate pairs were identified under the stated thresholds ($\text{pHash distance} \le 3$, $\text{SSIM} \ge 0.95$) across all candidate combinations. Patient-level split boundaries are strictly maintained.
