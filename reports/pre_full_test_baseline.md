# Pre-Full Test Baseline Snapshot

**Execution Timestamp:** 2026-09-30 03:52:00  
**Project:** Fingernail-Based Anemia Screening Decision-Support Prototype  
**Auditor:** Lead Implementation & Verification Engineer  

---

## 1. System & Environment Snapshot
* **Git Repository:** Working tree directory (unversioned local root)
* **Python Version:** `3.13.7 (AMD64)`
* **PyTorch Version:** `2.6.0+cu124`
* **Torchvision Version:** `0.21.0+cu124`
* **CUDA Driver / Runtime:** `CUDA 12.4`
* **GPU Hardware:** `NVIDIA GeForce RTX 3070 Laptop GPU (8.59 GB VRAM)`
* **OS:** Windows 11 (AMD64)

---

## 2. Core Model Artifacts & Configurations
| Component | Path | Size | SHA-256 |
| :--- | :--- | :--- | :--- |
| **Primary CNN** | `experiments/efficientnet_b0_v002/best_model.pth` | 16,334,822 bytes | `a5c0ad6ca54f0d954dc4aad008f79bb3090f47e7059521ec34809cbb272f6f56` |
| **Secondary MLP** | `models/jetx_gt/mlp_model.joblib` | 155,172 bytes | `d81d660e46fadc642e31a90edd8649e6a07d169af2591da21570ac6d1d6b9c94` |
| **Feature Scaler** | `models/jetx_gt/feature_scaler.joblib` | 1,287 bytes | `2089f87388ab2ab40e59985ec54053d3bbf77be220ea20cc5352db8974e2c79d` |
| **Logistic Fusion** | `configs/ensemble_fusion_v003.joblib` | 895 bytes | `6642b82274d5a3a0d0bd5d97b3d5ac43960eec593cca4efb303d86f947b1b272` |
| **Calibrator** | `configs/calibrator_isotonic_v003.joblib` | 742 bytes | `823c91e4bf78422dae139c265f299ee86830f886a0da7e487d8f418c00130ebc` |
| **Locked Threshold**| `configs/locked_tau_v003.json` | 613 bytes | `a71e15e05c882cd1fdbe9006a01bde314da0d0c668f81fac40acd0ad16b8a9bf` |

---

## 3. Dataset Baseline
* **Manifest File:** `data/final_manifest.csv`
* **Total Image Records:** 4,260
* **Unique Patients:** 554
* **Partition Split Counts:**
  - `train`: 2,993 images (387 patients)
  - `validation`: 416 images (55 patients)
  - `calibration`: 423 images (55 patients)
  - `test`: 428 images (57 patients)

---

## 4. Test Suite Baseline
* **Unit Test Suite:** `tests/`
* **Baseline Test Count:** 43 tests
* **Baseline Passing Status:** 43 Passed / 0 Failed / 0 Skipped
