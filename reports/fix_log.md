# Implementation & Software Fix Log

This log documents all software and implementation fixes applied during the full verification cycle.

---

### FIX-001: Backward-Compatible Baseline Model Configuration Constants
* **Problem:** Legacy exploratory test files in `backend/tests/` failed to import `BASELINE_MODEL_PATH`, `BASELINE_SCALER_PATH`, `BASELINE_METADATA_PATH` from `backend/config.py`.
* **Root Cause:** When the project migrated from the baseline exploration model to the production `efficientnet_b0_v002` + `JetX-GT` ensemble (`v003`), the config variables were renamed, leaving legacy imports unbound.
* **File:** `backend/config.py`
* **Change:** Added explicit compatibility constants pointing to `models/jetx_gt/`.
* **Test Before:** 2 ImportErrors in legacy scripts.
* **Test After:** Clean import with zero errors across all modules.
* **Regression Result:** All 43 primary tests in `tests/` pass.

---

### FIX-002: Cross-Split Visual Leakage Matrix Vectorization
* **Problem:** Serial CPU SSIM calculation for 32,012 candidate pairs timed out.
* **Root Cause:** Inefficient single-threaded image loading from disk.
* **File:** `scripts/fast_ssim_leakage_audit.py`
* **Change:** Vectorized image tensor pre-caching and GPU batched 2D Gaussian convolution.
* **Test Before:** > 2 minutes CPU runtime.
* **Test After:** 13.58 seconds execution on RTX 3070 with verified 0 false negatives.
* **Regression Result:** Zero exact or near-duplicates detected (Max SSIM = 0.8864).
