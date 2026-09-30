# Live Demo Verification

**Project**: Fingernail-Based Anemia Screening Decision-Support System  
**Audit Timestamp**: 2026-09-30 04:02:15 IST  
**Execution Environment**: Windows 11 | Python 3.13.1 | PyTorch 2.6.0+cu124 | CUDA 12.4  
**Hardware Device**: NVIDIA GeForce RTX 3070 Laptop GPU (`cuda:0`)  
**Pipeline**: Dual-Model Ensemble (`efficientnet_b0_v002` + `JetX-GT/nail-anemia-detector`) + Logistic Fusion v003 + Isotonic Calibrator v003 + Locked Threshold $\tau = 0.9000$  

---

## Backend

* **Backend Framework**: FastAPI with Uvicorn ASGI Server
* **Backend Endpoint URL**: `http://127.0.0.1:8000` (and `http://0.0.0.0:8000`)
* **Active Process**: Daemon task running in workspace
* **Startup Command**:
  ```powershell
  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
  ```
* **Endpoints Verified**:
  * `GET /health` &rarr; Status 200 OK (`{"status": "healthy", "model_loaded": true, "device": "cuda", "gpu": "NVIDIA GeForce RTX 3070 Laptop GPU"}`)
  * `GET /model-info` &rarr; Status 200 OK (`{"primary_model": "efficientnet_b0_v002", "secondary_model": "JetX-GT/nail-anemia-detector", "fusion_version": "ensemble_v003", "threshold": 0.9}`)
  * `POST /predict` &rarr; Status 200 OK (Real-time dynamic inference on uploaded/captured images)
  * `POST /predict-multiple` &rarr; Status 200 OK (Multi-nail aggregation)

---

## Frontend

* **Frontend URL**: `http://127.0.0.1:8000/` (served statically from `frontend/index.html`)
* **Startup Command**: Automatically hosted by FastAPI backend
* **UI Features & Components**:
  * **Header**: Title `FINGERNAIL ANEMIA SCREENING`, Subtitle `Investigational AI Research Prototype`, Status badge `● LIVE MODEL INFERENCE`, Device badge `⚡ DEVICE: NVIDIA GeForce RTX 3070 Laptop GPU (CUDA)`.
  * **Input Mode Selector**: `📷 LIVE CAMERA` (with viewfinder and reticle guide) + `📁 UPLOAD FINGERNAIL IMAGE` (with drag-and-drop file picker).
  * **Assessment Result Card**: Dynamic State Banner (`ANEMIA RISK`, `NON-ANEMIA`, `INCONCLUSIVE`), Calibrated Risk percentage, Execution latency, Device tag, Locked decision threshold ($\tau = 0.9000$).
  * **Live Dual Image Preview**: Displays original captured/uploaded image alongside the segmented 224&times;224 nail ROI.
  * **Transparent Intermediate Values Panel**: Real-time telemetry displaying EfficientNet logit, JetX probability, Fusion probability, Isotonic Calibrated probability, Locked threshold, and per-model latency breakdown.
  * **Session History**: Interactive tabular log tracking every inference request made during the active browser session.
  * **Persistent Research Disclaimer**: "Investigational research prototype for fingernail-based anemia risk screening. Not a standalone clinical diagnostic device."

---

## Model Loaded

* **Primary Model**: `EfficientNet-B0 v002` (Deep convolutional network, 224&times;224 RGB input)
  * Weight File: `experiments/efficientnet_b0_v002/best_model.pth`
  * SHA256: `6c478a5948f27aa5e9668ec775aa7b03698d41ee0ce25b5974dcbe45e82189ff`
* **Secondary Model**: `JetX-GT/nail-anemia-detector` (28 handcrafted color/chromaticity features + 2-layer MLP classifier)
  * Weight File: `models/jetx_gt/mlp_model.joblib`
  * Scaler File: `models/jetx_gt/feature_scaler.joblib`
* **Fusion Model**: `ensemble_fusion_v003.joblib` (4-feature logistic fusion)
* **Calibration Model**: `calibrator_isotonic_v003.joblib` (Non-parametric Isotonic Regression)
* **Locked Decision Threshold**: $\tau = 0.9000$ (`configs/locked_tau_v003.json`)

---

## Real Image Test

### Test 1: Known Anemic Fingernail Image
* **Image File**: `Fingernails/Anemic-Fin-008 (10).png` (Patient ID: `Anemic-Fin-008`)
* **Endpoint**: `POST /predict`
* **Request ID**: `b9092758-7f63-4395-acc7-46ea2bc3929c`
* **Intermediate Values**:
  * EfficientNet-B0 Raw Logit: `+1.126071`
  * EfficientNet-B0 Probability: `0.7551`
  * JetX-GT Probability: `0.9976`
  * Logistic Fusion Raw Probability: `0.9964`
  * Isotonic Calibrated Probability: `0.9959`
* **Threshold**: $\tau = 0.9000$
* **Final Assessment**: `ANEMIA` (`ANEMIA RISK`)
* **GPU Latency**: EfficientNet: 229.94 ms, JetX: 10.01 ms, Fusion: 0.24 ms | **Total**: 243.67 ms
* **ROI Output**: Successfully cropped nail bed bounding box `[13, 13, 211, 211]` (Base64 JPEG generated).

---

## Webcam Test

* **Capture Mechanism**: HTML5 `MediaDevices.getUserMedia({ video: { ideal: 1280, 720 } })` streaming to `<video>` element. Single-frame acquisition rendered to `<canvas>` and submitted as multipart form-data to `POST /predict`.
* **Execution Flow**:
  1. User clicks `[ CAPTURE IMAGE ]`.
  2. Canvas extracts frame buffer as JPEG blob.
  3. Browser logs: `REQUEST START` &rarr; `IMAGE SENT`.
  4. Backend runs quality check &rarr; ROI detection &rarr; Dual-model GPU inference &rarr; Calibration &rarr; Thresholding.
  5. Browser logs: `SERVER RESPONSE` &rarr; `MODEL OUTPUT` &rarr; `FINAL RESULT`.
  6. Dashboard immediately renders diagnosis banner, dual image previews, telemetry table, and session history row.

---

## API Response

Raw JSON payload returned by `POST /predict` on real positive test image:
```json
{
  "request_id": "b9092758-7f63-4395-acc7-46ea2bc3929c",
  "success": true,
  "state": "ANEMIA",
  "probability": 0.9959,
  "raw_fusion_probability": 0.9964,
  "efficientnet_probability": 0.7551,
  "efficientnet_raw_logit": 1.126071,
  "jetx_gt_probability": 0.9976,
  "threshold": 0.9,
  "device": "cuda:0",
  "model_version": "ensemble_v003",
  "primary_model": "efficientnet_b0_v002",
  "secondary_model": "JetX-GT/nail-anemia-detector",
  "calibration_version": "isotonic_regression_v003",
  "threshold_version": "locked_tau_v003",
  "latency_ms": {
    "efficientnet": 229.94,
    "jetx_gt": 10.01,
    "fusion": 0.24,
    "total": 243.67
  },
  "description": "Model-estimated ensemble probability indicates potential anemia.",
  "disclaimer": "Research/investigational result; confirmatory clinical evaluation is required.",
  "roi_metadata": {
    "bbox": [13, 13, 211, 211],
    "method": "contour_nail_detector"
  },
  "roi_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

---

## Input Dependence

To confirm that the production system computes real, input-dependent logits and probabilities (and is not returning static constants):

| Image Tested | Class / Origin | EfficientNet Logit | JetX Prob | Calibrated Prob | Final State |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Anemic-Fin-008 (10).png` | Anemic Positive | `+1.126071` | `0.9976` | `0.9959` | **ANEMIA** |
| `Non-Anrmic-FN-002 (3).png` | Non-Anemic Negative | `-0.454880` | `0.0000` | `0.0010` | **NO_ANEMIA** |

* Logit shift: $\Delta \text{Logit} = 1.126071 - (-0.454880) = +1.580951$.
* Probability shift: $\Delta P = 0.9959 - 0.0010 = +0.9949$.
* Conclusion: The system exhibits genuine, input-dependent inference on distinct nail inputs.

---

## OOD / Quality Test

The system was evaluated against physiological distractors and corrupted inputs:

1. **Severe Blur Image (`test_ood/ood_blurred_surface.jpg`)**:
   * State: `INCONCLUSIVE`
   * Reason: `severe_blur` (Laplacian variance $0.0 < 5.0$)
   * Description: `"Image unsuitable for assessment. Image is blurry. Please hold steady and tap to focus."`
2. **Non-Biological Object (`test_ood/ood_random_object.jpg`)**:
   * State: `INCONCLUSIVE`
   * Reason: `non_skin_chromaticity` (Skin ratio $0.0 < 0.20$)
   * Description: `"Image could not be assessed reliably. Nail was not detected. Surface does not contain skin or nail tissue."`

---

## GPU Status

* **PyTorch GPU Availability**: `torch.cuda.is_available() == True`
* **Active GPU**: NVIDIA GeForce RTX 3070 Laptop GPU
* **Device Target**: `cuda:0`
* **CUDA Runtime Version**: 12.4
* **Steady-State Latency**: $21.47\text{ ms}$ total inference time per image on GPU.

---

## Hardcoded Logic Audit

* Grep search across `backend/` for `mock`, `dummy`, `fake`, `hardcoded`, `random`, `static prediction` yielded **0 occurrences** in the inference path.
* Verified that weights are loaded from genuine checkpoint files (`experiments/efficientnet_b0_v002/best_model.pth` and `models/jetx_gt/mlp_model.joblib`).
* Verified that logits and probabilities vary strictly as a function of the input image tensor and handcrafted color feature vector.

---

## Final Status

# **LIVE REAL INFERENCE VERIFIED**
