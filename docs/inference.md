# Inference Pipeline & API Reference

## Endpoints Specification

### 1. Health Status (`GET /api/v1/health` and `GET /health`)
Returns model readiness, device information, and active versions.

**Response Example:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "EfficientNet-B0 + JetX-GT Ensemble",
  "model_version": "ensemble_v003",
  "primary_model": "efficientnet_b0_v002",
  "secondary_model": "JetX-GT/nail-anemia-detector",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3070 Laptop GPU",
  "api_version": "v1"
}
```

### 2. Model Information (`GET /api/v1/model-info` and `GET /model-info`)
Returns complete model lineage, locked threshold value, and clinical disclaimers.

### 3. Single Nail Assessment (`POST /api/v1/predict` and `POST /predict`)
Evaluates a single nail photograph from camera capture or upload.

**Request:** `multipart/form-data` with `file` (image).

**Response Example:**
```json
{
  "request_id": "83a839bc-ba3a-4191-80e7-8bc532bf243c",
  "success": true,
  "state": "ANEMIA",
  "probability": 0.9990,
  "raw_fusion_probability": 0.9942,
  "efficientnet_probability": 0.9856,
  "efficientnet_raw_logit": 4.2281,
  "jetx_gt_probability": 0.9712,
  "threshold": 0.9000,
  "device": "cuda:0",
  "model_version": "ensemble_v003",
  "primary_model": "efficientnet_b0_v002",
  "secondary_model": "JetX-GT/nail-anemia-detector",
  "calibration_version": "isotonic_regression_v003",
  "threshold_version": "locked_tau_v003",
  "latency_ms": {
    "efficientnet": 12.4,
    "jetx_gt": 4.2,
    "fusion": 0.8,
    "total": 18.5
  },
  "description": "Model-estimated ensemble probability indicates potential anemia.",
  "disclaimer": "Research/investigational result; confirmatory clinical evaluation is required.",
  "roi_metadata": {
    "bbox": [25, 30, 275, 270],
    "method": "contour_nail_detector"
  },
  "roi_image_base64": "data:image/jpeg;base64,..."
}
```

### 4. Multi-Nail Assessment (`POST /api/v1/predict-multiple` and `POST /predict-multiple`)
Evaluates 2 to 4 nail images from a patient and aggregates valid predictions using mean calibrated risk.
