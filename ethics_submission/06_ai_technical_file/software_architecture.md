# Software Architecture Specification

**Document Identifier:** ETH-SUB-TECH-002  
**System Name:** Anemia Nail Diagnostic Mobile & Backend Platform  
**Version:** 1.0  
**Date:** 2026-09-30  
**Ethics Submission Status:** Prepared for ethics submission (PENDING ETHICS APPROVAL)  

---

## 1. System Topology & Component Layout

```
[ CLINICAL SMARTPHONE ]
  │
  ├── UI Viewfinder: HTML5 / Vanilla CSS / JavaScript
  ├── Frame Capture: Canvas API (In-Memory JPEG/PNG Blob)
  └── Transport: HTTPS / TLS 1.3 REST API calls
        │
        ▼
[ FASTAPI BACKEND SERVER ]
  │
  ├── Web Framework: FastAPI (Python 3.13 / Uvicorn)
  ├── In-Memory Stream Decoders: PIL (Pillow)
  │
  ├── Preprocessing Engine:
  │     ├── Image Quality Gate (Laplacian / Luminance / Saturation)
  │     └── Nail Detector (OpenCV YCrCb Contour Extractor)
  │
  ├── Inference Service:
  │     ├── PyTorch EfficientNet-B0 (TorchVision Engine, CPU)
  │     ├── Isotonic Calibrator (Scikit-Learn Joblib Model)
  │     └── Patient-Level Aggregator (Mean Calibrated Probability)
  │
  └── Output Serializer:
        └── JSON Screening State {ANEMIA, NO_ANEMIA, INCONCLUSIVE}
```

---

## 2. Component Specifications

| Layer | Technology | Responsibilities | Privacy & Safety Controls |
|:---|:---|:---|:---|
| **Client Frontend** | HTML5, CSS3, ES6 JavaScript | Live camera display, guide overlay, multi-nail progress, state card rendering. | **No file picker, no gallery upload, no local disk caching.** |
| **Backend API** | FastAPI / Uvicorn | Endpoints `/health`, `/model-info`, `/predict`, `/predict-multiple`. | CORS restricted, strict MIME-type validation. |
| **Preprocessing** | OpenCV 4.x, NumPy | Blur variance, luminance exposure, glare detection, contour extraction. | Zero patient disk persistence; processed in RAM. |
| **Deep Learning** | PyTorch 2.14, TorchVision 0.29 | EfficientNet-B0 backbone forward inference. | Pre-allocated weights, deterministic CPU inference. |
| **Calibration** | Scikit-Learn 1.9, Joblib | Isotonic non-parametric probability transformation. | Monotonic probability scaling. |

---

## 3. Deployment Modes

1. **Development & Validation Mode:** Backend listens on local network (`0.0.0.0:8000`) for clinical tablet/phone testing in healthcare facilities.
2. **Clinical Data Collection Control:** Prospective participant data collection is **DISABLED** pending full institutional ethics approval from GHS-ERC.
