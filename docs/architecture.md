# System Architecture & Technical Specification

## Overview

Anemia AI is a modular, production-grade deep learning system designed for non-invasive, point-of-care anemia risk screening from fingernail photographs. The architecture decouples image acquisition, quality assurance, region-of-interest (ROI) extraction, model inference, probability calibration, and API delivery.

```
                      +---------------------------------------+
                      |       CLIENT / CAMERA / UPLOAD        |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         FASTAPI BACKEND ROUTER        |
                      |   /api/v1/predict, /predict-multiple   |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |          IMAGE QUALITY GATE           |
                      |  Blur, Exposure, Glare, Polish, Res   |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       NAIL ROI DETECTION ENGINE       |
                      |    Contour Segmentation & OOD Check   |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             +-----------------------+           +-----------------------+
             | EFFICIENTNET-B0 v002  |           | JETX-GT HF NAIL MODEL |
             | Deep Vision Backbone  |           | 27 Color Features/MLP |
             +-----------------------+           +-----------------------+
                        |                                   |
                   Raw Logit                           Probability
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         LOGISTIC FUSION MODEL         |
                      |          (Ensemble v003)              |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       ISOTONIC CALIBRATION v003       |
                      |     Empirical Probability Mapping     |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        LOCKED THRESHOLD (&tau;=0.90)       |
                      |   ANEMIA / NO_ANEMIA / INCONCLUSIVE   |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     STRUCTURED RESPONSE & PREVIEW     |
                      +---------------------------------------+
```

## Core Module Responsibilities

### 1. Preprocessing (`anemia_ai.preprocessing`)
- **`image_quality.py`**: Assesses incoming frames for optical blur (Laplacian variance), illumination (underexposure/overexposure), specular glare (>250 intensity), and artificial pigments (nail polish hues in HSV).
- **`nail_detection.py`**: Isolates subungual tissue via chrominance Otsu thresholding in YCrCb color space, morphology, bounding box expansion, and physiological chromaticity validation.
- **`feature_extraction.py`**: Extracts 28 handcrafted color/texture features for baseline and 27 color features for the JetX-GT Hugging Face model.

### 2. Model Architecture & Abstraction (`anemia_ai.models`)
- **`BaseModel`**: Polymorphic abstract base class defining `predict()`, `is_ready()`, `model_name`, and `model_version`.
- **`EfficientNetB0Model`**: Deep convolutional neural network fine-tuned on subungual nail photographs.
- **`JetXNailModel`**: Handcrafted feature model combining StandardScaler and MLPClassifier.
- **`ModelRegistry`**: Factory pattern facilitating seamless integration of future vision transformers or CNN architectures.

### 3. Calibration (`anemia_ai.calibration`)
- **`ProbabilityCalibrator`**: Fits isotonic regression piecewise monotone curves on held-out calibration partitions, reducing Expected Calibration Error (ECE) and optimizing Brier score.

### 4. Inference & Services (`anemia_ai.inference`, `anemia_ai.services`)
- **`TwoModelEnsembleService`**: Dual-model inference orchestrator with empirical logistic fusion, isotonic calibration, and locked thresholding.
- **`DiagnosticInferenceService`**: Single-model fallback pipeline for standalone EfficientNet-B0 evaluation.
- **`AssessmentService`**: High-level application service managing health checks, metadata, and execution tracing.

### 5. API Layer (`anemia_ai.api`)
- FastAPI application with versioned endpoints (`/api/v1/health`, `/api/v1/model-info`, `/api/v1/predict`, `/api/v1/predict-multiple`), request tracing middleware, CORS configuration, and domain exception handlers.

## Extension Points for Future Development

1. **New AI Models**: Implement `BaseModel` in `src/anemia_ai/models/` and register with `register_model()`.
2. **New Modalities**: Extend `NailDetector` with deep keypoint detectors (e.g. YOLOv8 or MediaPipe Hand landmarks).
3. **Clinical Cohorts**: Add demographic subgroup evaluation tooling in `anemia_ai.evaluation`.
4. **Cloud Inference**: Containerize with standard Dockerfile and deploy to Kubernetes or AWS ECS with GPU autoscaling.
