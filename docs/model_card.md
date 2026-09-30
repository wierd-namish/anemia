# Model Card: Anemia AI Ensemble (v003)

## Model Details

- **Model Identifier**: `ensemble_v003`
- **Primary Model**: `efficientnet_b0_v002` (Deep Convolutional Vision Backbone)
- **Secondary Model**: `JetX-GT/nail-anemia-detector` (Handcrafted 27 Color Features + Multi-Layer Perceptron)
- **Fusion Mechanism**: Empirical Logistic Regression (`configs/ensemble_fusion_v003.joblib`)
- **Probability Calibration**: Non-parametric Isotonic Regression (`configs/calibrator_isotonic_v003.joblib`)
- **Locked Decision Threshold**: &tau; = 0.9000 (`configs/locked_tau_v003.json`)
- **Release Date**: October 2026
- **License**: Research Prototype / Proprietary Academic License

## Intended Use

- **Intended Purpose**: Point-of-care investigational screening for pediatric anemia risk from fingernail photographs.
- **Intended Users**: Clinical researchers, healthcare workers, and study coordinators operating under clinical trial protocols.
- **Out-of-Scope Uses**:
  - Definitive clinical diagnosis without confirmatory laboratory blood tests (CBC/Hemoglobin).
  - Assessment on toenails, conjunctiva, or non-ungual anatomical surfaces.
  - Evaluation of fingers with nail polish, severe fungal dystrophy, artificial nails, or heavy trauma.

## Input & Output Modality

- **Input Modality**: Color photographic images of fingernails (JPEG, PNG, WebP) with minimum 128×128 pixel resolution.
- **Output Modality**:
  - Binary Diagnostic State: `ANEMIA`, `NO_ANEMIA`, or `INCONCLUSIVE`
  - Calibrated Risk Probability: Empirical estimate in range `[0.001, 0.999]`
  - Intermediate Pipeline Breakdown: EfficientNet logit, JetX probability, fusion score, and processing latency.

## Training & Evaluation Lineage

- **Development Cohort**: Retrospective pediatric Ghanaian participant dataset.
- **Partition Strategy**: Stratified 4-way patient-level split (Train: 70%, Validation: 10%, Calibration: 10%, Test: 10%) with verified zero patient ID overlap.
- **Training Budget**: 15 epochs classifier head warm-up + 40 epochs top backbone fine-tuning with Cosine Annealing.
- **Hardware**: NVIDIA GeForce RTX 3070 with mixed-precision training (AMP).

## Performance Summary on Held-Out Test Set

| Metric | Image-Level (Untouched Test) | Patient-Level (Aggregated Mean) |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.8650** | **0.8840** |
| **Sensitivity** | **91.20%** | **92.30%** |
| **Specificity** | **78.40%** | **81.50%** |
| **Brier Score** | **0.1120** | — |
| **Expected Calibration Error (ECE)** | **0.0380** | — |

## Known Limitations & Failure Modes

1. **Illumination Shift**: Extreme shadows or harsh direct flash glare may trigger inconclusive rejection.
2. **Artificial Pigments**: Blue, red, green, or black nail enamels obstruct subungual vascularization and are rejected.
3. **Severe Motion Blur**: Camera movement causing Laplacian variance < 5.0 is rejected.
4. **Demographic Generalization**: Retrospective validation performed on pediatric cohort; multi-center prospective validation is required for adult populations.

## Ethical & Regulatory Considerations

> [!CAUTION]
> This model is an **investigational research prototype**. It has not been cleared or approved by the US FDA, CE Mark authorities, or other national medical device regulatory bodies. Confirmatory laboratory hematology is mandatory.
