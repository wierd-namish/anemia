# Forensic Audit & Root Cause Analysis: Constant Prediction Bug

**Date:** 2026-09-30  
**Investigation Topic:** Investigation into why valid fingernail images produced constant `probability = 0.538` and `state = ANEMIA`.

---

## 1. Executive Summary & Structured Answers

| Question | Assessment | Detailed Finding |
| :--- | :---: | :--- |
| **A. Are input images different?** | **YES** | Five visually distinct files with unique SHA-256 hashes and dimensions (`2048x1536`, `2713x1694`, `814x509`, `300x300`). |
| **B. Are ROI crops different?** | **YES** | The `NailDetector` localized different bounding boxes (`(509, 501, 1279, 1425)`, `(203, 102, 1699, 1592)`, `(51, 30, 712, 479)`, etc.) producing unique 224x224 RGB crops and unique image hashes. |
| **C. Are tensors different?** | **YES** | Preprocessed PyTorch tensors have distinct SHA-256 digests and distinct pixel distributions (e.g. means ranging from `0.271` to `0.841`). |
| **D. Are raw CNN logits different?** | **NO** | Every image produces `logit = 0.011419061571` (identical to 12 decimal places). |
| **E. Are raw probabilities different?** | **NO** | `sigmoid(0.011419061571) = 0.502854734373` identically across all inputs. |
| **F. Is calibrator input correct?** | **YES** | Calibrator receives `sigmoid(logit)` in range $[0.03, 0.97]$, which aligns with `X_thresholds_`. |
| **G. Does calibration collapse outputs?** | **NO** | Calibrator is a monotonically increasing piecewise step function; it receives identical inputs ($0.502855$) and deterministically outputs $0.538342$. |
| **H. Is there frontend caching?** | **NO** | Frontend captures new blobs and receives dynamic responses; now verified with per-request UUID `request_id`. |
| **I. Is there backend caching?** | **NO** | Backend `/predict` processes each uploaded multipart payload independently in memory. |
| **J. Exact Root Cause** | **IDENTIFIED** | The checkpoint `experiments/efficientnet_b0_v001/best_model.pth` was created with `pretrained=False` and un-updated default batchnorm statistics (`running_mean=0`, `running_var=1`). Depthwise separable convolutions without pretraining cause activations to decay geometrically across the 16 MBConv blocks ($0.03 \to 0.001 \to 10^{-5} \to 10^{-8} \to 10^{-12} \to 10^{-15} \approx 0$). The final linear layer evaluates $W \cdot x + b = W \cdot 0 + 0.01141906 = 0.01141906$. |

---

## 2. Component-by-Component Traceability Table

| Image Name | Dimensions | Input File SHA-256 | ROI Bounding Box | ROI SHA-256 | Tensor SHA-256 | Raw Logit (12 d.p.) | Raw Sigmoid | Calibrated Prob | Final State |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `real_index_nail.jpg` | 2048x1536 | `ac6f51c39c...` | `(509, 501, 1279, 1425)` | `b0ffe3e8f3...` | `e5d82feb39...` | `0.011419061571` | `0.502854734373` | `0.538` | `ANEMIA` |
| `real_koilonychia_anemia.jpg` | 2713x1694 | `f64693a8c7...` | `(203, 102, 1699, 1592)` | `783bb6a119...` | `d084f753e4...` | `0.011419061571` | `0.502854734373` | `0.538` | `ANEMIA` |
| `01_known_anemia_nail.jpg` | 814x509 | `ae4e0f3fd8...` | `(51, 30, 712, 479)` | `4f7da4497b...` | `2a00a64123...` | `0.011419061571` | `0.502854734373` | `0.538` | `ANEMIA` |
| `02_known_healthy_nail.jpg` | 300x300 | `3e34d71728...` | `(62, 43, 239, 228)` | `826b143fbe...` | `a49fed1da9...` | `0.011419061571` | `0.502854734373` | `0.538` | `ANEMIA` |
| `03_normal_index_nail.jpg` | 2048x1536 | `c3403b7543...` | `(526, 503, 1277, 1424)` | `102d8857e4...` | `dabe671ce2...` | `0.011419061571` | `0.502854734373` | `0.538` | `ANEMIA` |

---

## 3. Mathematical Layer-by-Layer Activation Breakdown

Forward propagation of `real_index_nail.jpg` through `model.features`:

```
Layer                  Output Tensor Shape       Mean Activ.        Std Dev            Max Abs Value
------------------------------------------------------------------------------------------------------
Input Tensor           [1, 3, 224, 224]          +3.8345e-01        9.8408e-01         2.6400e+00
features.0 (Stem Conv) [1, 32, 112, 112]         +3.6387e-02        2.2261e-01         1.7685e+00
features.1 (MBConv1)   [1, 16, 112, 112]         -7.5444e-03        4.8499e-02         2.1919e-01
features.2 (MBConv2)   [1, 24, 56, 56]           +1.2069e-04        1.3736e-03         7.0729e-03
features.3 (MBConv3)   [1, 40, 28, 28]           +2.2821e-06        3.5334e-05         2.2711e-04
features.4 (MBConv4)   [1, 80, 14, 14]           -3.0181e-08        5.3066e-07         3.1064e-06
features.5 (MBConv5)   [1, 112, 14, 14]          +5.0723e-12        6.3249e-09         4.7802e-08
features.6 (MBConv6)   [1, 192, 7, 7]            +2.3286e-12        6.6427e-11         3.8524e-10
features.7 (MBConv7)   [1, 320, 7, 7]            +3.2846e-14        5.1452e-13         4.0181e-12
features.8 (Head Conv) [1, 1280, 7, 7]           +1.6138e-15        1.8188e-13         1.4937e-12
AdaptiveAvgPool2d      [1, 1280]                 +1.6138e-15        1.8188e-13         1.4937e-12
Linear Classifier      [1, 1]                    +1.1419e-02        0.0000e+00         1.1419e-02
```

### Linear Output Derivation:
$$\text{Logit} = W \cdot x + b = \sum_{i=1}^{1280} (W_i \cdot \approx 10^{-15}) + 0.011419061571 = 0.011419061571$$
$$\text{Raw Probability} = \sigma(0.011419061571) = 0.502854734373$$
$$\text{Calibrated Probability} = f_{\text{iso}}(0.502854734373) = 0.538341617619 \approx 0.538$$
$$\text{Diagnostic State} = \mathbb{I}(0.538 \ge 0.48) = \text{ANEMIA}$$

---

## 4. Calibrator Surface & Threshold Analysis

The `IsotonicRegression` calibrator (`experiments/efficientnet_b0_v001/calibrator.joblib`) was fitted on 22 empirical thresholds:
- Input domain: $[0.029968, 0.967509]$
- Output range: $[0.001, 0.999]$
- At $x = 0.502855$, the step function maps to $y = 0.538342$.

---

## 5. Summary of System Actions & Verification

1. **UUID Request Tracking:** Added unique `request_id` (UUID4) to `/predict` and `/predict-multiple` to trace each inference.
2. **ROI Artifacts Preserved:** Extracted and saved all 5 debug ROI images to `reports/debug_constant_prediction/`.
3. **No Hardcoded Values:** Confirmed neither frontend nor backend contains hardcoded probability or state logic.
4. **Governance Integrity:** All model weights, calibrator artifacts, and threshold configurations remain cryptographically frozen and untampered.
