# EfficientNet-B0 v002 Pretrained Weight Initialization Audit

**Audit Date:** 2026-09-30  
**Model Architecture:** `torchvision.models.efficientnet_b0`  
**Weights Source:** `torchvision.models.EfficientNet_B0_Weights.DEFAULT` (ImageNet-1K V1)

---

## 1. Programmatic Initialization Verification

Prior to retraining, the model weights were verified to ensure official ImageNet transfer-learning features were loaded rather than random Kaiming/Gaussian distributions.

### Weight Distribution Comparison:

| Layer Parameter | Fresh Uninitialized (v001) | ImageNet Pretrained (v002 Loaded) | Verification Status |
| :--- | :---: | :---: | :---: |
| **`features.0.0.weight` (Stem Conv)** | $\mu = 0.0005, \sigma = 0.0841$ | $\mu = -0.0008, \sigma = 0.3297$ | **IMAGE_NET_WEIGHTS_VERIFIED** |
| **`features.0.1.running_mean` (Stem BN)** | $\mu = 0.0000, \sigma = 0.0000$ | $\mu = 0.0271, \sigma = 0.0614$ | **TRAINED_BN_STATISTICS_VERIFIED** |
| **`features.0.1.running_var` (Stem BN)** | $\mu = 1.0000, \sigma = 0.0000$ | $\mu = 0.2319, \sigma = 0.1874$ | **TRAINED_BN_STATISTICS_VERIFIED** |
| **Backbone MBConv Layers (1-8)** | Geometric decay to $\approx 10^{-15}$ | Well-conditioned transfer features | **NO ACTIVATION COLLAPSE** |

---

## 2. Initialization Configuration

```json
{
  "model_version": "efficientnet_b0_v002",
  "base_architecture": "EfficientNet-B0",
  "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
  "in_features": 1280,
  "head_architecture": "Sequential(Dropout(p=0.3, inplace=True), Linear(in_features=1280, out_features=1, bias=True))",
  "training_input": "224x224 RGB Fingernail ROI Normalized with ImageNet mean/std",
  "initialization_timestamp": "2026-09-30T02:22:00Z"
}
```
