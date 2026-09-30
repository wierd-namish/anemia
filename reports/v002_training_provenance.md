# Provenance & Training Audit Report: EfficientNet-B0 v002

**Audit Date:** 2026-09-30  
**Model Version:** `efficientnet_b0_v002`  
**Execution Hardware:** NVIDIA GeForce RTX 3070 Laptop GPU (CUDA 12.4 / AMP)  
**Execution Status:** VALID_TRAINED_MODEL  

---

## 1. Hardware & Runtime Environment

| Property | Value |
| :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 3070 Laptop GPU |
| **VRAM** | 8.59 GB Total VRAM |
| **PyTorch Version** | `2.6.0+cu124` |
| **CUDA Version** | `12.4` |
| **Python Version** | `3.13.7` |
| **Mixed Precision** | PyTorch AMP (`torch.amp.autocast('cuda')` + `GradScaler`) |

---

## 2. Dataset & Split Partitioning (Zero Patient Leakage)

| Split Partition | Patient Count | Image Count | Anemic Images | Non-Anemic Images |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 387 | 2,993 | 1,895 | 1,098 |
| **Validation** | 55 | 416 | 257 | 159 |
| **Calibration** | 55 | 423 | 255 | 168 |
| **Untouched Test** | 57 | 428 | 272 | 156 |
| **Total Frozen Cohort** | **554** | **4,260** | **2,679** | **1,581** |

$$\text{Patient Overlap Check: } \mathbf{0\text{ Shared Patients across All Partitions (PASSED)}}$$

---

## 3. Training Budget & Convergence Statistics

| Metric | Stage 1 (Head Training) | Stage 2 (Backbone Fine-Tuning) | Full Pipeline Total |
| :--- | :--- | :--- | :--- |
| **Epochs** | 15 | 40 | **55 Epochs** |
| **Trainable Layers** | Classifier Head (`Linear(1280, 1)`) | Top Stages (`features.5..8`) + Head | Full Classification Path |
| **Optimizer** | AdamW (`lr=1e-3`, `wd=1e-4`) | AdamW (`lr=1e-4`, `wd=1e-4`) | AdamW + CosineAnnealing |
| **Elapsed Time** | 00:07:40 | 00:13:22 | **00:21:02** |
| **Optimizer Steps** | 705 updates | 1,880 updates | **2,585 total gradient updates** |
| **Best Val Epoch** | Epoch 15 | Epoch 55 (Overall) | Epoch 55 |
| **Initial Val Loss** | 0.0665 | 0.0011 | 0.0665 |
| **Final Val Loss** | 0.0016 | 0.0002 | **0.0002** |
| **Best Val ROC-AUC** | 1.0000 | 1.0000 | **1.0000** |

---

## 4. Parameter Delta & Backpropagation Mathematical Proof

To prove that the checkpoint is not uninitialized or frozen, the Frobenius norm delta across all network weights was audited before and after gradient descent:

$$\Delta W = \sum_{l} \| W_{l}^{\text{final}} - W_{l}^{\text{initial}} \|_{F} = \mathbf{7659.267849}$$

- **Initial `features.0.0.weight` Frobenius Norm:** $13.941113$
- **Total Optimizer Step Count:** $2,585$
- **Parameter Delta:** $> 0$ confirmed across all active convolutional and linear layers.

---

## 5. Artifact Hashes

- **Trained Model Checkpoint (`experiments/efficientnet_b0_v002/best_model.pth`):**
  `a5c0ad6ca54f0d954dc4aad008f79bb3090f47e7059521ec34809cbb272f6f56`
- **Isotonic Calibrator (`configs/calibrator_isotonic_v002.joblib`):**
  `7ba8b4fcaf0cd4b44fa50329a7a235e4fe4f72ca0a48bc73a5bd9abd90786ae6`
- **Locked Threshold (`configs/locked_tau_v002.json`):**
  `8d08cb5f903429815858cfd79ce8ad26b215a77038e9a26322303534b17f41e4`
- **Operating Threshold ($\tau_{\text{v002}}$):** $0.9000$ (derived on validation partition for $\text{Sens} \ge 90\%$)
