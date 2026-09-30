# Reproducible Model Training & Calibration

## Training Overview

The Anemia AI training pipeline implements a two-stage transfer learning schedule using PyTorch with deterministic random seeding, mixed precision (AMP), and patient-level data partitioning.

## Command Line Execution

To execute the training pipeline using configuration:

```bash
python scripts/train.py --config configs/training/efficientnet_b0_v002.yaml
```

## Two-Stage Optimization Schedule

```
                     +-----------------------------------+
                     |  STAGE 1: HEAD TRAINING (15 Ep)   |
                     |  Backbone: Frozen                 |
                     |  Classifier: AdamW (lr = 1e-3)    |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  STAGE 2: FINE-TUNING (40 Ep)     |
                     |  Top Backbone: features.5..8      |
                     |  Backbone lr = 1e-4, Head = 3e-4  |
                     |  Scheduler: CosineAnnealingLR     |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   ISOTONIC CALIBRATION (Cal Split)|
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |   THRESHOLD LOCKING (Val Split)   |
                     |   Target Sensitivity >= 90%       |
                     +-----------------------------------+
```

## Ensemble Training (v003)

To train the dual-model ensemble fusion and calibrator:

```bash
python scripts/train_ensemble_v003.py
```

This:
1. Extracts paired vision logits and handcrafted color probabilities.
2. Optimizes a logistic regression fusion model on the development train split.
3. Fits Isotonic Calibrator v003 on the held-out calibration split.
4. Locks decision threshold &tau;=0.9000 on the validation split.
5. Evaluates performance on the untouched test split.
