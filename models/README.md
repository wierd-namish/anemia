# Model Registry & Artifact Management

This directory manages machine learning models, manifests, and artifact metadata for the Anemia AI diagnostic system.

## Model Manifests

Model configurations and weights provenance are tracked deterministically in `models/manifests/`:

| Manifest File | Architecture | Version | Checkpoint Location | Input Resolution | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`efficientnet_b0_v001.json`](file:///c:/Users/Asus/MYPASS/models/manifests/efficientnet_b0_v001.json) | EfficientNet-B0 | `v001` | `experiments/efficientnet_b0_v001/best_model.pth` | 224×224 RGB | Historical Baseline |
| [`efficientnet_b0_v002.json`](file:///c:/Users/Asus/MYPASS/models/manifests/efficientnet_b0_v002.json) | EfficientNet-B0 | `v002` | `experiments/efficientnet_b0_v002/best_model.pth` | 224×224 RGB | Primary Vision Model |
| [`ensemble_v003.json`](file:///c:/Users/Asus/MYPASS/models/manifests/ensemble_v003.json) | Two-Model Ensemble | `v003` | `configs/ensemble_fusion_v003.joblib` | 224×224 RGB | Active Production |
| [`jetx_gt.json`](file:///c:/Users/Asus/MYPASS/models/manifests/jetx_gt.json) | 27 Color MLP | HF Official | `models/jetx_gt/mlp_model.joblib` | 224×224 RGB | Secondary Feature Model |

## External Model Synchronization

To automatically download and verify external Hugging Face model artifacts:

```bash
python scripts/download_models.py
```

## Checkpoint Verification

To verify the integrity and SHA256 hashes of all local weights:

```bash
python scripts/verify_checkpoint.py
```
