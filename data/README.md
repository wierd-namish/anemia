# Data Directory & Clinical Protection Guidelines

## Data Protection & Privacy Notice

> [!IMPORTANT]
> **Strict Data Privacy Policy**: Clinical patient photographs, identifiable patient records, and protected health information (PHI) must **never** be committed to version control. The repository `.gitignore` strictly excludes `Fingernails/`, `data/raw/`, `data/processed/`, and individual clinical CSVs.

## Directory Structure

```
data/
├── README.md               <- This documentation
├── samples/                <- Synthetic/demonstration images for local testing
├── splits/                 <- Patient-partitioned split index files (train, val, cal, test)
├── raw/                    <- [IGNORED] Raw clinical photographic datasets
└── processed/              <- [IGNORED] Extracted and cropped ROI datasets
```

## Dataset Partitions & Partition Independence

To prevent clinical data leakage, dataset partitions must be stratified strictly at the **patient identifier level** (`patient_id`), ensuring zero overlap across:
1. `train.csv`: Model parameter optimization
2. `val.csv`: Model selection, checkpoint freezing, threshold optimization
3. `calibration.csv`: Isotonic regression / Platt scaling fitting
4. `test.csv`: Untouched held-out clinical generalization evaluation

### Partition Schema

Each split CSV file must contain the following standardized columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | `str` | Unique image file identifier |
| `patient_id` | `str` | De-identified participant code (used for leakage-free splitting) |
| `image_path` | `str` | Relative or absolute path to local photographic image |
| `anemia_label` | `int` | Binary ground-truth indicator (`1` = Anemia, `0` = Non-Anemic) |
| `hb_level` | `float` | Gold-standard laboratory hemoglobin concentration in g/dL |

## Verification of Patient Leakage

To verify that your dataset split files have zero patient ID overlap:

```bash
python -c "
from anemia_ai.evaluation.leakage import audit_partitions
import pandas as pd
splits = {
    'train': pd.read_csv('data/splits/train.csv'),
    'val': pd.read_csv('data/splits/val.csv'),
    'calibration': pd.read_csv('data/splits/calibration.csv'),
    'test': pd.read_csv('data/splits/test.csv')
}
print(audit_partitions(splits))
"
```
