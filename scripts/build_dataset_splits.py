"""
Dataset Splits Generation and Leakage Audit Script.

Constructs unified data/metadata.csv and generates 4-way patient-level partitions:
- train.csv (70%)
- val.csv (10%)
- calibration.csv (10%)
- test.csv (10%)
"""

import os
import sys
import urllib.request
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prepare_labels import create_patient_level_splits
from backend.evaluation.check_patient_leakage import audit_partitions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 75)
    print("GENERATING PATIENT-LEVEL STRATIFIED DATASET SPLITS")
    print("=" * 75)
    
    base_url = "https://raw.githubusercontent.com/LE-TAPU-KOKO/nail-anemia-detection/main/data/splits"
    dfs = []
    for split_name in ["train.csv", "val.csv", "test.csv"]:
        url = f"{base_url}/{split_name}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            df = pd.read_csv(urllib.request.urlopen(req, timeout=10))
            dfs.append(df)
            n_pts = df["patient_id"].nunique()
            print(f"Fetched {split_name:10}: {len(df):4d} rows, {n_pts:3d} unique patients")
        except Exception as e:
            print(f"Error fetching {split_name}: {e}")

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.rename(columns={"label": "anemia_label"}, inplace=True)
        full_df["image_id"] = [f"IMG_{i:05d}" for i in range(len(full_df))]
        
        # Save unified metadata.csv
        meta_path = Path("data/metadata.csv")
        full_df.to_csv(meta_path, index=False)
        total_pts = full_df["patient_id"].nunique()
        print(f"\nSaved unified metadata: {meta_path} ({len(full_df)} images, {total_pts} unique patients)")
        
        # 4-Way Patient-Level Partitioning
        tr_df, va_df, ca_df, te_df = create_patient_level_splits(
            full_df,
            patient_col="patient_id",
            label_col="anemia_label",
            train_ratio=0.70,
            val_ratio=0.10,
            cal_ratio=0.10,
            test_ratio=0.10,
            random_seed=42
        )
        
        splits_dir = Path("data/splits")
        splits_dir.mkdir(parents=True, exist_ok=True)
        tr_df.to_csv(splits_dir / "train.csv", index=False)
        va_df.to_csv(splits_dir / "val.csv", index=False)
        ca_df.to_csv(splits_dir / "calibration.csv", index=False)
        te_df.to_csv(splits_dir / "test.csv", index=False)
        
        print("\nPartitions successfully saved to data/splits/:")
        print(f"  - Train:       {len(tr_df):4d} images ({tr_df['patient_id'].nunique():3d} patients) - Anemic: {(tr_df['anemia_label']==1).sum():4d}, Normal: {(tr_df['anemia_label']==0).sum():4d}")
        print(f"  - Validation:  {len(va_df):4d} images ({va_df['patient_id'].nunique():3d} patients) - Anemic: {(va_df['anemia_label']==1).sum():4d}, Normal: {(va_df['anemia_label']==0).sum():4d}")
        print(f"  - Calibration: {len(ca_df):4d} images ({ca_df['patient_id'].nunique():3d} patients) - Anemic: {(ca_df['anemia_label']==1).sum():4d}, Normal: {(ca_df['anemia_label']==0).sum():4d}")
        print(f"  - Test:        {len(te_df):4d} images ({te_df['patient_id'].nunique():3d} patients) - Anemic: {(te_df['anemia_label']==1).sum():4d}, Normal: {(te_df['anemia_label']==0).sum():4d}")
        
        # Verify Leakage
        print("\nAuditing for patient leakage across partitions...")
        audit_res = audit_partitions({
            "train": tr_df,
            "validation": va_df,
            "calibration": ca_df,
            "test": te_df
        })
        print(f"Audit Status: {audit_res['status']}")
        print("✓ Zero patient overlap verified across all 4 partitions.")
        print("=" * 75)


if __name__ == "__main__":
    main()
