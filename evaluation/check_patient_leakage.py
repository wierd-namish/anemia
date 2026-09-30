"""
Patient Leakage Verification Utility.

Audits data partitions to ensure zero patient ID contamination exists between:
- Training
- Validation
- Calibration
- Final Test

Fails the execution pipeline immediately if any leakage is discovered.
"""

from pathlib import Path
from typing import Dict, Set, List
import pandas as pd


class PatientLeakageError(Exception):
    """Raised when patient overlap is detected between distinct data partitions."""
    pass


def audit_partitions(
    splits: Dict[str, pd.DataFrame],
    patient_col: str = "patient_id"
) -> Dict[str, any]:
    """
    Audits a dictionary of partition DataFrames for patient ID leakage.
    
    Args:
        splits: Dict mapping partition name (e.g., 'train', 'val', 'calibration', 'test') to DataFrame.
        patient_col: Column name containing patient identifiers.
        
    Returns:
        Summary dictionary with patient counts, image counts, and overlap matrices.
        
    Raises:
        PatientLeakageError if any patient appears in more than one partition.
    """
    patient_sets: Dict[str, Set[str]] = {}
    summary = {}
    
    for name, df in splits.items():
        if patient_col not in df.columns:
            raise ValueError(f"Partition '{name}' is missing '{patient_col}' column.")
        pts = set(df[patient_col].dropna().unique())
        patient_sets[name] = pts
        summary[name] = {
            "image_count": len(df),
            "patient_count": len(pts),
            "anemic_images": int((df["anemia_label"] == 1).sum()) if "anemia_label" in df.columns else None,
            "non_anemic_images": int((df["anemia_label"] == 0).sum()) if "anemia_label" in df.columns else None,
        }
        
    # Check pairwise intersections
    partition_names = list(splits.keys())
    violations = []
    
    for i in range(len(partition_names)):
        for j in range(i + 1, len(partition_names)):
            p1, p2 = partition_names[i], partition_names[j]
            overlap = patient_sets[p1] & patient_sets[p2]
            if len(overlap) > 0:
                violations.append({
                    "partition_1": p1,
                    "partition_2": p2,
                    "leaked_patient_count": len(overlap),
                    "leaked_patient_ids": sorted(list(overlap))[:10]  # sample
                })
                
    if violations:
        error_msg = "\n" + "!" * 80 + "\n"
        error_msg += "CRITICAL PATIENT LEAKAGE DETECTED BETWEEN PARTITIONS!\n"
        for v in violations:
            error_msg += f"  - Overlap between '{v['partition_1']}' and '{v['partition_2']}': {v['leaked_patient_count']} patients\n"
            error_msg += f"    Sample IDs: {v['leaked_patient_ids']}\n"
        error_msg += "!" * 80 + "\n"
        raise PatientLeakageError(error_msg)
        
    return {
        "status": "PASSED_ZERO_LEAKAGE",
        "partitions_audited": partition_names,
        "partition_metrics": summary
    }


def verify_split_files(splits_dir: Path, patient_col: str = "patient_id") -> bool:
    """Verifies CSV split files on disk."""
    splits_dir = Path(splits_dir)
    partition_files = {
        "train": splits_dir / "train.csv",
        "validation": splits_dir / "val.csv",
        "calibration": splits_dir / "calibration.csv",
        "test": splits_dir / "test.csv",
    }
    
    dfs = {}
    for name, path in partition_files.items():
        if path.exists():
            dfs[name] = pd.read_csv(path)
            
    if not dfs:
        raise FileNotFoundError(f"No split files found in {splits_dir}")
        
    result = audit_partitions(dfs, patient_col=patient_col)
    print("✓ Patient Leakage Audit: PASSED (Zero patient overlap between any partitions)")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        splits_path = Path(sys.argv[1])
        verify_split_files(splits_path)
    else:
        print("Usage: python check_patient_leakage.py <path_to_splits_dir>")
