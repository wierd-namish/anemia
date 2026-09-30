"""
Patient-level data leakage audit and split verification.
"""

from typing import Any, Dict
import pandas as pd


def audit_partitions(
    splits: Dict[str, pd.DataFrame],
    patient_col: str = "patient_id",
) -> Dict[str, Any]:
    """
    Audits dataset partitions to ensure zero patient ID overlap between splits.
    """
    patient_sets = {}
    for name, df in splits.items():
        if patient_col not in df.columns:
            raise ValueError(f"Partition '{name}' is missing '{patient_col}' column.")
        patient_sets[name] = set(df[patient_col].unique())

    split_names = list(splits.keys())
    overlaps = {}
    has_leakage = False

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            intersection = patient_sets[s1].intersection(patient_sets[s2])
            pair_key = f"{s1}_vs_{s2}"
            overlaps[pair_key] = len(intersection)
            if len(intersection) > 0:
                has_leakage = True

    return {
        "status": "FAIL (Patient Leakage Detected)" if has_leakage else "PASS (Zero Patient Leakage)",
        "patient_counts": {name: len(s) for name, s in patient_sets.items()},
        "total_unique_patients": len(set.union(*patient_sets.values())) if patient_sets else 0,
        "pairwise_overlaps": overlaps,
    }


# Backwards compatibility alias
audit_patient_leakage = audit_partitions
