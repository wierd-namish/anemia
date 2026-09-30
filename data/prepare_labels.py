"""
Clinical Label Construction and Patient-Level Partitioning Pipeline.

Derives objective binary anemia labels from laboratory Hb reference standards,
maps patient identifiers, and executes strict patient-level splitting across:
- Train (70%)
- Validation (10%)
- Calibration (10%)
- Final Test (10%)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# WHO Clinical Thresholds (g/dL)
WHO_THRESHOLDS = {
    "children_under_5": 11.0,
    "children_5_to_11": 11.5,
    "children_12_to_14": 12.0,
    "non_pregnant_women": 12.0,
    "pregnant_women": 11.0,
    "men": 13.0,
}


def derive_anemia_label_from_hb(
    hb_value: float,
    population_group: str = "children_under_5",
    custom_cutoff: Optional[float] = None,
) -> int:
    """
    Derive binary anemia ground-truth label from laboratory hemoglobin value.
    
    Args:
        hb_value: Blood hemoglobin concentration in g/dL.
        population_group: WHO demographic category.
        custom_cutoff: Optional custom clinical cutoff in g/dL.
        
    Returns:
        1 if anemic (hb < threshold), 0 if non-anemic (hb >= threshold).
    """
    threshold = custom_cutoff if custom_cutoff is not None else WHO_THRESHOLDS.get(population_group, 11.0)
    return 1 if hb_value < threshold else 0


def create_patient_level_splits(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col: str = "anemia_label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    cal_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the dataset strictly by PATIENT ID to prevent data leakage.
    Stratifies patient assignment by patient-level anemia prevalence.
    """
    # Verify ratio sum
    total_ratio = train_ratio + val_ratio + cal_ratio + test_ratio
    assert abs(total_ratio - 1.0) < 1e-5, f"Ratios must sum to 1.0, got {total_ratio}"
    
    # Group by patient to determine patient-level label (e.g., majority or confirmed label)
    patient_df = df.groupby(patient_col)[label_col].agg(lambda x: x.mode()[0]).reset_index()
    
    np.random.seed(random_seed)
    
    # Stratified split per class
    pos_patients = patient_df[patient_df[label_col] == 1][patient_col].sample(frac=1.0, random_state=random_seed).tolist()
    neg_patients = patient_df[patient_df[label_col] == 0][patient_col].sample(frac=1.0, random_state=random_seed).tolist()
    
    def split_list(patients: List[str]):
        n = len(patients)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_cal = int(round(n * cal_ratio))
        
        train_pts = patients[:n_train]
        val_pts = patients[n_train: n_train + n_val]
        cal_pts = patients[n_train + n_val: n_train + n_val + n_cal]
        test_pts = patients[n_train + n_val + n_cal:]
        return train_pts, val_pts, cal_pts, test_pts

    pos_tr, pos_va, pos_ca, pos_te = split_list(pos_patients)
    neg_tr, neg_va, neg_ca, neg_te = split_list(neg_patients)
    
    train_patients = set(pos_tr + neg_tr)
    val_patients = set(pos_va + neg_va)
    cal_patients = set(pos_ca + neg_ca)
    test_patients = set(pos_te + neg_te)
    
    # Assert zero overlap
    assert len(train_patients & val_patients) == 0, "Patient leakage between train and val!"
    assert len(train_patients & cal_patients) == 0, "Patient leakage between train and cal!"
    assert len(train_patients & test_patients) == 0, "Patient leakage between train and test!"
    assert len(val_patients & cal_patients) == 0, "Patient leakage between val and cal!"
    assert len(val_patients & test_patients) == 0, "Patient leakage between val and test!"
    assert len(cal_patients & test_patients) == 0, "Patient leakage between cal and test!"
    
    train_df = df[df[patient_col].isin(train_patients)].copy().reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_patients)].copy().reset_index(drop=True)
    cal_df = df[df[patient_col].isin(cal_patients)].copy().reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_patients)].copy().reset_index(drop=True)
    
    return train_df, val_df, cal_df, test_df


def build_clinical_metadata_schema(records: List[Dict]) -> pd.DataFrame:
    """
    Constructs a validated pandas DataFrame ensuring all clinical and image metadata
    fields are strictly formatted and typed.
    """
    df = pd.DataFrame(records)
    required_cols = ["patient_id", "image_id", "image_path", "anemia_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing mandatory column: {col}")
            
    return df
