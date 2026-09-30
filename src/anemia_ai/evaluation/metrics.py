"""
Comprehensive Diagnostic and Statistical Evaluation Metrics.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def compute_diagnostic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes diagnostic classification and probabilistic metrics.

    Calculates:
    - Sensitivity, Specificity, PPV, NPV, F1 Score
    - Confusion Matrix (TP, TN, FP, FN)
    - ROC-AUC and PR-AUC
    - Brier Score
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(rec_vals, prec_vals))
    except Exception:
        pr_auc = 0.0

    brier = float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))

    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "threshold": round(float(threshold), 4),
    }


def aggregate_patient_predictions(
    df_preds: pd.DataFrame,
    prob_col: str = "calibrated_prob",
    patient_col: str = "patient_id",
    label_col: str = "anemia_label",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregates image-level predictions by patient identifier using mean probability.
    """
    grouped = df_preds.groupby(patient_col).agg({
        prob_col: "mean",
        label_col: lambda x: int(x.mode()[0]),
    }).reset_index()
    return grouped[label_col].values, grouped[prob_col].values
