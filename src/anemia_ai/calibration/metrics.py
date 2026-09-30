"""
Calibration Performance Metrics: Expected Calibration Error (ECE) and Brier Score.
"""

from typing import Any, Dict
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def compute_calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Calculates Brier Score and Expected Calibration Error (ECE).
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.clip(np.asarray(probabilities), 0.0, 1.0)

    # 1. Brier Score Loss: mean((prob - true)^2)
    brier = float(brier_score_loss(y_true, probs))

    # 2. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_metrics = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        bin_size = int(np.sum(in_bin))

        if bin_size > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_size / len(probs)) * bin_error

            bin_metrics.append({
                "bin_range": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "sample_count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_error": round(bin_error, 4),
            })

    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")

    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(float(ece), 4),
        "calibration_curve_true": [float(v) for v in prob_true],
        "calibration_curve_pred": [float(v) for v in prob_pred],
        "bin_details": bin_metrics,
    }
