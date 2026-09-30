"""
Probability Calibration Fitting and Evaluation Tooling.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.calibration.calibrator import ProbabilityCalibrator
from anemia_ai.calibration.metrics import compute_calibration_metrics
from anemia_ai.config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="Fit and Evaluate Probability Calibrator")
    parser.add_argument("--method", type=str, default="isotonic", choices=["isotonic", "platt"])
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    cal_file = settings.experiments_dir / "efficientnet_b0_v002" / "cal_predictions.csv"
    if not cal_file.exists():
        print(f"[ERROR] Calibration predictions file not found: {cal_file}")
        sys.exit(1)

    df_cal = pd.read_csv(cal_file)
    scores = df_cal["sigmoid_prob"].values
    labels = df_cal["anemia_label"].values

    calibrator = ProbabilityCalibrator(method=args.method)
    calibrator.fit(scores, labels)

    uncal_metrics = compute_calibration_metrics(labels, scores)
    cal_scores = calibrator.calibrate(scores)
    cal_metrics = compute_calibration_metrics(labels, cal_scores)

    print("=" * 70)
    print(f"CALIBRATION AUDIT — METHOD: {args.method.upper()}")
    print("=" * 70)
    print(f"Uncalibrated ECE: {uncal_metrics['expected_calibration_error']:.4f} | Brier: {uncal_metrics['brier_score']:.4f}")
    print(f"Calibrated ECE:   {cal_metrics['expected_calibration_error']:.4f} | Brier: {cal_metrics['brier_score']:.4f}")

    if args.save_path:
        out_p = Path(args.save_path)
        calibrator.save(out_p)
        print(f"[OK] Calibrator saved to: {out_p}")


if __name__ == "__main__":
    main()
