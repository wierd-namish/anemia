"""
Ensemble Fusion Trainer and Calibrator (v003).
"""

from pathlib import Path
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from anemia_ai.calibration.calibrator import ProbabilityCalibrator
from anemia_ai.core.logging import logger
from anemia_ai.evaluation.metrics import aggregate_patient_predictions, compute_diagnostic_metrics
from anemia_ai.utils.hashing import compute_sha256


class EnsembleTrainer:
    """Trains logistic regression fusion and isotonic calibrator for dual-model ensemble."""

    def __init__(self, configs_dir: Path):
        self.configs_dir = Path(configs_dir)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def fit_fusion_and_calibration(
        self,
        train_features: pd.DataFrame,
        cal_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Fits fusion on train, calibrator on calibration, derives threshold on validation, evaluates on test.
        """
        feature_cols = ["eff_logit", "eff_prob", "jetx_logit", "jetx_prob"]
        X_train = train_features[feature_cols].values
        y_train = train_features["anemia_label"].values

        # 1. Train Fusion Classifier
        fusion_model = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        fusion_model.fit(X_train, y_train)

        fusion_path = self.configs_dir / "ensemble_fusion_v003.joblib"
        joblib.dump(fusion_model, fusion_path)
        fusion_sha = compute_sha256(fusion_path)

        # 2. Raw Probabilities
        cal_raw = fusion_model.predict_proba(cal_features[feature_cols].values)[:, 1]
        val_raw = fusion_model.predict_proba(val_features[feature_cols].values)[:, 1]
        test_raw = fusion_model.predict_proba(test_features[feature_cols].values)[:, 1]

        # 3. Fit Calibrator
        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(cal_raw, cal_features["anemia_label"].values)

        calib_path = self.configs_dir / "calibrator_isotonic_v003.joblib"
        joblib.dump(calibrator, calib_path)
        calib_sha = compute_sha256(calib_path)

        # 4. Apply Calibration
        val_cal = calibrator.calibrate(val_raw)
        test_cal = calibrator.calibrate(test_raw)

        # 5. Derive Threshold
        best_tau = 0.50
        best_spec = 0.0
        y_val = val_features["anemia_label"].values
        for tau in np.linspace(0.10, 0.90, 81):
            m = compute_diagnostic_metrics(y_val, val_cal, threshold=tau)
            if m["sensitivity"] >= 0.90 and m["specificity"] >= best_spec:
                best_spec = m["specificity"]
                best_tau = float(tau)

        # 6. Evaluate
        y_test = test_features["anemia_label"].values
        test_img_metrics = compute_diagnostic_metrics(y_test, test_cal, threshold=best_tau)

        return {
            "fusion_sha256": fusion_sha,
            "calibrator_sha256": calib_sha,
            "locked_threshold": round(best_tau, 4),
            "test_image_metrics": test_img_metrics,
        }
