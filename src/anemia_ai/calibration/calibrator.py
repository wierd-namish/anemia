"""
Probability Calibration Algorithms for Clinical Reliability.

Supports:
- Isotonic Regression (Non-parametric piecewise calibration)
- Platt Scaling (Sigmoid logistic calibration)
"""

from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from anemia_ai.core.exceptions import CalibrationError
from anemia_ai.core.interfaces import BaseCalibrator


class ProbabilityCalibrator(BaseCalibrator):
    """Fits and applies probability calibration transformations."""

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        self.calibrator = None
        self.is_fitted = False

    def fit(self, uncalibrated_scores: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fits calibration transformation on independent calibration set."""
        scores = np.asarray(uncalibrated_scores).reshape(-1, 1)
        labels = np.asarray(y_true).astype(int)

        if self.method == "platt":
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(scores, labels)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
            self.calibrator.fit(scores.ravel(), labels)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

        self.is_fitted = True
        return self

    def calibrate(self, uncalibrated_scores: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated scores to calibrated medical probabilities."""
        if not self.is_fitted and self.calibrator is None:
            raise CalibrationError("Calibrator must be fitted or loaded before calling calibrate()")

        scores = np.asarray(uncalibrated_scores)
        if self.method == "platt":
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.predict(scores.ravel())

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves fitted calibrator artifact using joblib."""
        if not self.is_fitted and self.calibrator is None:
            raise CalibrationError("Cannot save unfitted calibrator.")
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "ProbabilityCalibrator":
        """Loads fitted calibrator artifact."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibrator artifact not found at: {path}")
        loaded = joblib.load(path)
        if isinstance(loaded, cls):
            return loaded
        # If raw sklearn object was stored directly
        cal = cls(method="isotonic" if isinstance(loaded, IsotonicRegression) else "platt")
        cal.calibrator = loaded
        cal.is_fitted = True
        return cal
