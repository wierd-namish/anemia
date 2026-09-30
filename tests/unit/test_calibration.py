"""
Unit tests for probability calibration and calibration metrics.
"""

import unittest
import numpy as np
from anemia_ai.calibration.calibrator import ProbabilityCalibrator
from anemia_ai.calibration.metrics import compute_calibration_metrics


class TestCalibration(unittest.TestCase):
    """Verifies calibration fitting, monotone probability mapping, and ECE calculation."""

    def test_isotonic_calibration_fit_and_transform(self):
        np.random.seed(42)
        raw_scores = np.linspace(0.1, 0.9, 50)
        labels = (raw_scores > 0.5).astype(int)

        calibrator = ProbabilityCalibrator(method="isotonic")
        calibrator.fit(raw_scores, labels)
        calibrated = calibrator.calibrate(raw_scores)

        self.assertEqual(len(calibrated), len(raw_scores))
        self.assertTrue(np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0))
        # Verify monotonicity
        self.assertTrue(np.all(np.diff(calibrated) >= 0.0))

    def test_platt_calibration_fit_and_transform(self):
        np.random.seed(42)
        raw_scores = np.linspace(-2.0, 2.0, 50)
        labels = (raw_scores > 0.0).astype(int)

        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(raw_scores, labels)
        calibrated = calibrator.calibrate(raw_scores)

        self.assertEqual(len(calibrated), len(raw_scores))
        self.assertTrue(np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0))

    def test_compute_calibration_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        metrics = compute_calibration_metrics(y_true, probs, n_bins=5)

        self.assertIn("brier_score", metrics)
        self.assertIn("expected_calibration_error", metrics)
        self.assertLess(metrics["brier_score"], 0.1)


if __name__ == "__main__":
    unittest.main()
