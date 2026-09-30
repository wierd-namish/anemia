"""
Unit tests for Isotonic Calibrator v002.
"""

import unittest
import numpy as np
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

class TestCalibrationV002(unittest.TestCase):
    def setUp(self):
        self.calib_path = BASE_DIR / "configs/calibrator_isotonic_v002.joblib"

    def test_calibrator_exists(self):
        self.assertTrue(self.calib_path.exists(), "calibrator_isotonic_v002.joblib must exist")

    def test_monotonic_and_bounded(self):
        calibrator = joblib.load(self.calib_path)
        probs = np.linspace(0.0, 1.0, 100)
        cal_probs = calibrator.calibrate(probs)

        # Check bounds
        self.assertTrue(np.all(cal_probs >= 0.0), "Calibrated probabilities must be >= 0.0")
        self.assertTrue(np.all(cal_probs <= 1.0), "Calibrated probabilities must be <= 1.0")

        # Check non-decreasing (monotonicity of isotonic regression)
        diffs = np.diff(cal_probs)
        self.assertTrue(np.all(diffs >= -1e-7), "Isotonic calibration must be monotonically non-decreasing")

    def test_calibrator_not_constant(self):
        calibrator = joblib.load(self.calib_path)
        cal_low = calibrator.calibrate([0.05])[0]
        cal_high = calibrator.calibrate([0.95])[0]
        self.assertLess(cal_low, cal_high, "Calibrator must not map all inputs to a constant value")

if __name__ == "__main__":
    unittest.main()
