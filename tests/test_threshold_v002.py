"""
Unit tests for Locked Diagnostic Threshold v002.
"""

import unittest
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class TestThresholdV002(unittest.TestCase):
    def setUp(self):
        self.tau_path = BASE_DIR / "configs/locked_tau_v002.json"

    def test_threshold_config_exists(self):
        self.assertTrue(self.tau_path.exists(), "locked_tau_v002.json must exist")

    def test_threshold_properties(self):
        with open(self.tau_path, "r") as f:
            data = json.load(f)

        self.assertEqual(data.get("model_version"), "efficientnet_b0_v002")
        self.assertEqual(data.get("calibration_version"), "isotonic_regression_v002")
        tau = data.get("locked_threshold")
        self.assertIsNotNone(tau)
        self.assertGreater(tau, 0.0)
        self.assertLess(tau, 1.0)
        self.assertEqual(data.get("derived_on"), "validation_split")

        val_metrics = data.get("validation_metrics", {})
        self.assertIn("sensitivity", val_metrics)
        self.assertIn("specificity", val_metrics)

if __name__ == "__main__":
    unittest.main()
