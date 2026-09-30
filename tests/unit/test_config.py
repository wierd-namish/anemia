"""
Unit tests for configuration and settings.
"""

import os
import unittest
from pathlib import Path

from anemia_ai.config.constants import (
    DEFAULT_DIAGNOSTIC_THRESHOLD,
    ENSEMBLE_VERSION,
    PRIMARY_MODEL_VERSION,
    SECONDARY_MODEL_VERSION,
    STATE_ANEMIA,
    STATE_INCONCLUSIVE,
    STATE_NO_ANEMIA,
)
from anemia_ai.config.settings import AppSettings, get_settings


class TestConfigUnit(unittest.TestCase):
    """Verifies settings loading, environment overrides, and constants."""

    def test_diagnostic_states(self):
        self.assertEqual(STATE_ANEMIA, "ANEMIA")
        self.assertEqual(STATE_NO_ANEMIA, "NO_ANEMIA")
        self.assertEqual(STATE_INCONCLUSIVE, "INCONCLUSIVE")

    def test_version_constants(self):
        self.assertEqual(PRIMARY_MODEL_VERSION, "efficientnet_b0_v002")
        self.assertEqual(SECONDARY_MODEL_VERSION, "JetX-GT/nail-anemia-detector")
        self.assertEqual(ENSEMBLE_VERSION, "ensemble_v003")

    def test_settings_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        self.assertIs(s1, s2)
        self.assertIsInstance(s1.base_dir, Path)

    def test_locked_threshold_loading(self):
        settings = get_settings()
        tau_v002 = settings.load_locked_threshold("v002")
        tau_v003 = settings.load_locked_threshold("v003")
        self.assertIsInstance(tau_v002, float)
        self.assertIsInstance(tau_v003, float)
        self.assertTrue(0.0 < tau_v002 <= 1.0)
        self.assertTrue(0.0 < tau_v003 <= 1.0)


if __name__ == "__main__":
    unittest.main()
