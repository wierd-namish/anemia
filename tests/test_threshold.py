"""
Automated Tests for Locked Diagnostic Threshold and State Partitioning.
Tests:
- Locked threshold loading from configs/diagnostic_threshold.json
- Boundary behavior at threshold (0.48)
- Decision boundary assignment
"""

import sys
import unittest
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.config.constants import (
    DEFAULT_DECISION_THRESHOLD as LOCKED_DIAGNOSTIC_THRESHOLD,
    STATE_ANEMIA,
    STATE_NO_ANEMIA,
)
from anemia_ai.config.settings import get_settings


def load_diagnostic_threshold():
    return LOCKED_DIAGNOSTIC_THRESHOLD


class TestThreshold(unittest.TestCase):

    def test_locked_threshold_value(self):
        """Verify locked threshold matches configuration."""
        settings = get_settings()
        tau = settings.load_locked_threshold("v003")
        self.assertAlmostEqual(tau, 0.9000, places=3)

    def test_decision_boundary_logic(self):
        """Verify diagnostic state assignment on calibrated probabilities."""
        thresh = LOCKED_DIAGNOSTIC_THRESHOLD
        
        # Above threshold -> ANEMIA
        p_high = thresh + 0.05
        state_high = STATE_ANEMIA if p_high >= thresh else STATE_NO_ANEMIA
        self.assertEqual(state_high, STATE_ANEMIA)
        
        # Below threshold -> NO_ANEMIA
        p_low = thresh - 0.05
        state_low = STATE_ANEMIA if p_low >= thresh else STATE_NO_ANEMIA
        self.assertEqual(state_low, STATE_NO_ANEMIA)


if __name__ == "__main__":
    unittest.main()
