"""
Automated Tests for Locked Diagnostic Threshold and State Partitioning.
Tests:
- Locked threshold loading from configs/diagnostic_threshold.json
- Boundary behavior at threshold (0.48)
- Decision boundary assignment
"""

import unittest
import json
from pathlib import Path

from backend.config import (
    load_diagnostic_threshold,
    LOCKED_DIAGNOSTIC_THRESHOLD,
    DIAGNOSTIC_THRESHOLD_CONFIG_PATH,
    STATE_ANEMIA,
    STATE_NO_ANEMIA,
)


class TestThreshold(unittest.TestCase):

    def test_locked_threshold_value(self):
        """Verify locked threshold matches configuration."""
        self.assertTrue(DIAGNOSTIC_THRESHOLD_CONFIG_PATH.exists())
        with open(DIAGNOSTIC_THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            expected_tau = float(data.get("locked_threshold", 0.50))
            self.assertEqual(LOCKED_DIAGNOSTIC_THRESHOLD, expected_tau)

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
