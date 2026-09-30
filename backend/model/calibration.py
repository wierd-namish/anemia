"""
Backward-compatibility facade for calibration.
Re-exports from anemia_ai.calibration.
"""

from anemia_ai.calibration.calibrator import ProbabilityCalibrator
from anemia_ai.calibration.metrics import compute_calibration_metrics

__all__ = ["ProbabilityCalibrator", "compute_calibration_metrics"]
