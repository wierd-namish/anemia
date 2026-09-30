"""
Backward-compatibility facade for ensemble pipeline.
Re-exports TwoModelEnsembleService and constants from anemia_ai.inference.
"""

from anemia_ai.config.constants import (
    CALIBRATION_V003_VERSION,
    ENSEMBLE_VERSION,
    PRIMARY_MODEL_VERSION,
    SECONDARY_MODEL_VERSION,
    THRESHOLD_V003_VERSION,
)
from anemia_ai.inference.pipeline import TwoModelEnsembleService

__all__ = [
    "TwoModelEnsembleService",
    "PRIMARY_MODEL_VERSION",
    "SECONDARY_MODEL_VERSION",
    "ENSEMBLE_VERSION",
    "CALIBRATION_V003_VERSION",
    "THRESHOLD_V003_VERSION",
]
