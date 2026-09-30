"""
Core abstractions, exceptions, interfaces, and logging.
"""

from anemia_ai.core.exceptions import (
    AnemiaAIError,
    CalibrationError,
    ConfigurationError,
    ImageQualityError,
    ModelInferenceError,
    ModelLoadError,
    RoiDetectionError,
)
from anemia_ai.core.interfaces import (
    BaseCalibrator,
    BaseDetector,
    BaseModel,
    BaseQualityChecker,
)
from anemia_ai.core.logging import logger, setup_logger

__all__ = [
    "AnemiaAIError",
    "ImageQualityError",
    "RoiDetectionError",
    "ModelInferenceError",
    "ModelLoadError",
    "CalibrationError",
    "ConfigurationError",
    "BaseModel",
    "BaseCalibrator",
    "BaseDetector",
    "BaseQualityChecker",
    "logger",
    "setup_logger",
]
