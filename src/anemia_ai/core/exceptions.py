"""
Domain-specific exception hierarchy for Anemia AI.
"""

from typing import Any, Dict, Optional


class AnemiaAIError(Exception):
    """Base exception for all Anemia AI runtime errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ImageQualityError(AnemiaAIError):
    """Raised when an input image fails medical quality standards."""
    pass


class RoiDetectionError(AnemiaAIError):
    """Raised when nail region of interest cannot be extracted or validated."""
    pass


class ModelInferenceError(AnemiaAIError):
    """Raised during model execution or score calculation."""
    pass


class ModelLoadError(AnemiaAIError):
    """Raised when model weights or artifacts fail to load or verify."""
    pass


class CalibrationError(AnemiaAIError):
    """Raised during probability calibration fitting or transformation."""
    pass


class ConfigurationError(AnemiaAIError):
    """Raised when required configuration or artifact is missing or invalid."""
    pass
