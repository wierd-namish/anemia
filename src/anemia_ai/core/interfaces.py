"""
Abstract base classes and interfaces for modular extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image


class BaseModel(ABC):
    """Abstract base class for all single or ensemble anemia models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model architecture."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version identifier of the model."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Returns True if the model and required artifacts are loaded and ready."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes model inference on a single normalized PIL Image.
        Returns dictionary containing raw logit and sigmoid probability.
        """
        pass


class BaseCalibrator(ABC):
    """Abstract interface for probability calibration algorithms."""

    @abstractmethod
    def fit(self, uncalibrated_scores: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        """Fits calibration curve on independent calibration partition."""
        pass

    @abstractmethod
    def calibrate(self, uncalibrated_scores: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated model scores into calibrated medical probabilities."""
        pass


class BaseDetector(ABC):
    """Abstract interface for fingernail region-of-interest detection."""

    @abstractmethod
    def detect_and_crop(
        self,
        img: Image.Image,
        guide_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Image.Image, Tuple[int, int, int, int], Dict[str, Any]]:
        """Detects and crops subungual nail region."""
        pass


class BaseQualityChecker(ABC):
    """Abstract interface for pre-inference image quality assurance."""

    @abstractmethod
    def assess(self, image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluates image for blur, exposure, glare, resolution, and foreign pigments."""
        pass
