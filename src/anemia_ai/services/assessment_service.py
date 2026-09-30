"""
Application Service Orchestrator for Anemia Assessment.
"""

from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch

from anemia_ai.config.constants import (
    CALIBRATION_V003_VERSION,
    CLINICAL_POPULATION_NOTE,
    ENSEMBLE_VERSION,
    PERSISTENT_CLINICAL_DISCLAIMER,
    PREPROCESSING_VERSION,
    PRIMARY_MODEL_VERSION,
    SECONDARY_MODEL_VERSION,
    THRESHOLD_V003_VERSION,
)
from anemia_ai.config.settings import get_settings
from anemia_ai.core.logging import logger
from anemia_ai.inference.pipeline import TwoModelEnsembleService


class AssessmentService:
    """Singleton service for application-level operations and metadata."""

    def __init__(self):
        self.settings = get_settings()
        self.ensemble = TwoModelEnsembleService()

    def get_hardware_info(self) -> Dict[str, Any]:
        """Returns execution device and GPU details."""
        if torch.cuda.is_available():
            return {
                "device": "cuda",
                "gpu": torch.cuda.get_device_name(0),
                "cuda_version": str(torch.version.cuda),
            }
        return {
            "device": "cpu",
            "gpu": "None",
            "cuda_version": "None",
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Returns system health information."""
        hw = self.get_hardware_info()
        return {
            "status": "healthy",
            "model_loaded": self.ensemble.is_ready(),
            "model_name": "EfficientNet-B0 + JetX-GT Ensemble",
            "model_version": ENSEMBLE_VERSION,
            "primary_model": PRIMARY_MODEL_VERSION,
            "secondary_model": SECONDARY_MODEL_VERSION,
            "device": hw["device"],
            "gpu": hw["gpu"],
            "api_version": self.settings.api_version,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Returns detailed model lineage and clinical parameters."""
        hw = self.get_hardware_info()
        return {
            "model": "EfficientNet-B0 + JetX-GT Ensemble",
            "primary_model": PRIMARY_MODEL_VERSION,
            "secondary_model": SECONDARY_MODEL_VERSION,
            "fusion_version": ENSEMBLE_VERSION,
            "calibrator_version": CALIBRATION_V003_VERSION,
            "threshold_version": THRESHOLD_V003_VERSION,
            "threshold": self.ensemble.threshold,
            "locked_threshold": self.ensemble.threshold,
            "device": hw["device"],
            "gpu": hw["gpu"],
            "cuda_version": hw["cuda_version"],
            "input_type": "Fingernail ROI (224x224 RGB)",
            "preprocessing": PREPROCESSING_VERSION,
            "clinical_development_population": CLINICAL_POPULATION_NOTE,
            "disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
            "research_status_disclaimer": PERSISTENT_CLINICAL_DISCLAIMER,
        }

    def assess_single_nail(
        self,
        image: Image.Image,
        guide_box: Optional[Tuple[float, float, float, float]] = None,
        return_roi_base64: bool = True,
    ) -> Dict[str, Any]:
        """Runs single-nail assessment."""
        return self.ensemble.predict_single(
            image=image,
            guide_box=guide_box,
            return_roi_base64=return_roi_base64,
        )

    def assess_multiple_nails(self, images: List[Image.Image]) -> Dict[str, Any]:
        """Runs multi-nail assessment."""
        return self.ensemble.predict_multiple(images)


_assessment_service_instance: Optional[AssessmentService] = None


def get_assessment_service() -> AssessmentService:
    global _assessment_service_instance
    if _assessment_service_instance is None:
        _assessment_service_instance = AssessmentService()
    return _assessment_service_instance
