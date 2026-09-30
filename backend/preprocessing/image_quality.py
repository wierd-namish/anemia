"""
Backward-compatibility facade for image quality assessment.
Re-exports from anemia_ai.preprocessing.image_quality.
"""

from anemia_ai.preprocessing.image_quality import (
    ImageQualityChecker,
    assess_image_quality,
)

__all__ = ["assess_image_quality", "ImageQualityChecker"]
