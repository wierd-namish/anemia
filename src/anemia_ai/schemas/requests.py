"""
Pydantic schemas for structured API requests.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class GuideBox(BaseModel):
    """Normalized bounding box coordinates (ymin, xmin, ymax, xmax) in [0.0, 1.0]."""
    ymin: float = Field(..., ge=0.0, le=1.0)
    xmin: float = Field(..., ge=0.0, le=1.0)
    ymax: float = Field(..., ge=0.0, le=1.0)
    xmax: float = Field(..., ge=0.0, le=1.0)


class SinglePredictionOptions(BaseModel):
    """Optional parameters for single nail prediction."""
    guide_box: Optional[GuideBox] = None
    return_roi_base64: bool = True
