"""
Model metadata and lineage information route.
"""

from typing import Any, Dict
from fastapi import APIRouter
from anemia_ai.schemas.responses import ModelInfoResponse
from anemia_ai.services.assessment_service import get_assessment_service

router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info() -> Dict[str, Any]:
    """Returns technical metadata, lineage, calibration version, and clinical disclaimer."""
    service = get_assessment_service()
    return service.get_model_info()
