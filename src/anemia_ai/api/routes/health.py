"""
Health check route.
"""

from typing import Any, Dict
from fastapi import APIRouter
from anemia_ai.schemas.responses import HealthResponse
from anemia_ai.services.assessment_service import get_assessment_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> Dict[str, Any]:
    """System health, device availability, and model readiness check."""
    service = get_assessment_service()
    return service.get_health_status()
