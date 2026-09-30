"""
API route modules.
"""

from anemia_ai.api.routes.health import router as health_router
from anemia_ai.api.routes.model_info import router as model_info_router
from anemia_ai.api.routes.predict import router as predict_router

__all__ = ["health_router", "model_info_router", "predict_router"]
