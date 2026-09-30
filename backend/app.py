"""
Backward-compatibility facade for FastAPI application.
Re-exports the application from anemia_ai.api.
"""

from anemia_ai.api.app import app
from anemia_ai.api.routes.health import health_check
from anemia_ai.api.routes.model_info import get_model_info
from anemia_ai.api.routes.predict import predict_multiple_nails, predict_single_nail

__all__ = [
    "app",
    "health_check",
    "get_model_info",
    "predict_single_nail",
    "predict_multiple_nails",
]
