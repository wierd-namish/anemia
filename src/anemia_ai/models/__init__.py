"""
Models package for deep vision, handcrafted, and ensemble models.
"""

from anemia_ai.models.base import BaseModel
from anemia_ai.models.baseline import BaselineNailAnemiaModel
from anemia_ai.models.cnn_factory import create_cnn_backbone
from anemia_ai.models.efficientnet import EfficientNetB0Model
from anemia_ai.models.jetx_nail import JetXNailAnemiaDetector, JetXNailModel
from anemia_ai.models.registry import get_model, list_registered_models, register_model

__all__ = [
    "BaseModel",
    "EfficientNetB0Model",
    "JetXNailModel",
    "JetXNailAnemiaDetector",
    "BaselineNailAnemiaModel",
    "create_cnn_backbone",
    "get_model",
    "register_model",
    "list_registered_models",
]
