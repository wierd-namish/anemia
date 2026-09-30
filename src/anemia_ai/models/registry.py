"""
Model Registry and Factory.
Enables pluggable addition of future AI models without restructuring inference pipelines.
"""

from typing import Callable, Dict, List, Type
from anemia_ai.core.interfaces import BaseModel
from anemia_ai.models.baseline import BaselineNailAnemiaModel
from anemia_ai.models.efficientnet import EfficientNetB0Model
from anemia_ai.models.jetx_nail import JetXNailModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "efficientnet_b0": EfficientNetB0Model,
    "efficientnet_b0_v002": EfficientNetB0Model,
    "jetx_gt": JetXNailModel,
    "jetx_nail": JetXNailModel,
    "baseline": BaselineNailAnemiaModel,
}


def register_model(name: str, model_cls: Type[BaseModel]) -> None:
    """Registers a new model architecture class."""
    MODEL_REGISTRY[name.lower()] = model_cls


def get_model(name: str, **kwargs) -> BaseModel:
    """Instantiates a model by registered name."""
    name_clean = name.lower().strip()
    if name_clean not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name_clean](**kwargs)


def list_registered_models() -> List[str]:
    """Returns a list of all registered model identifiers."""
    return sorted(list(MODEL_REGISTRY.keys()))
