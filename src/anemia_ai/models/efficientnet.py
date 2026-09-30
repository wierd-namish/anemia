"""
EfficientNet-B0 Deep Learning Model Implementation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from anemia_ai.config.constants import PRIMARY_MODEL_VERSION
from anemia_ai.config.settings import get_settings
from anemia_ai.core.exceptions import ModelInferenceError, ModelLoadError
from anemia_ai.core.interfaces import BaseModel
from anemia_ai.core.logging import logger
from anemia_ai.models.cnn_factory import create_cnn_backbone
from anemia_ai.preprocessing.transforms import get_inference_transform


class EfficientNetB0Model(BaseModel):
    """
    EfficientNet-B0 transfer-learning model for subungual nail anemia assessment.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        settings = get_settings()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else settings.v002_model_path
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = get_inference_transform(target_size=(224, 224))
        self.model: Optional[nn.Module] = None
        self._load_checkpoint()

    @property
    def model_name(self) -> str:
        return "EfficientNet-B0"

    @property
    def model_version(self) -> str:
        return PRIMARY_MODEL_VERSION

    def _load_checkpoint(self) -> None:
        """Instantiates backbone and loads checkpoint weights."""
        if not self.checkpoint_path.exists():
            logger.warning("EfficientNet checkpoint not found at: %s", self.checkpoint_path)
            self.model = None
            return

        try:
            model = create_cnn_backbone(architecture="efficientnet_b0", pretrained=False)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            elif isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt)

            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("EfficientNet-B0 (%s) successfully loaded on %s", self.model_version, self.device)
        except Exception as e:
            logger.error("Failed to load EfficientNet checkpoint: %s", e)
            self.model = None
            raise ModelLoadError(f"Failed to load EfficientNet checkpoint: {e}") from e

    def is_ready(self) -> bool:
        if self.model is None and self.checkpoint_path.exists():
            try:
                self._load_checkpoint()
            except Exception:
                return False
        return self.model is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes forward pass on cropped nail image.

        Returns:
            Dict containing raw_logit, probability, and execution metadata.
        """
        if not self.is_ready():
            raise ModelInferenceError(f"EfficientNet-B0 model is not loaded from {self.checkpoint_path}")

        try:
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                logit = float(out.squeeze().item())
                prob = float(1.0 / (1.0 + np.exp(-logit)))

            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "raw_logit": round(logit, 6),
                "probability": round(prob, 4),
                "device": str(self.device),
            }
        except Exception as e:
            raise ModelInferenceError(f"EfficientNet inference failed: {e}") from e
