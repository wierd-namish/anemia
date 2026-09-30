"""
Hugging Face JetX-GT/nail-anemia-detector Model Implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import urllib.request
import joblib
import numpy as np
from PIL import Image

from anemia_ai.config.constants import SECONDARY_MODEL_VERSION
from anemia_ai.config.settings import get_settings
from anemia_ai.core.exceptions import ModelInferenceError, ModelLoadError
from anemia_ai.core.interfaces import BaseModel
from anemia_ai.core.logging import logger
from anemia_ai.preprocessing.feature_extraction import extract_jetx_27_features
from anemia_ai.utils.hashing import compute_sha256

HF_REPO: str = "JetX-GT/nail-anemia-detector"
HF_BASE_URL: str = "https://huggingface.co/JetX-GT/nail-anemia-detector/resolve/main"

FILES_TO_DOWNLOAD = {
    "mlp_model.joblib": f"{HF_BASE_URL}/mlp_model.joblib",
    "feature_scaler.joblib": f"{HF_BASE_URL}/feature_scaler.joblib",
    "model_metadata.json": f"{HF_BASE_URL}/model_metadata.json",
}


def ensure_jetx_artifacts_downloaded(cache_dir: Path) -> Dict[str, str]:
    """Downloads official Hugging Face JetX-GT model artifacts if not present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for fname, url in FILES_TO_DOWNLOAD.items():
        dst = cache_dir / fname
        if not dst.exists() or dst.stat().st_size == 0:
            logger.info("Downloading %s from Hugging Face (%s)...", fname, url)
            urllib.request.urlretrieve(url, dst)
        h = compute_sha256(dst)
        hashes[fname] = h
        logger.debug("JetX artifact %s verified (SHA256: %s...)", fname, h[:16])
    return hashes


class JetXNailModel(BaseModel):
    """
    Hugging Face JetX-GT nail anemia detector (27 handcrafted color features + MLP).
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        settings = get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else settings.jetx_cache_dir
        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.hashes: Dict[str, str] = {}
        self._load()

    @property
    def model_name(self) -> str:
        return "JetX-GT Nail Anemia Detector"

    @property
    def model_version(self) -> str:
        return SECONDARY_MODEL_VERSION

    def _load(self) -> None:
        """Downloads/loads and validates the MLP model and scaler."""
        try:
            self.hashes = ensure_jetx_artifacts_downloaded(self.cache_dir)
            mlp_path = self.cache_dir / "mlp_model.joblib"
            scaler_path = self.cache_dir / "feature_scaler.joblib"
            meta_path = self.cache_dir / "model_metadata.json"

            if mlp_path.exists() and scaler_path.exists():
                self.model = joblib.load(mlp_path)
                self.scaler = joblib.load(scaler_path)

                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)

                logger.info("JetX-GT model loaded successfully from %s", self.cache_dir)
            else:
                logger.warning("JetX-GT model artifacts missing in %s", self.cache_dir)
        except Exception as e:
            logger.error("Failed to load JetX model: %s", e)
            self.model = None
            self.scaler = None
            raise ModelLoadError(f"Failed to load JetX model: {e}") from e

    def is_ready(self) -> bool:
        if (self.model is None or self.scaler is None) and self.cache_dir.exists():
            try:
                self._load()
            except Exception:
                return False
        return self.model is not None and self.scaler is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extracts 27 color features and computes MLP probability and approximate logit.
        """
        if not self.is_ready():
            raise ModelInferenceError("JetX-GT model is not loaded.")

        try:
            features = extract_jetx_27_features(image)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            proba = float(self.model.predict_proba(features_scaled)[0, 1])

            # Invert sigmoid to obtain approximate logit: log(p / (1-p))
            eps = 1e-6
            p_clipped = np.clip(proba, eps, 1.0 - eps)
            logit = float(np.log(p_clipped / (1.0 - p_clipped)))

            return {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "probability": round(proba, 4),
                "raw_logit": round(logit, 6),
                "features_27": features.tolist(),
                "model_repo": HF_REPO,
                "hashes": self.hashes,
            }
        except Exception as e:
            raise ModelInferenceError(f"JetX-GT inference failed: {e}") from e


# Backward compatibility alias
JetXNailAnemiaDetector = JetXNailModel
