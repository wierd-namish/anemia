"""
Baseline Fingernail Anemia Model Wrapper.
Legacy 28-feature model implementation for historical provenance and testing.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import joblib
import numpy as np
from PIL import Image

from anemia_ai.config.constants import (
    BASELINE_BALANCED_THRESHOLD,
    PREPROCESSING_VERSION,
    PRIMARY_MODEL_VERSION,
    THRESHOLD_V002_VERSION,
)
from anemia_ai.config.settings import get_settings
from anemia_ai.core.interfaces import BaseModel
from anemia_ai.preprocessing.feature_extraction import extract_feature_dict, extract_features
from anemia_ai.preprocessing.image_quality import assess_image_quality


class BaselineNailAnemiaModel(BaseModel):
    """
    Baseline ML model using handcrafted 28 color/pallor features and MLPClassifier.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        settings = get_settings()
        cache = settings.jetx_cache_dir
        self.model_path = Path(model_path) if model_path else cache / "mlp_model.joblib"
        self.scaler_path = Path(scaler_path) if scaler_path else cache / "feature_scaler.joblib"
        self.metadata_path = Path(metadata_path) if metadata_path else cache / "model_metadata.json"

        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = 28
        self._load_artifacts()

    @property
    def model_name(self) -> str:
        return "Baseline MLP Model"

    @property
    def model_version(self) -> str:
        return "baseline_v001"

    def _load_artifacts(self) -> None:
        """Loads pretrained model weights and scaler."""
        if not self.model_path.exists() or not self.scaler_path.exists():
            return

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                pass

        self.classes_ = getattr(self.model, "classes_", np.array([0, 1]))
        self.n_features_in_ = getattr(self.model, "n_features_in_", 28)

    def is_ready(self) -> bool:
        if self.model is None or self.scaler is None:
            self._load_artifacts()
        return self.model is not None and self.scaler is not None

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Runs standard feature extraction and returns probability."""
        if not self.is_ready():
            raise RuntimeError("Baseline model is not loaded.")

        feat_vec = extract_features(image)
        feat_scaled = self.scaler.transform(feat_vec.reshape(1, -1))
        probabilities = self.model.predict_proba(feat_scaled)[0]
        raw_score = float(probabilities[1])

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "probability": round(raw_score, 4),
        }

    def predict_image(
        self,
        img: Image.Image,
        threshold: float = BASELINE_BALANCED_THRESHOLD,
        check_quality: bool = True,
    ) -> Dict[str, Any]:
        """Comprehensive legacy inference interface for acceptance tests."""
        if check_quality:
            passed, msg, metrics = assess_image_quality(img)
            if not passed:
                return {
                    "success": False,
                    "error": "image_quality_failed",
                    "message": msg,
                    "quality_metrics": metrics,
                    "diagnosis": None,
                    "model_score": None,
                    "is_calibrated": False,
                    "disclaimer": "Unable to make a reliable assessment due to image quality.",
                }

        feat_vec = extract_features(img)
        feat_dict = extract_feature_dict(img)
        feat_scaled = self.scaler.transform(feat_vec.reshape(1, -1))
        probabilities = self.model.predict_proba(feat_scaled)[0]
        raw_score = float(probabilities[1])

        is_positive = raw_score >= threshold
        diagnosis = "ANEMIA" if is_positive else "NO ANEMIA DETECTED"

        margin = abs(raw_score - threshold)
        if margin > 0.30:
            confidence = "HIGH"
        elif margin > 0.15:
            confidence = "MODERATE"
        else:
            confidence = "LOW"

        explanation = (
            "The model identified nail-image characteristics (pallor and reduced vascular redness) "
            "associated with anemia."
            if is_positive
            else "The model did not identify a sufficiently strong anemia-associated pallor pattern in the provided nail image."
        )

        return {
            "success": True,
            "diagnosis": diagnosis,
            "is_anemic_prediction": bool(is_positive),
            "model_score": round(raw_score, 4),
            "calibrated_probability": None,
            "is_calibrated": False,
            "confidence": confidence,
            "threshold_used": threshold,
            "explanation": explanation,
            "model_version": self.model_version,
            "preprocessing_version": PREPROCESSING_VERSION,
            "threshold_version": THRESHOLD_V002_VERSION,
            "quality_metrics": {},
            "feature_summary": {
                "brightness_mean": feat_dict["brightness_mean"],
                "redness_mean": feat_dict["redness_mean"],
                "white_ratio": feat_dict["white_ratio"],
                "pink_ratio": feat_dict["pink_ratio"],
                "hb_proxy_mean": feat_dict["hb_proxy_mean"],
            },
            "disclaimer": (
                "MEDICAL DISCLAIMER: This is a screening tool score and not a definitive clinical diagnosis. "
                "All results must be confirmed with a standard laboratory blood test (hemoglobin/CBC)."
            ),
        }
