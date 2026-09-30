"""
Backward-compatibility facade for JetX model.
Re-exports from anemia_ai.models.jetx_nail.
"""

from anemia_ai.models.jetx_nail import (
    FILES_TO_DOWNLOAD,
    HF_BASE_URL,
    HF_REPO,
    JetXNailAnemiaDetector,
    JetXNailModel,
    ensure_jetx_artifacts_downloaded,
)
from anemia_ai.preprocessing.feature_extraction import extract_jetx_27_features
from anemia_ai.utils.hashing import compute_sha256

__all__ = [
    "JetXNailAnemiaDetector",
    "JetXNailModel",
    "extract_jetx_27_features",
    "ensure_jetx_artifacts_downloaded",
    "compute_sha256",
    "HF_REPO",
    "HF_BASE_URL",
    "FILES_TO_DOWNLOAD",
]
