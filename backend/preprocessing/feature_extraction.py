"""
Backward-compatibility facade for feature extraction.
Re-exports from anemia_ai.preprocessing.feature_extraction.
"""

from anemia_ai.preprocessing.feature_extraction import (
    FEATURE_NAMES,
    extract_feature_dict,
    extract_features,
    extract_jetx_27_features,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "extract_jetx_27_features",
    "extract_feature_dict",
]
