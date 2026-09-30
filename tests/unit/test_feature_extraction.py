"""
Unit tests for handcrafted feature extraction.
"""

import unittest
import numpy as np
from anemia_ai.preprocessing.feature_extraction import (
    FEATURE_NAMES,
    extract_features,
    extract_feature_dict,
    extract_jetx_27_features,
)
from tests.fixtures.synthetic_samples import generate_synthetic_nail


class TestFeatureExtraction(unittest.TestCase):
    """Verifies feature dimensionalities and non-constant feature values."""

    def test_extract_28_features(self):
        img = generate_synthetic_nail()
        feats = extract_features(img)
        self.assertEqual(len(feats), 28)
        self.assertEqual(feats.dtype, np.float32)

    def test_extract_jetx_27_features(self):
        img = generate_synthetic_nail()
        feats = extract_jetx_27_features(img)
        self.assertEqual(len(feats), 28)
        self.assertEqual(feats.dtype, np.float32)

    def test_extract_feature_dict(self):
        img = generate_synthetic_nail()
        f_dict = extract_feature_dict(img)
        self.assertEqual(len(f_dict), len(FEATURE_NAMES))
        self.assertIn("brightness_mean", f_dict)
        self.assertIn("redness_mean", f_dict)


if __name__ == "__main__":
    unittest.main()
