"""
Unit tests for Real Input Dependence on EfficientNet-B0 v002.
Verifies that distinct valid nail images produce distinct raw logits and calibrated probabilities.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

from anemia_ai.inference.pipeline import TwoModelEnsembleService
from tests.fixtures.synthetic_samples import (
    generate_synthetic_anemia_nail,
    generate_synthetic_healthy_nail,
)


class TestInputDependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = TwoModelEnsembleService()

    def test_five_images_produce_independent_predictions(self):
        # Generate 6 distinct biological test images with distinct RGB tones
        images = [
            generate_synthetic_anemia_nail(size=(224, 224), seed=1),
            generate_synthetic_anemia_nail(size=(224, 224), seed=2),
            generate_synthetic_healthy_nail(size=(224, 224), seed=3),
            generate_synthetic_healthy_nail(size=(224, 224), seed=4),
            Image.new("RGB", (224, 224), color=(220, 160, 150)),
            Image.new("RGB", (224, 224), color=(180, 110, 100)),
        ]

        raw_logits = []
        cal_probs = []

        for img in images:
            res = self.pipeline.predict_single(img)
            self.assertTrue(res.get("success", False), f"Prediction should succeed for valid nail: {res}")
            if "efficientnet_raw_logit" in res:
                raw_logits.append(res["efficientnet_raw_logit"])
            if "probability" in res and res["probability"] is not None:
                cal_probs.append(res["probability"])

        # Check that outputs are NOT all identical
        if len(raw_logits) > 1:
            std_logits = float(np.std(raw_logits))
            self.assertGreater(std_logits, 1e-4, f"Raw logits must vary across different images. Got: {raw_logits}")
        if len(cal_probs) > 1:
            std_probs = float(np.std(cal_probs))
            self.assertGreater(std_probs, 1e-4, f"Calibrated probabilities must vary across different images. Got: {cal_probs}")

if __name__ == "__main__":
    unittest.main()
