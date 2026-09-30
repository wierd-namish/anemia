"""
Model Smoke Tests and Input Dependence Verification.
Ensures models are truly active and outputs vary strictly with image input.
"""

import unittest
from PIL import Image
import numpy as np

from anemia_ai.inference.pipeline import TwoModelEnsembleService, DiagnosticInferenceService
from anemia_ai.models.efficientnet import EfficientNetB0Model
from anemia_ai.models.jetx_nail import JetXNailModel
from tests.fixtures.synthetic_samples import generate_synthetic_nail


class TestModelSmoke(unittest.TestCase):
    """Verifies that models load properly and output input-dependent predictions."""

    @classmethod
    def setUpClass(cls):
        cls.ensemble = TwoModelEnsembleService()
        cls.diagnostic = DiagnosticInferenceService()

    def test_ensemble_is_ready(self):
        self.assertTrue(self.ensemble.is_ready())

    def test_diagnostic_is_ready(self):
        self.assertTrue(self.diagnostic.is_ready())

    def test_input_dependence_different_images_produce_different_scores(self):
        """Validates that distinct input images generate distinct numerical scores."""
        nail1 = generate_synthetic_nail(is_anemic=False)
        nail2 = generate_synthetic_nail(is_anemic=True)

        res1 = self.ensemble.predict_single(nail1)
        res2 = self.ensemble.predict_single(nail2)

        self.assertTrue(res1["success"])
        self.assertTrue(res2["success"])
        self.assertIsNotNone(res1["probability"])
        self.assertIsNotNone(res2["probability"])

        # Compare probabilities and raw features
        p1 = res1["probability"]
        p2 = res2["probability"]
        logit1 = res1["efficientnet_raw_logit"]
        logit2 = res2["efficientnet_raw_logit"]

        self.assertNotEqual(
            logit1,
            logit2,
            "Model output logit should not be constant across different images!",
        )

    def test_primary_efficientnet_standalone(self):
        model = EfficientNetB0Model()
        self.assertTrue(model.is_ready())
        img = generate_synthetic_nail(is_anemic=False)
        out = model.predict(img)
        self.assertIn("raw_logit", out)
        self.assertIn("probability", out)
        self.assertTrue(0.0 <= out["probability"] <= 1.0)

    def test_secondary_jetx_standalone(self):
        model = JetXNailModel()
        self.assertTrue(model.is_ready())
        img = generate_synthetic_nail(is_anemic=False)
        out = model.predict(img)
        self.assertIn("raw_logit", out)
        self.assertIn("probability", out)
        self.assertTrue(0.0 <= out["probability"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
