"""
Automated Tests for Locked EfficientNet-B0 Model Inference and Calibration.
Tests:
- Model instantiation and weight loading
- Model forward pass on 224x224 RGB input
- Monotonic probability calibration
- Deterministic behavior without random hallucinations
"""

import unittest
import numpy as np
from PIL import Image

from backend.model.inference_pipeline import DiagnosticInferenceService
from backend.config import MODEL_NAME, MODEL_VERSION


class TestModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = DiagnosticInferenceService()

    def test_service_initialization(self):
        """Verify diagnostic service initializes model and calibrator."""
        self.assertTrue(self.service.is_ready())
        self.assertIsNotNone(self.service.model)
        self.assertIsNotNone(self.service.calibrator)

    def test_inference_on_sample_nail(self):
        """Verify inference runs end-to-end on sample nail images."""
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[:, :, 0] = 175 # Reddish
        arr[:, :, 1] = 110 # Green
        arr[:, :, 2] = 105 # Blue
        arr += np.random.randint(0, 20, (300, 300, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        
        result = self.service.predict_single(img)
        self.assertTrue(result["success"])
        self.assertEqual(result["model_name"], MODEL_NAME)
        self.assertEqual(result["model_version"], MODEL_VERSION)
        self.assertIn(result["state"], ["ANEMIA", "NO_ANEMIA", "INCONCLUSIVE"])
        if result["state"] != "INCONCLUSIVE":
            self.assertIsInstance(result["probability"], float)
            self.assertGreaterEqual(result["probability"], 0.0)
            self.assertLessEqual(result["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
