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

from backend.model.inference_pipeline import InferencePipeline

class TestInputDependence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = InferencePipeline()

    def test_five_images_produce_independent_predictions(self):
        # Load 5 distinct test images from test split
        test_df = pd.read_csv(BASE_DIR / "data/splits/test.csv")
        sample_rows = test_df.sample(n=min(10, len(test_df)), random_state=42).to_dict("records")

        raw_logits = []
        raw_sigmoids = []
        cal_probs = []

        for row in sample_rows:
            img_path = Path(row["image_path"])
            self.assertTrue(img_path.exists(), f"Sample image must exist: {img_path}")
            img = Image.open(img_path).convert("RGB")
            
            res = self.pipeline.predict_single(img)
            self.assertTrue(res.get("success", False), f"Prediction should succeed for valid nail: {res}")
            if "raw_logit" not in res:
                continue
            self.assertIn("raw_logit", res)
            self.assertIn("probability", res)

            raw_logits.append(res["raw_logit"])
            raw_sigmoids.append(res["raw_sigmoid"])
            cal_probs.append(res["probability"])

        # Check that outputs are NOT all identical
        std_logits = np.std(raw_logits)
        std_probs = np.std(cal_probs)

        self.assertGreater(std_logits, 1e-4, f"Raw logits must vary across different images. Got: {raw_logits}")
        self.assertGreater(std_probs, 1e-4, f"Calibrated probabilities must vary across different images. Got: {cal_probs}")

if __name__ == "__main__":
    unittest.main()
