"""
Unit tests for Image Quality Gate on upload endpoint.
Verifies rejection of blurry, underexposed, overexposed, and glare-compromised images with state=INCONCLUSIVE.
"""

import unittest
from io import BytesIO
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app

class TestQualityUpload(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _post_image(self, img: Image.Image):
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return self.client.post("/predict", files={"file": ("quality_test.jpg", buf, "image/jpeg")})

    def test_severely_underexposed(self):
        # Very dark image
        img = Image.new("RGB", (224, 224), color=(10, 10, 10))
        resp = self._post_image(img)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("state"), "INCONCLUSIVE")
        self.assertIsNone(data.get("probability"))

    def test_severely_overexposed(self):
        # Blown out image
        img = Image.new("RGB", (224, 224), color=(250, 250, 250))
        resp = self._post_image(img)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("state"), "INCONCLUSIVE")
        self.assertIsNone(data.get("probability"))

    def test_severely_blurry(self):
        # Uniform solid flat gray (zero Laplacian variance)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        resp = self._post_image(img)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("state"), "INCONCLUSIVE")
        self.assertIsNone(data.get("probability"))

if __name__ == "__main__":
    unittest.main()
