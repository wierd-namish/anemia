"""
Unit tests for Out-of-Distribution (OOD) rejection on uploaded images.
Verifies that non-nail objects (desk, clothing, wood, random noise) return INCONCLUSIVE.
"""

import sys
import unittest
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

from anemia_ai.api.app import app

class TestOODUpload(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def _post_image(self, img: Image.Image):
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return self.client.post("/predict", files={"file": ("test.jpg", buf, "image/jpeg")})

    def test_pure_green_cloth(self):
        # Non-physiological green cloth
        img = Image.new("RGB", (224, 224), color=(30, 180, 50))
        resp = self._post_image(img)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("state"), "INCONCLUSIVE", "Green cloth must be rejected as INCONCLUSIVE")
        self.assertIsNone(data.get("probability"))

    def test_pure_blue_denim(self):
        # Non-physiological blue surface
        img = Image.new("RGB", (224, 224), color=(20, 50, 190))
        resp = self._post_image(img)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("state"), "INCONCLUSIVE", "Blue denim must be rejected as INCONCLUSIVE")

    def test_random_noise(self):
        # Pure random noise image
        arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        resp = self._post_image(img)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("state"), "INCONCLUSIVE", "Random noise must be rejected as INCONCLUSIVE")

if __name__ == "__main__":
    unittest.main()
