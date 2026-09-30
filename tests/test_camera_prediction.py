"""
Unit tests for Camera/Single-Image Prediction endpoint.
"""

import unittest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

from anemia_ai.api.app import app

class TestCameraPrediction(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_camera_post_predict(self):
        img = Image.new("RGB", (224, 224), color=(210, 150, 130))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("camera_frame.jpg", buf, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("success", data)
        self.assertIn("state", data)
        self.assertIn(data.get("primary_model"), ["efficientnet_b0_v002", None])
        self.assertIn(data.get("model_version"), ["efficientnet_b0_v002", "ensemble_v003"])

if __name__ == "__main__":
    unittest.main()
