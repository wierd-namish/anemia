"""
Unit tests for Image Upload Endpoint (POST /predict with multipart/form-data).
Verifies:
- Accepts JPG, PNG, WEBP files
- Validates MIME type
- Rejects non-image files with HTTP 400
- Returns structured response with model metadata
"""

import unittest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app

class TestUploadEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_upload_valid_jpeg(self):
        img = Image.new("RGB", (224, 224), color=(200, 140, 120))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("nail_test.jpg", buf, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("state", data)
        self.assertIn(data.get("model_version"), ["efficientnet_b0_v002", "ensemble_v003"])

    def test_upload_valid_png(self):
        img = Image.new("RGB", (224, 224), color=(180, 120, 100))
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = self.client.post(
            "/predict",
            files={"file": ("nail_test.png", buf, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("state", data)

    def test_upload_non_image_rejection(self):
        buf = BytesIO(b"Not an image text data")
        response = self.client.post(
            "/predict",
            files={"file": ("document.pdf", buf, "application/pdf")}
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_corrupt_image_rejection(self):
        buf = BytesIO(b"\xFF\xD8\xFF\xE0corrupt data")
        response = self.client.post(
            "/predict",
            files={"file": ("corrupt.jpg", buf, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
