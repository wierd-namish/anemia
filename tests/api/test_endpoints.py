"""
API Integration Tests using FastAPI TestClient.
Tests versioned /api/v1/ routes and root backward-compatibility routes.
"""

import io
import unittest
from fastapi.testclient import TestClient
from PIL import Image

from anemia_ai.api.app import app
from tests.fixtures.synthetic_samples import generate_synthetic_nail, generate_blurry_image


class TestApiEndpoints(unittest.TestCase):
    """Verifies all HTTP endpoints, payload formats, error responses, and headers."""

    def setUp(self):
        self.client = TestClient(app)

    def _image_to_bytes(self, img: Image.Image, fmt="JPEG") -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("X-Request-ID", response.headers)

    def test_root_health_backward_compatibility(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_model_info_endpoint(self):
        response = self.client.get("/api/v1/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("threshold", data)
        self.assertIn("disclaimer", data)
        self.assertIn("primary_model", data)

    def test_predict_single_endpoint(self):
        img = generate_synthetic_nail(is_anemic=False)
        img_bytes = self._image_to_bytes(img)

        response = self.client.post(
            "/api/v1/predict",
            files={"file": ("test_nail.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn(data["state"], ["ANEMIA", "NO_ANEMIA", "INCONCLUSIVE"])
        self.assertIn("probability", data)

    def test_predict_multiple_endpoint(self):
        img1 = self._image_to_bytes(generate_synthetic_nail(is_anemic=False))
        img2 = self._image_to_bytes(generate_synthetic_nail(is_anemic=True))

        files = [
            ("files", ("nail1.jpg", img1, "image/jpeg")),
            ("files", ("nail2.jpg", img2, "image/jpeg")),
        ]
        response = self.client.post("/api/v1/predict-multiple", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total_images"], 2)

    def test_reject_corrupted_file(self):
        corrupted = b"not a valid image content"
        response = self.client.post(
            "/api/v1/predict",
            files={"file": ("corrupt.jpg", corrupted, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)

    def test_reject_blurry_image_as_inconclusive(self):
        blurry_bytes = self._image_to_bytes(generate_blurry_image())
        response = self.client.post(
            "/api/v1/predict",
            files={"file": ("blurry.jpg", blurry_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "INCONCLUSIVE")
        self.assertIsNone(data["probability"])


if __name__ == "__main__":
    unittest.main()
