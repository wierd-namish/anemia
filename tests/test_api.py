"""
Automated Integration Tests for Backend API Endpoints.
Tests:
- GET /health
- GET /model-info
- POST /predict
- POST /predict-multiple
- Error handling on invalid/corrupted payloads
"""

import io
import asyncio
import unittest
from fastapi import UploadFile, HTTPException
from PIL import Image

from anemia_ai.api.routes.health import health_check
from anemia_ai.api.routes.model_info import get_model_info
from anemia_ai.api.routes.predict import predict_multiple_nails, predict_single_nail
from anemia_ai.config.constants import ENSEMBLE_VERSION, LOCKED_DIAGNOSTIC_THRESHOLD


class TestBackendAPI(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def create_upload_file(self, color=(180, 100, 100), size=(300, 300), filename="test.jpg", content_type="image/jpeg"):
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return UploadFile(filename=filename, file=buf, headers={"content-type": content_type})

    def test_health_endpoint(self):
        """Verify health check returns healthy status and model readiness."""
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertTrue(res["model_loaded"])
        self.assertIn("model_version", res)

    def test_model_info_endpoint(self):
        """Verify model-info returns locked metadata and clinical disclaimers."""
        res = get_model_info()
        self.assertIn("primary_model", res)
        self.assertIn("locked_threshold", res)
        self.assertIn("disclaimer", res)

    def test_predict_single_invalid_file(self):
        """Verify /predict rejects non-image payload with HTTPException 400."""
        buf = io.BytesIO(b"not an image")
        upload = UploadFile(filename="test.txt", file=buf, headers={"content-type": "text/plain"})
        with self.assertRaises(HTTPException) as ctx:
            self.loop.run_until_complete(predict_single_nail(file=upload))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_predict_single_endpoint(self):
        """Verify /predict processes a valid image payload."""
        upload = self.create_upload_file()
        res = self.loop.run_until_complete(predict_single_nail(file=upload))
        self.assertTrue(res["success"])
        self.assertIn(res["state"], ["ANEMIA", "NO_ANEMIA", "INCONCLUSIVE"])

    def test_predict_multiple_endpoint(self):
        """Verify /predict-multiple handles multiple image uploads."""
        up1 = self.create_upload_file(filename="nail1.jpg")
        up2 = self.create_upload_file(filename="nail2.jpg")
        res = self.loop.run_until_complete(predict_multiple_nails(files=[up1, up2]))
        self.assertTrue(res["success"])
        self.assertIn(res["state"], ["ANEMIA", "NO_ANEMIA", "INCONCLUSIVE"])


if __name__ == "__main__":
    unittest.main()
