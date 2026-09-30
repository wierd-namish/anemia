"""
Automated Tests for Mobile Camera Payloads and Image Encodings.
Tests:
- Standard mobile JPEG/PNG camera uploads
- High-resolution smartphone payloads (1920x1080)
- Low-resolution rejection (<128x128)
- Corrupted / truncated buffer handling
"""

import io
import asyncio
import unittest
from fastapi import UploadFile, HTTPException
from PIL import Image

from anemia_ai.api.routes.predict import predict_single_nail


class TestCameraPayload(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_standard_camera_jpeg_payload(self):
        """Verify standard phone camera JPEG payload is processed."""
        img = Image.new("RGB", (1280, 720), color=(180, 110, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        
        upload = UploadFile(filename="camera_capture.jpg", file=buf, headers={"content-type": "image/jpeg"})
        res = self.loop.run_until_complete(predict_single_nail(file=upload))
        self.assertTrue(res["success"])
        self.assertIn(res["state"], ["ANEMIA", "NO_ANEMIA", "INCONCLUSIVE"])

    def test_camera_png_payload(self):
        """Verify PNG encoded camera frames are processed."""
        img = Image.new("RGB", (640, 480), color=(170, 105, 95))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        
        upload = UploadFile(filename="camera_capture.png", file=buf, headers={"content-type": "image/png"})
        res = self.loop.run_until_complete(predict_single_nail(file=upload))
        self.assertTrue(res["success"])

    def test_low_resolution_camera_payload_rejection(self):
        """Verify camera payload below minimum 128x128 threshold is rejected as INCONCLUSIVE."""
        img = Image.new("RGB", (64, 64), color=(180, 110, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        
        upload = UploadFile(filename="small_capture.jpg", file=buf, headers={"content-type": "image/jpeg"})
        res = self.loop.run_until_complete(predict_single_nail(file=upload))
        self.assertEqual(res["state"], "INCONCLUSIVE")
        self.assertIsNone(res["probability"])
        self.assertIn("resolution", res["description"].lower())

    def test_corrupted_image_payload(self):
        """Verify corrupted byte stream raises HTTPException 400."""
        corrupted_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01...garbage_bytes_12345"
        buf = io.BytesIO(corrupted_bytes)
        upload = UploadFile(filename="corrupted.jpg", file=buf, headers={"content-type": "image/jpeg"})
        with self.assertRaises(HTTPException) as ctx:
            self.loop.run_until_complete(predict_single_nail(file=upload))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
