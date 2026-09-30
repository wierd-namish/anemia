"""
Unit tests for image quality assessment and preprocessing checks.
"""

import unittest
from PIL import Image
from anemia_ai.preprocessing.image_quality import assess_image_quality, ImageQualityChecker
from tests.fixtures.synthetic_samples import (
    generate_synthetic_nail,
    generate_blurry_image,
    generate_underexposed_image,
    generate_overexposed_image,
    generate_polished_nail,
)


class TestPreprocessingQuality(unittest.TestCase):
    """Verifies that low quality, blurry, underexposed, overexposed, and polished images are rejected."""

    def setUp(self):
        self.checker = ImageQualityChecker()

    def test_valid_synthetic_nail_passes(self):
        nail = generate_synthetic_nail(is_anemic=False)
        passed, msg, metrics = self.checker.assess(nail)
        self.assertTrue(passed)
        self.assertIn("acceptable", msg.lower())

    def test_low_resolution_rejection(self):
        small = Image.new("RGB", (64, 64), color=(180, 140, 120))
        passed, msg, metrics = self.checker.assess(small)
        self.assertFalse(passed)
        self.assertEqual(metrics.get("error_type"), "low_resolution")

    def test_blurry_image_rejection(self):
        blurry = generate_blurry_image()
        passed, msg, metrics = self.checker.assess(blurry)
        self.assertFalse(passed)
        self.assertEqual(metrics.get("error_type"), "severe_blur")

    def test_underexposed_image_rejection(self):
        dark = generate_underexposed_image()
        passed, msg, metrics = self.checker.assess(dark)
        self.assertFalse(passed)
        self.assertEqual(metrics.get("error_type"), "underexposed")

    def test_overexposed_image_rejection(self):
        bright = generate_overexposed_image()
        passed, msg, metrics = self.checker.assess(bright)
        self.assertFalse(passed)
        self.assertEqual(metrics.get("error_type"), "overexposed")

    def test_nail_polish_detection(self):
        polished = generate_polished_nail()
        passed, msg, metrics = self.checker.assess(polished)
        self.assertFalse(passed)
        self.assertEqual(metrics.get("error_type"), "nail_polish_detected")


if __name__ == "__main__":
    unittest.main()
