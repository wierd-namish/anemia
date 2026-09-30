"""
Out-of-Distribution (OOD) and Adversarial Surface Rejection Tests.
"""

import unittest
from anemia_ai.inference.pipeline import TwoModelEnsembleService
from tests.fixtures.synthetic_samples import (
    generate_blurry_image,
    generate_overexposed_image,
    generate_polished_nail,
    generate_underexposed_image,
    generate_wood_desk_ood,
)


class TestOODDefense(unittest.TestCase):
    """Verifies that non-nail, corrupted, or synthetic OOD inputs do not produce fabricated diagnoses."""

    @classmethod
    def setUpClass(cls):
        cls.service = TwoModelEnsembleService()

    def test_reject_wood_surface(self):
        img = generate_wood_desk_ood()
        res = self.service.predict_single(img)
        self.assertEqual(res["state"], "INCONCLUSIVE")
        self.assertIsNone(res["probability"])

    def test_reject_blurry_surface(self):
        img = generate_blurry_image()
        res = self.service.predict_single(img)
        self.assertEqual(res["state"], "INCONCLUSIVE")
        self.assertIsNone(res["probability"])

    def test_reject_underexposed_surface(self):
        img = generate_underexposed_image()
        res = self.service.predict_single(img)
        self.assertEqual(res["state"], "INCONCLUSIVE")
        self.assertIsNone(res["probability"])

    def test_reject_overexposed_surface(self):
        img = generate_overexposed_image()
        res = self.service.predict_single(img)
        self.assertEqual(res["state"], "INCONCLUSIVE")
        self.assertIsNone(res["probability"])

    def test_reject_colored_polish(self):
        img = generate_polished_nail()
        res = self.service.predict_single(img)
        self.assertEqual(res["state"], "INCONCLUSIVE")
        self.assertIsNone(res["probability"])


if __name__ == "__main__":
    unittest.main()
