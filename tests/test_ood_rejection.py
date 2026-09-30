"""
Automated Tests for Out-Of-Distribution (OOD) and Historical JetX Failure Regression.
Tests:
- Wood desk background (Historical JetX gave 99.99% anemia -> MUST be INCONCLUSIVE)
- Skin-only photograph (no nail plate) -> MUST be INCONCLUSIVE
- Clothing / textile fabric -> MUST be INCONCLUSIVE
- Random ceramic / mug / desk objects -> MUST be INCONCLUSIVE
- Blurry, underexposed, and overexposed frames -> MUST be INCONCLUSIVE
- Verify NO OOD sample produces ANEMIA or NO_ANEMIA
"""

import unittest
from pathlib import Path
from PIL import Image

from anemia_ai.config.constants import STATE_INCONCLUSIVE, STATE_ANEMIA, STATE_NO_ANEMIA
from anemia_ai.inference.pipeline import TwoModelEnsembleService

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ood"


class TestOODRejection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = TwoModelEnsembleService()

    def test_historical_jetx_wood_desk_regression(self):
        """
        CRITICAL REGRESSION TEST:
        Historical JetX-GT baseline assigned 99.99% anemia to wooden background.
        Our clinical pipeline MUST intercept and reject it as INCONCLUSIVE.
        """
        wood_desk = FIXTURES_DIR / "ood_wood_desk.jpg"
        if wood_desk.exists():
            img = Image.open(wood_desk)
            result = self.service.predict_single(img)
            self.assertEqual(result["state"], STATE_INCONCLUSIVE, f"Failed on {wood_desk.name}: got {result['state']}")
            self.assertIsNone(result["probability"])
            self.assertNotEqual(result["state"], STATE_ANEMIA)
            self.assertNotEqual(result["state"], STATE_NO_ANEMIA)

    def test_all_ood_samples_rejection(self):
        """Verify all synthetic/real OOD samples in fixtures are rejected as INCONCLUSIVE."""
        ood_images = list(FIXTURES_DIR.glob("*.jpg"))
        self.assertGreater(len(ood_images), 0, "No OOD test images found")
        
        for img_p in ood_images:
            img = Image.open(img_p)
            result = self.service.predict_single(img)
            
            self.assertEqual(
                result["state"],
                STATE_INCONCLUSIVE,
                f"OOD sample {img_p.name} was not rejected! Output state: {result['state']}, prob: {result.get('probability')}"
            )
            self.assertIsNone(result["probability"])
            self.assertIn("description", result)
            self.assertGreater(len(result["description"]), 0)


if __name__ == "__main__":
    unittest.main()
