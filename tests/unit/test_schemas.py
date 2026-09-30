"""
Unit tests for Pydantic schemas.
"""

import unittest
from anemia_ai.schemas.responses import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    MultiPredictionResponse,
    ErrorResponse,
)


class TestSchemasUnit(unittest.TestCase):
    """Verifies schema validation, serialization, and field types."""

    def test_health_response_schema(self):
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            model_name="EfficientNet-B0 + JetX-GT Ensemble",
            model_version="ensemble_v003",
            primary_model="efficientnet_b0_v002",
            secondary_model="JetX-GT/nail-anemia-detector",
            device="cuda",
            gpu="RTX 3070",
        )
        self.assertEqual(resp.status, "healthy")
        self.assertTrue(resp.model_loaded)
        data = resp.model_dump()
        self.assertIn("status", data)

    def test_prediction_response_schema(self):
        resp = PredictionResponse(
            request_id="test-1234",
            success=True,
            state="ANEMIA",
            probability=0.9250,
            threshold=0.9000,
            model_version="ensemble_v003",
            description="Test description",
            disclaimer="Test disclaimer",
        )
        self.assertEqual(resp.state, "ANEMIA")
        self.assertEqual(resp.probability, 0.925)

    def test_error_response_schema(self):
        err = ErrorResponse(
            error="ValidationError",
            message="Invalid image payload",
        )
        self.assertFalse(err.success)
        self.assertEqual(err.error, "ValidationError")


if __name__ == "__main__":
    unittest.main()
