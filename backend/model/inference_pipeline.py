"""
Backward-compatibility facade for inference pipeline.
Re-exports DiagnosticInferenceService and InferencePipeline from anemia_ai.inference.
"""

from anemia_ai.inference.pipeline import (
    DiagnosticInferenceService,
    InferencePipeline,
)

__all__ = ["DiagnosticInferenceService", "InferencePipeline"]
