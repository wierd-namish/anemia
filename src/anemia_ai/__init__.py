"""
Anemia AI — Deep Vision and Ensemble Diagnostics for Non-Invasive Anemia Assessment.
"""

import sys
from anemia_ai.version import __version__, __api_version__
import anemia_ai.calibration.calibrator
import anemia_ai.models.cnn_factory
import anemia_ai.inference.pipeline

# Register backward-compatibility aliases for unpickling legacy joblib model artifacts
sys.modules.setdefault("backend", sys.modules[__name__])
sys.modules.setdefault("backend.calibration", anemia_ai.calibration)
sys.modules.setdefault("backend.calibration.calibration", anemia_ai.calibration.calibrator)
sys.modules.setdefault("backend.model", anemia_ai.models)
sys.modules.setdefault("backend.model.calibration", anemia_ai.calibration.calibrator)
sys.modules.setdefault("backend.model.diagnostic_model", anemia_ai.models.cnn_factory)
sys.modules.setdefault("backend.model.inference_pipeline", anemia_ai.inference.pipeline)

__all__ = ["__version__", "__api_version__"]
