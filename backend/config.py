"""
Backward-compatibility configuration facade.
Re-exports settings and constants from anemia_ai.config.
"""

from pathlib import Path
from anemia_ai.config.constants import (
    ASSESSMENT_RESULT_DISCLAIMER,
    BASELINE_BALANCED_THRESHOLD,
    BASELINE_HIGH_SENSITIVITY_THRESHOLD,
    BLUR_LAPLACIAN_VAR_THRESHOLD,
    CALIBRATION_V002_VERSION,
    CALIBRATION_V003_VERSION,
    CLINICAL_POPULATION_NOTE,
    DEFAULT_DIAGNOSTIC_THRESHOLD,
    ENSEMBLE_VERSION,
    GLARE_PIXEL_RATIO_MAX,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    NON_PHYSIOLOGICAL_SATURATION_MAX,
    OVEREXPOSURE_THRESHOLD,
    PERSISTENT_CLINICAL_DISCLAIMER,
    PREPROCESSING_VERSION,
    PRIMARY_MODEL_VERSION,
    SECONDARY_MODEL_VERSION,
    STATE_ANEMIA,
    STATE_INCONCLUSIVE,
    STATE_NO_ANEMIA,
    THRESHOLD_V002_VERSION,
    THRESHOLD_V003_VERSION,
    UNDEREXPOSURE_THRESHOLD,
)
from anemia_ai.config.settings import get_settings

_settings = get_settings()

BASE_DIR = _settings.base_dir
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = _settings.data_dir
CONFIGS_DIR = _settings.configs_dir
EXPERIMENTS_DIR = _settings.experiments_dir
FRONTEND_DIR = _settings.frontend_dir

MODEL_NAME = "EfficientNet-B0"
MODEL_VERSION = PRIMARY_MODEL_VERSION
MODEL_PATH = _settings.v002_model_path
EFFICIENTNET_WEIGHTS_PATH = MODEL_PATH

CALIBRATION_VERSION = CALIBRATION_V002_VERSION
CALIBRATION_PATH = _settings.calibrator_v002_path
CALIBRATOR_PATH = CALIBRATION_PATH
CALIBRATION_METHOD = "Isotonic Regression"
PATIENT_AGGREGATION_STRATEGY = "mean"

THRESHOLD_VERSION = THRESHOLD_V002_VERSION
DIAGNOSTIC_THRESHOLD_CONFIG_PATH = _settings.locked_tau_v002_path


def load_diagnostic_threshold() -> float:
    return _settings.load_locked_threshold("v002")


THRESHOLD = load_diagnostic_threshold()
LOCKED_DIAGNOSTIC_THRESHOLD = THRESHOLD

BASELINE_MODEL_PATH = _settings.jetx_cache_dir / "mlp_model.joblib"
BASELINE_SCALER_PATH = _settings.jetx_cache_dir / "feature_scaler.joblib"
BASELINE_METADATA_PATH = _settings.jetx_cache_dir / "model_metadata.json"
