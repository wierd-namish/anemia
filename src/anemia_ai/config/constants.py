"""
Global constants for Anemia AI diagnostic system.
"""

# Diagnostic States
STATE_ANEMIA: str = "ANEMIA"
STATE_NO_ANEMIA: str = "NO_ANEMIA"
STATE_INCONCLUSIVE: str = "INCONCLUSIVE"

# Model and Pipeline Versions
MODEL_NAME: str = "EfficientNet-B0 + JetX-GT Ensemble"
MODEL_VERSION: str = "ensemble_v003"
PRIMARY_MODEL_VERSION: str = "efficientnet_b0_v002"
SECONDARY_MODEL_VERSION: str = "JetX-GT/nail-anemia-detector"
ENSEMBLE_VERSION: str = "ensemble_v003"
PREPROCESSING_VERSION: str = "nail_roi_224x224_rgb_v1.0.0"
CALIBRATION_V002_VERSION: str = "isotonic_regression_v002"
CALIBRATION_V003_VERSION: str = "isotonic_regression_v003"
THRESHOLD_V002_VERSION: str = "locked_tau_v002"
THRESHOLD_V003_VERSION: str = "locked_tau_v003"

# Image Quality Assessment Thresholds
MIN_IMAGE_WIDTH: int = 128
MIN_IMAGE_HEIGHT: int = 128
BLUR_LAPLACIAN_VAR_THRESHOLD: float = 5.0
UNDEREXPOSURE_THRESHOLD: float = 40.0
OVEREXPOSURE_THRESHOLD: float = 230.0
GLARE_PIXEL_RATIO_MAX: float = 0.20
NON_PHYSIOLOGICAL_SATURATION_MAX: float = 0.15

# Default Decision Thresholds
LOCKED_DIAGNOSTIC_THRESHOLD: float = 0.9000
DEFAULT_DECISION_THRESHOLD: float = 0.9000
DEFAULT_DIAGNOSTIC_THRESHOLD: float = 0.50
BASELINE_BALANCED_THRESHOLD: float = 0.50
BASELINE_HIGH_SENSITIVITY_THRESHOLD: float = 0.30

# Clinical Disclaimers
PERSISTENT_CLINICAL_DISCLAIMER: str = (
    "Research prototype. This system is investigational and requires prospective clinical "
    "validation before general clinical use."
)
ASSESSMENT_RESULT_DISCLAIMER: str = (
    "Research/investigational result; confirmatory clinical evaluation is required."
)
CLINICAL_POPULATION_NOTE: str = (
    "Pediatric Ghanaian cohort (retrospective research dataset)"
)
