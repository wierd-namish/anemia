"""
Application settings and typed configuration loading with environment overrides.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from anemia_ai.config.constants import (
    DEFAULT_DIAGNOSTIC_THRESHOLD,
    ENSEMBLE_VERSION,
    PRIMARY_MODEL_VERSION,
    SECONDARY_MODEL_VERSION,
)


@dataclass
class AppSettings:
    """System-wide configuration settings with environment variable support."""

    # Project Roots
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ANEMIA_BASE_DIR", str(Path(__file__).resolve().parent.parent.parent.parent))
        )
    )

    # API Server Settings
    api_title: str = "AI Nail Anemia Assessment API"
    api_description: str = "Medical-AI service for evaluating anemia probability from fingernail photographs."
    api_version: str = "v1"
    host: str = field(default_factory=lambda: os.getenv("ANEMIA_API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("ANEMIA_API_PORT", "8000")))
    cors_origins: List[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("ANEMIA_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )
    log_level: str = field(default_factory=lambda: os.getenv("ANEMIA_LOG_LEVEL", "INFO").upper())
    environment: str = field(default_factory=lambda: os.getenv("ANEMIA_ENV", "development"))

    # Directory Paths
    @property
    def configs_dir(self) -> Path:
        custom = os.getenv("ANEMIA_CONFIGS_DIR")
        return Path(custom) if custom else self.base_dir / "configs"

    @property
    def experiments_dir(self) -> Path:
        custom = os.getenv("ANEMIA_EXPERIMENTS_DIR")
        return Path(custom) if custom else self.base_dir / "experiments"

    @property
    def models_dir(self) -> Path:
        custom = os.getenv("ANEMIA_MODELS_DIR")
        return Path(custom) if custom else self.base_dir / "models"

    @property
    def frontend_dir(self) -> Path:
        custom = os.getenv("ANEMIA_FRONTEND_DIR")
        return Path(custom) if custom else self.base_dir / "frontend"

    @property
    def data_dir(self) -> Path:
        custom = os.getenv("ANEMIA_DATA_DIR")
        return Path(custom) if custom else self.base_dir / "data"

    # Artifact Paths
    @property
    def v002_model_path(self) -> Path:
        custom = os.getenv("ANEMIA_V002_MODEL_PATH")
        return Path(custom) if custom else self.experiments_dir / "efficientnet_b0_v002" / "best_model.pth"

    @property
    def fusion_model_path(self) -> Path:
        custom = os.getenv("ANEMIA_FUSION_MODEL_PATH")
        return Path(custom) if custom else self.configs_dir / "ensemble_fusion_v003.joblib"

    @property
    def calibrator_v002_path(self) -> Path:
        custom = os.getenv("ANEMIA_CALIBRATOR_V002_PATH")
        return Path(custom) if custom else self.configs_dir / "calibrator_isotonic_v002.joblib"

    @property
    def calibrator_v003_path(self) -> Path:
        custom = os.getenv("ANEMIA_CALIBRATOR_V003_PATH")
        return Path(custom) if custom else self.configs_dir / "calibrator_isotonic_v003.joblib"

    @property
    def locked_tau_v002_path(self) -> Path:
        custom = os.getenv("ANEMIA_LOCKED_TAU_V002_PATH")
        return Path(custom) if custom else self.configs_dir / "locked_tau_v002.json"

    @property
    def locked_tau_v003_path(self) -> Path:
        custom = os.getenv("ANEMIA_LOCKED_TAU_V003_PATH")
        return Path(custom) if custom else self.configs_dir / "locked_tau_v003.json"

    @property
    def jetx_cache_dir(self) -> Path:
        custom = os.getenv("ANEMIA_JETX_CACHE_DIR")
        return Path(custom) if custom else self.models_dir / "jetx_gt"

    def load_locked_threshold(self, version: str = "v003") -> float:
        """Loads locked decision threshold from configuration JSON with validation."""
        path = self.locked_tau_v003_path if version == "v003" else self.locked_tau_v002_path
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    val = float(data.get("locked_threshold", DEFAULT_DIAGNOSTIC_THRESHOLD))
                    if 0.0 < val < 1.0:
                        return val
            except Exception:
                return DEFAULT_DIAGNOSTIC_THRESHOLD
        return DEFAULT_DIAGNOSTIC_THRESHOLD


_settings_instance: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Singleton getter for application settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance
