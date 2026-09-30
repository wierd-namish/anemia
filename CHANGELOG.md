# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-01

### Added
- Modular package layout under `src/anemia_ai/` (`config`, `schemas`, `core`, `preprocessing`, `models`, `calibration`, `inference`, `services`, `api`, `training`, `evaluation`, `utils`).
- Polymorphic `BaseModel` abstract interface and `ModelRegistry` for pluggable future AI models.
- Two-Model Ensemble pipeline (`ensemble_v003`) integrating EfficientNet-B0 v002 and Hugging Face JetX-GT/nail-anemia-detector.
- Non-parametric Isotonic Regression probability calibration (`v003`) and locked clinical decision thresholding (&tau;=0.9000).
- Robust Image Quality Gate rejecting optical blur, underexposure, overexposure, specular glare, and artificial nail polish.
- Complete automated test suite covering unit tests, API tests, model smoke tests, input dependence, OOD rejection, and patient leakage auditing.
- Automated quality gate runner (`scripts/run_quality_checks.py`).
- Deterministic model artifact downloader (`scripts/download_models.py`) and checkpoint verification (`scripts/verify_checkpoint.py`).
- Reproducible config-driven training entry point (`scripts/train.py`).
- Comprehensive documentation: architecture, development, deployment, model card, clinical validation, and ADRs.
- GitHub Actions CI/CD workflows for automated testing, linting, and secret scanning.
- Dockerfile and docker-compose configurations for containerized deployment.
- Backward compatibility facade in `backend/` for existing scripts and external imports.

### Changed
- Refactored monolithic backend scripts into decoupled, single-responsibility modules.
- Replaced hardcoded constants with typed configuration settings (`anemia_ai.config.settings`).
- Archived historical phase-specific scripts into `unnecessary/archived_scripts/` and `unnecessary/debug/`.
