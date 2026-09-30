# ADR 0001: Modular Package Layout (`src/anemia_ai`)

## Status
Accepted

## Context
The previous repository structure placed application logic directly in a flat `backend/` directory alongside experiments, training scripts, and notebooks without clear modular boundaries or a formal package hierarchy.

## Decision
Refactor all production application code into a standard Python package structure under `src/anemia_ai/` with separated domains:
- `config/`: Centralized settings and constants.
- `schemas/`: Pydantic request and response contracts.
- `core/`: Base interfaces, logging, and custom exception hierarchy.
- `preprocessing/`: Quality gates, ROI localization, and feature extraction.
- `models/`: Polymorphic model implementations.
- `calibration/`: Probability calibration algorithms.
- `inference/`: Orchestration pipelines.
- `services/`: Application domain services.
- `api/`: FastAPI routes and middleware.
- `training/`: Reproducible training engines.
- `evaluation/`: Clinical metrics and data leakage auditors.
- `utils/`: Shared hashing, timing, and image utilities.

A thin backward-compatibility facade is retained in `backend/` to prevent breakage of existing external scripts and tests.

## Consequences
- Clean separation of concerns and single-responsibility modules.
- Extensibility for future vision models and modalities without architectural disruption.
- Full backward compatibility maintained for all existing imports.
