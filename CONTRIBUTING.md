# Contributing to Anemia AI

Thank you for your interest in contributing to Anemia AI.

## Code of Conduct & Medical Responsibility

Contributors must adhere to the highest standards of scientific rigor, software engineering quality, and patient data protection.

## Development Setup

1. Fork and clone the repository.
2. Create and activate a Python virtual environment (`python -m venv .venv`).
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Download external model weights:
   ```bash
   python scripts/download_models.py
   ```

## Development Standards

- **Type Annotations**: All public functions, methods, and classes must include complete type hints.
- **Documentation**: Write clear docstrings explaining arguments, returns, and domain rationale.
- **Testing**: Every new feature or bugfix must be accompanied by unit and/or integration tests.
- **Model Versioning**: Never overwrite existing model manifests or checkpoint hashes silently. Increment versions (e.g. `v004`) and document them in `models/manifests/`.

## Running Quality Checks

Before submitting a Pull Request, you must execute the full quality gate:

```bash
python scripts/run_quality_checks.py
```

All 3 gates (Secret scan, checkpoint verification, and automated tests) must pass.

## Pull Request Process

1. Create a feature branch (`git checkout -b feat/your-feature-name`).
2. Commit changes using Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
3. Ensure all tests pass.
4. Submit your PR using the provided Pull Request template.
