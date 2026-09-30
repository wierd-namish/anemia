.PHONY: help install dev-install test test-unit test-api test-inference quality lint format run train evaluate download-models

PYTHON := python
PIP := $(PYTHON) -m pip
UVICORN := $(PYTHON) -m uvicorn

help:
	@echo "Anemia AI — Developer Commands"
	@echo "======================================"
	@echo "  make install         Install runtime dependencies"
	@echo "  make dev-install     Install package in editable mode with dev dependencies"
	@echo "  make test            Run complete automated test suite"
	@echo "  make test-unit       Run unit test suite"
	@echo "  make test-api        Run API integration test suite"
	@echo "  make test-inference  Run inference and smoke test suite"
	@echo "  make quality         Run full quality gates (secrets, checkpoints, tests)"
	@echo "  make run             Start the FastAPI development server"
	@echo "  make train           Run model training pipeline"
	@echo "  make evaluate        Run evaluation on test split"
	@echo "  make download-models Download external model weights"

install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

test-unit:
	$(PYTHON) -m unittest discover -s tests/unit -p "test_*.py"

test-api:
	$(PYTHON) -m unittest discover -s tests/api -p "test_*.py"

test-inference:
	$(PYTHON) -m unittest discover -s tests/inference -p "test_*.py"

quality:
	$(PYTHON) scripts/run_quality_checks.py

run:
	$(UVICORN) anemia_ai.api.app:app --host 0.0.0.0 --port 8000 --reload

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py --split test

download-models:
	$(PYTHON) scripts/download_models.py
