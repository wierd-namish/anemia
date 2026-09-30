# Developer Guide & Local Workflow

This document outlines the setup, testing, formatting, and contribution workflows for developers working on Anemia AI.

## Prerequisites

- **Python**: 3.9, 3.10, 3.11, 3.12, or 3.13
- **Hardware**: CPU supported; NVIDIA GPU with CUDA 12+ recommended for training.
- **Git**: For version control.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd MYPASS
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

## Running the Application

To start the FastAPI development server with hot reload:

```bash
uvicorn anemia_ai.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your web browser to test the mobile camera and file upload interface.

## Running Tests

Run the complete test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Or run targeted test suites:

```bash
# Unit tests
python -m unittest discover -s tests/unit -p "test_*.py"

# API integration tests
python -m unittest discover -s tests/api -p "test_*.py"

# Inference & model smoke tests
python -m unittest discover -s tests/inference -p "test_*.py"

# Training & leakage audit tests
python -m unittest discover -s tests/training -p "test_*.py"
```

## Running Automated Quality Gates

Before committing changes, execute the quality check runner:

```bash
python scripts/run_quality_checks.py
```

This verifies:
1. Absence of secrets, credentials, or private patient data.
2. Checkpoint SHA256 integrity and model weights loading.
3. Execution and passage of all 43+ unit, API, and inference tests.
