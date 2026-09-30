# Multi-stage production build for Anemia AI
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANEMIA_ENV=production

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications and install
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application source, configs, models, frontend, and tests
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY configs/ /app/configs/
COPY models/ /app/models/
COPY experiments/ /app/experiments/
COPY frontend/ /app/frontend/
COPY scripts/ /app/scripts/

# Non-root security user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "anemia_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
