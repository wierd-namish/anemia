# Production Deployment Guide

## Overview

Anemia AI is designed for containerized deployment across cloud environments, edge medical servers, or local clinic workstations.

## Docker Deployment

### Building the Image

```bash
docker build -t anemia-ai:latest .
```

### Running with Docker

```bash
docker run -d \
  --name anemia-ai-server \
  -p 8000:8000 \
  -e ANEMIA_ENV=production \
  -e ANEMIA_LOG_LEVEL=INFO \
  anemia-ai:latest
```

### Running with Docker Compose

```bash
docker compose up -d
```

Check health:
```bash
curl http://localhost:8000/api/v1/health
```

## GPU Acceleration (NVIDIA Container Toolkit)

To enable GPU inference inside Docker containers:

```bash
docker run -d \
  --gpus all \
  --name anemia-ai-gpu \
  -p 8000:8000 \
  anemia-ai:latest
```

## Environment Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ANEMIA_ENV` | `development` | Runtime environment (`development`, `production`) |
| `ANEMIA_API_HOST` | `0.0.0.0` | Bind IP address for server |
| `ANEMIA_API_PORT` | `8000` | Port for HTTP server |
| `ANEMIA_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated for production) |
| `ANEMIA_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ANEMIA_BASE_DIR` | `<repo_root>` | Custom base directory override |

## Reverse Proxy (Nginx) Configuration

```nginx
server {
    listen 80;
    server_name anemia-screening.clinic.internal;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
