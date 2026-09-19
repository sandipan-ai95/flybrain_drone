# Dockerfile for FlyBrain Drone dashboard.
# Works locally AND on Hugging Face Spaces (Docker SDK, default port 7860).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    FLYBRAIN_CONNECTOME=gesture \
    FLYBRAIN_MAX_NEURONS=8000 \
    FLYBRAIN_FPS=15 \
    FLYBRAIN_PHYSICS=quadrotor

WORKDIR /app

# System deps kept lean; add build-essential only if wheels are missing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source and install the package itself
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY configs ./configs
RUN pip install -e .

# HF Spaces expects the container to listen on $PORT (7860 by default)
EXPOSE 7860

# Use a shell so env vars expand at container start time
CMD python -m flybrain.server.dashboard_server \
        --host ${HOST} \
        --port ${PORT} \
        --connectome ${FLYBRAIN_CONNECTOME} \
        --max-neurons ${FLYBRAIN_MAX_NEURONS} \
        --fps ${FLYBRAIN_FPS} \
        --physics ${FLYBRAIN_PHYSICS}
