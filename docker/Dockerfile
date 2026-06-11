# syntax=docker/dockerfile:1
# ============================================================
# Fruit Detector v2 — Production-grade multi-stage Dockerfile
# ============================================================
#
# Stages:
#   deps        — Install Python packages (cached layer)
#   production  — Slim runtime with non-root user
#   development — Dev tools + hot reload
#
# Build:
#   docker build -f docker/Dockerfile -t fruit-detector:latest .
#   docker build -f docker/Dockerfile --target development -t fruit-detector:dev .
#
# Run:
#   docker run --rm fruit-detector:latest --help
#   docker run --gpus all -v ./dataset:/app/dataset:ro -v ./runs:/app/runs fruit-detector:latest train
# ============================================================

# ---------------------------------------------------------------------------
# Stage 1: Dependencies (cached independently of source code changes)
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS deps

WORKDIR /app

# System packages needed for OpenCV (webcam), Pillow, and general operation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM deps AS production

WORKDIR /app

# Create non-root user with fixed UID/GID for security
RUN groupadd -g 1001 fruitapp && \
    useradd -u 1001 -g fruitapp -m -s /bin/bash fruitapp

# Copy source and install the package
COPY --chown=fruitapp:fruitapp pyproject.toml .
COPY --chown=fruitapp:fruitapp src/ src/
RUN pip install --no-cache-dir . && \
    pip cache purge 2>/dev/null || true

# Create directories the app needs at runtime (writable by non-root user)
RUN mkdir -p /app/runs /app/cache /app/output && \
    chown -R fruitapp:fruitapp /app/runs /app/cache /app/output

# Switch to non-root user
USER fruitapp

# Health check: verify the CLI is importable and responsive
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from fruit_detector.model import FruitDetectorV2; print('ok')" || exit 1

# Metadata
LABEL maintainer="Anshu" \
      description="Anchor-free fruit detection with ConvNeXt + PANet + DFL" \
      version="2.0"

ENTRYPOINT ["fruit-detect"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# Stage 3: Development image (includes dev tools, tests, source mount)
# ---------------------------------------------------------------------------
FROM deps AS development

WORKDIR /app

# Install dev dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[dev,metrics]"

# Copy test suite
COPY tests/ tests/

# Development runs as root for flexibility with volume mounts
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["fruit-detect"]
CMD ["--help"]
