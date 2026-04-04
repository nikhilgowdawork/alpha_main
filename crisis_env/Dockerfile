# -------------------------
# Build stage
# -------------------------
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install git (needed for VCS dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build arguments
ARG BUILD_MODE=in-repo
ARG ENV_NAME=crisis_response_env

# Copy environment code
COPY . /app/env
WORKDIR /app/env

# Install uv if not present (for local builds)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv (cache enabled)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# -------------------------
# Runtime stage
# -------------------------
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy environment code
COPY --from=builder /app/env /app/env

# Use virtual environment by default
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
WORKDIR /app/env
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]