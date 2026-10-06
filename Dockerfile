# One image for every Sentinel service (api, detector, drift job, replayer); the compose
# service picks the command. Models and data are mounted at run time (DVC-tracked, not baked).

# --- dashboard ---------------------------------------------------------------------------
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- python ------------------------------------------------------------------------------
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
# Dependencies first (cached unless the lock file changes); CPU-only torch per pyproject.
# uv's download cache lives in a BuildKit cache mount, not in the image (it doubled its size).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
COPY params.yaml ./
COPY --from=frontend /frontend/dist frontend/dist
RUN useradd --create-home sentinel && mkdir -p data/cache && chown -R sentinel /app
USER sentinel
EXPOSE 8000 9101 9103
ENTRYPOINT ["sentinel"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
