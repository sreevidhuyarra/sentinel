# Sentinel public demo for Hugging Face Spaces (Docker SDK, port 7860).
# Built from the folder that scripts/stage_hf_space.py assembles: the code, the deployed model
# bundles and a SQLite snapshot of the alert database. No Postgres, Kafka, MLflow or LLM key.

# --- dashboard ---------------------------------------------------------------------------
FROM node:24-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- python ------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    SENTINEL_DEMO=1 \
    POSTGRES_URL=sqlite:////app/data/demo/sentinel_demo.sqlite
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN uv sync --frozen --no-dev
COPY params.yaml ./
COPY models/ models/
COPY reports/ reports/
COPY data/demo/ data/demo/
COPY --from=frontend /frontend/dist frontend/dist
# Spaces run the container as uid 1000; the blocklist cache and SQLite need a writable /app.
RUN useradd --create-home --uid 1000 user && mkdir -p data/cache && chown -R user /app
USER user
EXPOSE 7860
CMD ["sentinel", "serve", "--host", "0.0.0.0", "--port", "7860"]
