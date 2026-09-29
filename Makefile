# On Windows without `make`, run the uv commands on the right directly
# (see README "Common commands").
.PHONY: setup data data-synth test test-all lint typecheck check up down logs

setup:            ## install deps and git hooks
	uv sync
	uv run pre-commit install

data:             ## raw CSVs -> processed Parquet (reproducible via DVC)
	uv run dvc repro

data-synth:       ## small synthetic dataset for quick experiments
	uv run sentinel data synth

test:             ## fast tests (synthetic data only)
	uv run pytest -m "not slow"

test-all:         ## includes checks against the real processed dataset
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

check: lint typecheck test

up:               ## start infrastructure (Postgres, MLflow, Redpanda)
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100
