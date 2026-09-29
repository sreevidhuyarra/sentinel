# Sentinel — notes for Claude Code

Local-first AI threat-detection platform (mini SOC). Built module by module, then
integrated into one app. Design doc: `Sentinel_Project_Report.pdf` (six modules, six weeks).

## Status
- [x] Week 1 — scaffold, data pipeline (ingest → clean → label → temporal split → schema), DVC, CI, compose infra
- [x] Module 1 — supervised IDS: logreg / LightGBM / XGBoost / MLP, imbalance study, Optuna,
      calibration, benign-FPR threshold, LightGBM+MLP ensemble, TreeSHAP reasons, MLflow registry, FastAPI `/score/flows`
- [x] Module 1 follow-up — cross-dataset test on CIC-UNSW-NB15 (goal G3): `sentinel ids cross-dataset`,
      DVC stage `cross_dataset`, report `reports/ids/cross_dataset.md`. Result: no transfer (ROC-AUC 0.55–0.68)
      vs UNSW-trained reference 0.995.
- [x] Module 2 — anomaly detection: autoencoder (bottleneck 8) + Isolation Forest baseline on benign
      flows; fusion into `sentinel.detection.fusion.Detector` ("Unknown anomaly"); held-out-family and
      UNSW benign-only studies; `anomaly-detector` in MLflow; API serves the fused verdict
- [ ] Module 3 — phishing email + URL classifier (DeBERTa LoRA → ONNX int8)
  - [x] data: `sentinel phishing download|prepare` (Zenodo 8339691, 200k emails, grouped split, DVC stage
        `prepare_phishing`) → Kaggle upload folder `kaggle/phishing-emails/`
  - [x] Kaggle notebook `notebooks/phishing_lora_kaggle.ipynb` (built from the `.py` with
        `sentinel phishing notebook`; smoke test: `SENTINEL_SMOKE=1 SENTINEL_DATA_DIR=kaggle/phishing-emails`)
  - [ ] user runs it on Kaggle → `models/phishing/phishing_lora_output.zip`
  - [ ] merge + ONNX int8 + benchmarks + URL model + API
- [ ] Module 4 — adversarial robustness (ART)
- [ ] Module 5 — SOC copilot (LangGraph, RAG, injection guard)
- [ ] Module 6 — MLOps (Prometheus/Grafana, Evidently, Prefect) + streaming detector + React dashboard

## Commands (Windows: `make` is not installed, `uv` is not on PATH → `python -m uv` or `.venv\Scripts\*`)
- `uv sync` — install; `uv run pytest -m "not slow"` — fast tests (synthetic data)
- `uv run ruff check src tests`, `uv run ruff format src tests`, `uv run mypy`
- `uv run dvc repro` — data build + IDS training (needs MLflow up)
- `uv run sentinel data download|ingest|build|synth`
- `uv run sentinel ids train [--sample dev|full] [--models ...] [--no-register]`
- `uv run sentinel ids tune --family lightgbm|xgboost`, `uv run sentinel ids imbalance`
- `uv run sentinel ids cross-dataset [--download]`, `uv run sentinel anomaly train [--studies fusion,holdout,unsw]`
- `uv run sentinel serve` — API on :8000 (loads `ids-classifier` and `anomaly-detector` @production,
  falls back to `models/ids/`, `models/anomaly/`)
- `dvc repro -s <stage>` runs one stage only; plain `dvc repro <stage>` also reruns stale upstream
  stages (and deletes their outputs first). After a pure refactor, `dvc commit -f <stage>` instead.
- `docker compose up -d --wait` — Postgres :5432, MLflow :5000 (v3.16, must match client), Redpanda :19092
- Set `MLFLOW_DISABLE_AGENT_HINT=1` to silence MLflow's startup hint.

## Conventions
- src layout, package `sentinel`; one CLI (`src/sentinel/cli.py`, Typer) — each subcommand is a DVC stage.
- Polars for data (not pandas) except where a library needs pandas/numpy.
- All tunables in `params.yaml`, loaded via `sentinel.common.config.load_params()`; secrets in `.env`.
- Column names are canonical snake_case from `sentinel.data.columns.canonical()`.
- Model inputs come only from `FeatureSpec` (`data/processed/feature_spec.json`), fitted on train;
  the same spec must be used by training and serving. Never feed IDs/IPs/timestamps/labels as features.
- Splits: always the temporal split in `sentinel.data.splits` (split column in processed parquet);
  random splits exist only as a leakage comparison.
- Evaluation protocol: fit on train; calibration, threshold, ensemble weight and every
  hyperparameter choice on **validation**; test is scored once and never used to choose.
- Class order is `sentinel.data.labels.FAMILIES` everywhere (index 0 = Benign).
- Models expose `logits(X)`; probabilities only come from `TemperatureBias` calibration.
- The deployable artifact is `IDSBundle` (dir) → MLflow pyfunc `ids-classifier`; aliases
  `@staging` (every run) and `@production` (only if test macro-F1 improves within FPR budget).
- mypy strict must pass; ruff line length 100 (`X`/`X_val` names allowed).
- Tests must not need the real dataset unless marked `@pytest.mark.slow`; model tests use a
  throwaway SQLite MLflow (see `tests/model/test_ids_training.py`).

## Data facts (corrected CIC-IDS2017, DistriNet release)
- 2,099,976 raw flows → 1,841,347 after dedup; 84 model features.
- 'X - Attempted' labels (no payload) → Benign by default (`data.attempted_policy`).
- 'Infiltration - Portscan' (~72k) → PortScan family; true Infiltration is only 36 flows.
- Rare classes: Heartbleed 11, WebAttack ~100, Infiltration 36 — use macro-F1 / per-class recall.

## Module 3 notes
- Label 1 = phishing only for Nazario*, fraud for Nigerian*, spam elsewhere → `kind` column; report
  phishing recall separately (only ~1.5k true phishing emails).
- transformers 5.x: load DeBERTa with `dtype=torch.float32` (checkpoint is fp16; default keeps it →
  fp16 AMP crash on GPU, ~100x slower on CPU). `warmup_ratio` is gone: pass a float to `warmup_steps`.
- Notebook pins transformers 5.17.0 / peft 0.21.1 / accelerate 1.15.0 to match the laptop.

## Module 2 findings
- Autoencoder test ROC-AUC 0.943 vs Isolation Forest 0.918; 33% vs 13% recall at 1% benign FPR.
- Held out of supervised training: Infiltration 92%, Bot 100% caught by the autoencoder
  (supervised 0%); WebAttack ~2% — payload attacks are invisible at flow level.
- Fusion costs ~1% benign FPR (~2,900 alerts on the test split) for +0.1 pt recall when all
  families are known → anomaly alerts are Low/Medium severity.
- UNSW: CIC-trained AE scores 0.36 (below chance); retrained on UNSW benign only → 0.914.
- Registry helpers are generic in `sentinel.common.registry` (promote needs param sample=full).

## Module 1 findings (keep in mind for later modules)
- Class weighting is essential (unweighted LightGBM: Infiltration recall 0). sqrt weights chosen.
- MLP: standard scaling beats median/IQR (18 features have zero IQR); focal loss did not beat
  cross-entropy on validation, so `focal_gamma: 0`.
- Possible testbed shortcuts: `bwd_init_win_bytes` dominates DoS, `dst_port` dominates
  BruteForce/Bot/Infiltration. Module 4 (feature hardening) should test this.
- Tuning must use the real class mix (full train split); the dev sample over-represents
  rare classes ~10x and picked settings that overfit them (full-data macro-F1 0.88).
- Cross-dataset: CIC-UNSW-NB15 (HF mirror `bencorn/CIC-UNSW-NB15`; official download is
  behind a form) lacks 5 of our features (ICMP code/type, total TCP flow time, fwd/bwd RST
  flags) — shared set is computed at runtime. Benign traffic shifts on every top feature
  (PSI 0.3–4.3), so CIC-trained models do not transfer.
