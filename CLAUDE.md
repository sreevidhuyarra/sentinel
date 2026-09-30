# Sentinel — notes for Claude Code

Local-first AI threat-detection platform (mini SOC). Built module by module, then
integrated into one app. Design doc: `Sentinel_Project_Report.pdf` (six modules, six weeks).

## Status
- [x] Week 1 — scaffold, data pipeline (ingest → clean → label → temporal split → schema), DVC, CI, compose infra
- [x] Module 1 — supervised IDS: logreg / LightGBM / XGBoost / MLP, imbalance study, Optuna,
      calibration, benign-FPR threshold, LightGBM+MLP ensemble, TreeSHAP reasons, MLflow registry, FastAPI `/score/flows`
- [x] Module 1 CI — GitHub Actions green on the first push (private repo sreevidhuyarra/sentinel)
- [x] Module 1 follow-up — cross-dataset test on CIC-UNSW-NB15 (goal G3): `sentinel ids cross-dataset`,
      DVC stage `cross_dataset`, report `reports/ids/cross_dataset.md`. Result: no transfer (ROC-AUC 0.55–0.68)
      vs UNSW-trained reference 0.995.
- [x] Module 2 — anomaly detection: autoencoder (bottleneck 8) + Isolation Forest baseline on benign
      flows; fusion into `sentinel.detection.fusion.Detector` ("Unknown anomaly"); held-out-family and
      UNSW benign-only studies; `anomaly-detector` in MLflow; API serves the fused verdict
- [x] Module 3 — phishing email + URL classifier (DeBERTa LoRA → ONNX int8); CI green on 76b688c
  - [x] data: `sentinel phishing download|prepare` (Zenodo 8339691, 200k emails, grouped split, DVC stage
        `prepare_phishing`) → Kaggle upload folder `kaggle/phishing-emails/`
  - [x] Kaggle notebook `notebooks/phishing_lora_kaggle.ipynb` (built from the `.py` with
        `sentinel phishing notebook`; smoke test: `SENTINEL_SMOKE=1 SENTINEL_DATA_DIR=kaggle/phishing-emails`)
  - [x] Kaggle run done (68 min, 2× T4) → `models/phishing/phishing_lora_output/` (DVC-tracked)
  - [x] `sentinel phishing build` / DVC `build_phishing`: merge → ONNX (torch.export) → int8 embeddings,
        benchmark, `phishing-classifier` v1 @production; `POST /score/phishing` with sentence explanations
  - [x] URL model: `sentinel phishing url-train` / DVC `train_url` (PhiUSIIL + Hannousse; ISCX-URL2016 is
        form-gated and its mirrors lack raw URLs), `url-classifier` v3 @production, `POST /score/url`;
        combined email verdict evaluated — links left out (email_threshold None), see url_results.md
- [x] Module 4 — adversarial robustness (ART): `sentinel adversarial run` / DVC `robustness`
      (feature-space FGSM/PGD + transfer, HopSkipJump, problem-space pad×delay; defenses:
      adversarially trained MLP, hardened LightGBMs, review flag) → reports/adversarial/results.md
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

## Module 4 notes
- Threat model in `sentinel.adversarial.threat`: only CONTROLLABLE features move (44/84), TIMING and
  OWN_SIZE increase-only; `project()` enforces this after every attack step. Budgets are L-inf in
  the MLP scaler's z-space; transfer to LightGBM via `X0 + (Z_adv - Z0) * scale`.
- `pad_and_delay()` must keep dependent features consistent (rates, means, variance, max);
  unit tests guard it. Budget names are free: code uses the largest (`top`), never "high".
- Evasion counts only targets the detector caught clean. ART HSJ needs `init_eval < max_eval`.
- Findings: white-box PGD evades the MLP 95% at eps 0.5; ~55% transfers to LightGBM/ensemble at
  eps ≥ 1. Adversarial training helps at eps ≤ 0.5 only (no clean cost). Review flag (disagreement
  or AE anomaly, 1.4% benign to review) is the best defense: ≤5.5% feature-space, 3.5% realistic.
  Realistic high budget: Bot 94% evades ensemble (pad+delay combined only), PortScan 41% (delay),
  16% of PortScan beats the review flag — weakest spot. Dropping timing: 20.8% → 9.4%, −1.2 pt F1.

## Module 3 notes
- Label 1 = phishing only for Nazario*, fraud for Nigerian*, spam elsewhere → `kind` column; report
  phishing recall separately (only ~1.5k true phishing emails).
- transformers 5.x: load DeBERTa with `dtype=torch.float32` (checkpoint is fp16; default keeps it →
  fp16 AMP crash on GPU, ~100x slower on CPU). `warmup_ratio` is gone: pass a float to `warmup_steps`.
- Notebook pins transformers 5.17.0 / peft 0.21.1 / accelerate 1.15.0 to match the laptop, and
  uninstalls Kaggle's torchao 0.10 (PEFT rejects torchao < 0.16 even when unused).
- ONNX export must use `dynamo=True`: the legacy exporter is wrong on padded batches
  (`tests/unit/test_phishing_export.py` guards this). Strip `graph.value_info` before quantizing.
- Deployed quantization = embeddings only (full int8 flips 85/1000 verdicts at threshold 0.989).
- Test F1 0.944 at the deployed threshold (0.31% false alarms); 0.970 at 0.5 (3.99%).
- URL data shortcut: PhiUSIIL legitimate URLs are all bare homepages. Train source-balanced,
  set thresholds and judge promotion on Hannousse (realistic legitimate URLs), never on the
  PhiUSIIL-dominated overall numbers.

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
