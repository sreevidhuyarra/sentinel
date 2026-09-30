# Sentinel

**An AI-powered threat detection and response platform that runs entirely on a laptop.**

Sentinel behaves like a miniature Security Operations Center: it streams network flows
through ML detectors, catches known and never-seen attacks, flags phishing, explains every
alert, and hands alerts to an LLM copilot that writes cited incident reports. Models are
attacked and hardened, and the lifecycle (data versioning, tracking, drift, retraining)
is automated.

> Status: **Modules 1–2 complete** — data foundation, supervised intrusion detection,
> anomaly detection.
> Modules are built one at a time; see [CLAUDE.md](CLAUDE.md) for the checklist.

## Getting started

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker Desktop, ~10 GB disk,
16 GB RAM (32 GB comfortable). No GPU needed.

```bash
git clone <repo> sentinel && cd sentinel
cp .env.example .env
uv sync
uv run pre-commit install

docker compose up -d --wait        # Postgres :5432, MLflow :5000, Redpanda :19092
uv run sentinel data download      # corrected CIC-IDS2017, ~330 MB zip -> 1.1 GB CSV
uv run sentinel ids cross-dataset --download   # optional: fetch CIC-UNSW-NB15 (1.8 GB)
uv run dvc repro                   # data (~30 s), IDS (~20 min), anomaly (~50 min), cross-dataset (~15 min)
uv run sentinel serve              # API on http://127.0.0.1:8000 (docs at /docs)
```

No data yet? `uv run sentinel data synth` writes a small synthetic dataset in the same
format; the test suite and CI use it.

## Common commands

| Task | Command |
|---|---|
| Fast tests | `uv run pytest -m "not slow"` |
| Lint / format / types | `uv run ruff check src tests` · `uv run ruff format src tests` · `uv run mypy` |
| Rebuild data and models | `uv run dvc repro` |
| Metrics of the last run | `uv run dvc metrics show` |
| Train only (quick, 10% sample) | `uv run sentinel ids train --sample dev` |
| Hyperparameter search | `uv run sentinel ids tune --family lightgbm` |
| Imbalance study | `uv run sentinel ids imbalance` |
| Cross-dataset test | `uv run sentinel ids cross-dataset [--download]` |
| Anomaly detector + studies | `uv run sentinel anomaly train` |
| Serve the API | `uv run sentinel serve` |
| Infra up / down | `docker compose up -d --wait` · `docker compose down` |

On Linux/macOS/WSL, the `Makefile` wraps the common ones (`make check`, `make data`, `make up`).

## Data

Primary dataset: the **corrected CIC-IDS2017** release by Engelen et al. (DistriNet, KU Leuven),
which fixes labelling and feature-extraction errors in the original.

| | |
|---|---|
| Raw flows | 2,099,976 across 5 days |
| After removing exact duplicates | 1,841,347 |
| Model features | 84 (82 CICFlowMeter + 2 derived ratios) |
| Families | Benign, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration |

Decisions worth knowing:

- **Attempted attacks** (`X - Attempted`: the attack tool connected but sent no malicious
  payload) are labelled Benign by default, since their traffic is indistinguishable from benign.
  Configurable via `data.attempted_policy` in `params.yaml`.
- **"Infiltration - Portscan"** (~72k flows, a scan launched from the compromised host) is
  mapped to PortScan; true Infiltration is only 36 flows.
- **No random splits.** Each attack in CIC-IDS2017 runs in one contiguous time window, and
  consecutive flows of a session are near-identical. Flows are ordered in time within every
  (day, label) group; the earliest 60% go to train, the next 20% to validation, the latest 20%
  to test, with a small gap dropped at each boundary. A held-out-family split (for the anomaly
  detector) and a random split (to show the leakage it causes) are also available.
- Identifiers (IPs, source port, flow ID, timestamps) are never model inputs.

## Module 1: supervised intrusion detection

Classifies each flow as Benign or one of seven attack families and explains every alert.

**Pipeline.** Four models are trained on the 84 features: class-weighted logistic regression
(the floor), LightGBM (the production model), XGBoost (comparison) and a PyTorch MLP.
Each is calibrated on validation (temperature scaling plus a per-class bias, which also
corrects the prior shift that class weighting introduces). A flow becomes an alert when its
calibrated attack probability clears a threshold chosen on validation to maximise macro-F1
while keeping benign false positives at or below 1%. The final model averages the calibrated
LightGBM (0.7) and MLP (0.3) probabilities. Every alert carries a severity band and its top
five TreeSHAP reasons, e.g. *"destination port = 22 increased the likelihood of BruteForce"*.

Every choice (hyperparameters, class weights, calibration, threshold, ensemble weight) is made
on the validation split. The test split is scored once at the end.

### Results (test split: 367,940 flows, later in time than all training data)

| Model | Macro-F1 | Benign FPR | Attack recall | PR-AUC | Latency / 1k flows |
|---|---|---|---|---|---|
| Logistic regression | 0.9524 | 0.176% | 0.9922 | 0.9480 | 0.7 ms |
| LightGBM | 0.9926 | 0.001% | 0.9958 | 0.9999 | 18.6 ms |
| XGBoost | 0.9987 | 0.002% | 0.9963 | 0.9999 | 8.7 ms |
| MLP | 0.9748 | 0.013% | 0.9712 | 0.9870 | 1.1 ms |
| **Ensemble (deployed)** | **0.9957** | **0.001%** | **0.9959** | **0.9999** | **17.5 ms** |

The deployed ensemble raised 3 false alarms on 286,833 benign test flows, and every one of
its attack predictions named the right family (precision 1.0 for all seven). What it misses
are 329 attack flows scored as benign, mostly low-volume DoS and scan flows. Full per-family
tables, the confusion matrix and SHAP summaries: [reports/ids/](reports/ids/) (generated by
training; the [model card](reports/ids/model_card.md) lists limitations).

XGBoost scores slightly higher on test, but it tied LightGBM on validation (0.99996 vs
0.99994), where choices are made. The test gap comes down to a few flows of classes with
8–22 test examples. LightGBM trains ~4x faster, which matters for drift-triggered retraining.

### What we learned

- **Class weighting is essential.** Without it LightGBM misses every Infiltration flow
  (recall 0). Square-root and inverse-frequency weights, random oversampling and SMOTE all
  fix it about equally ([imbalance study](reports/ids/imbalance.md)); weights are simpler.
- **Tune on the real class mix.** Tuning first ran on a 10% dev sample that keeps rare
  classes whole. That makes them ~10x more common than in reality, and precision depends on
  prevalence: the dev-tuned LightGBM (182 leaves, strong weights) labelled ~100 benign test
  flows as Infiltration and scored 0.88 macro-F1. Re-tuning on the full training split
  chose smaller trees, gentler weights (power 0.22) and more regularisation: 0.99.
- **Focal loss did not help** the MLP here; plain cross-entropy with square-root weights
  scored higher on validation, so it is used. Standard scaling beat median/IQR scaling by
  ~5 points because 18 features are zero in over 75% of flows.
- **Possible shortcut features.** The TCP initial window size of the server's reply
  (`bwd_init_win_bytes`) dominates DoS detection, and destination port drives BruteForce,
  Bot and Infiltration. These may describe the 2017 lab's hosts more than attack behaviour;
  the window size is the most-shifted feature in the cross-dataset test below. The
  adversarial module (feature hardening) will test models trained without them.
- **These scores do not transfer to another network** (next section).

### Validation vs test

Validation is where every choice was made, so it reads slightly optimistic; test is the
honest number. The gap is small, so the tuning did not overfit validation badly.

| Model | Val macro-F1 | Val benign FPR | Test macro-F1 | Test benign FPR |
|---|---|---|---|---|
| Logistic regression | 0.9674 | 0.556% | 0.9524 | 0.176% |
| LightGBM | 0.9999 | 0.002% | 0.9926 | 0.001% |
| XGBoost | 0.9999 | 0.004% | 0.9987 | 0.002% |
| MLP | 0.9881 | 0.030% | 0.9748 | 0.013% |
| Ensemble | 0.9999 | 0.001% | 0.9957 | 0.001% |

### Cross-dataset test: CIC-IDS2017 → UNSW-NB15

`uv run sentinel ids cross-dataset [--download]` retrains the models on the 79 features both
datasets share and applies them, unchanged, to
[CIC-UNSW-NB15](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html): the UNSW-NB15 captures
(a different network, 2015, 3.5M flows) re-extracted with the same CICFlowMeter tool. UNSW
uses different attack categories, so the test is benign vs attack.
Full report: [reports/ids/cross_dataset.md](reports/ids/cross_dataset.md).

| Model | CIC test macro-F1 | UNSW attacks caught | UNSW benign FPR | UNSW ROC-AUC |
|---|---|---|---|---|
| LightGBM (CIC-trained) | 0.9806 | 3.1% | 0.85% | 0.570 |
| XGBoost (CIC-trained) | 0.9982 | 5.6% | 1.28% | 0.677 |
| MLP (CIC-trained) | 0.9473 | 8.6% | 5.77% | 0.552 |
| Ensemble (CIC-trained) | 0.9694 | 4.6% | 3.33% | 0.577 |
| *LightGBM trained on UNSW itself (reference)* | - | 87.4% | 1.13% | 0.995 |

- **The models do not transfer.** Near-perfect in their own lab, they catch 3–9% of UNSW
  attacks and rank flows barely better than chance (ROC-AUC 0.55–0.68). Picking a new
  threshold would not help: even a threshold chosen with UNSW's own labels catches only
  2.6–5.3% at 1% FPR. Every category is missed, including the two with a counterpart:
  Reconnaissance (0.02% flagged) and DoS (7%).
- **It is the network, not the features.** A model trained on UNSW with the same 79
  features reaches ROC-AUC 0.995. The five features UNSW lacks carry 0.15% of the model's
  gain; without them the CIC test scores change only through 4 rare-class flows.
- **Where it breaks.** Even *benign* traffic differs sharply between the two networks on
  every one of the model's 15 most important features (population stability index
  0.3–4.3; above 0.25 counts as a major shift). CIC's benign traffic is DNS-heavy (median
  destination port 53) with a median server TCP window of 0; UNSW's has median port 14,350
  and window 14,480. `bwd_init_win_bytes`, flagged above as a possible shortcut, shifts most.
- **What it means for Sentinel.** A flow classifier learns the network it was trained on.
  Deployment needs training or fine-tuning on the target network, which is the reason for the
  anomaly detector (Module 2, which learns only what normal looks like), drift monitoring
  and retraining (Module 6), and feature hardening (Module 4).

## Module 2: anomaly detection (unknown attacks)

A supervised model only recognises what it was trained on. Module 2 learns what *normal*
traffic looks like instead, so it can flag attacks nobody has labelled yet.

**Pipeline.** An autoencoder (84 → 64 → 32 → 8 → 32 → 64 → 84) is trained to reconstruct
benign training flows only; a flow it cannot reconstruct scores high. The code size (8) was
picked from {8, 12, 16} on validation, and the threshold is the score exceeded by 1% of benign
validation flows. An Isolation Forest is the baseline. In the deployed detector the supervised
verdict stands when it raises an alert; when it says Benign but the autoencoder fires, the flow
becomes an **"Unknown anomaly"** alert (severity Low, or Medium at twice the threshold) whose
reasons name the features that looked wrong and what a normal-looking flow would have, e.g.
*"forward header length = 18,868 is unusual for normal traffic (normal-looking value ~ 860)"*
(a Heartbleed flow the supervised model scored 0.3% likely to be an attack).
`uv run sentinel anomaly train` (or `dvc repro`) trains it and runs every study below;
full tables: [reports/anomaly/results.md](reports/anomaly/results.md).

### Results (CIC-IDS2017 test split)

| Detector | ROC-AUC | Attacks caught at 1% benign FPR |
|---|---|---|
| Isolation Forest (baseline) | 0.918 | 12.6% |
| **Autoencoder** | **0.943** | **33.2%** |

**Unknown-attack test.** Each family was removed from supervised training in turn; all of its
flows were then scored:

| Held-out family | Flows | Supervised model | Autoencoder | Isolation Forest | Supervised + autoencoder |
|---|---|---|---|---|---|
| Infiltration | 36 | 0% | 92% | 39% | **92%** |
| Bot | 708 | 0% | 100% | 4% | **100%** |
| WebAttack | 102 | 1% | 1% | 0% | **2%** |

Benign false alarms go from 0.001% (supervised alone) to 1.0% with the autoencoder added.

**New network, no attack labels** (CIC-UNSW-NB15, 521k test flows):

| Detector | Needs from the new network | ROC-AUC |
|---|---|---|
| Supervised models trained on CIC-IDS2017 (Module 1) | nothing | 0.55–0.68 |
| Autoencoder trained on CIC-IDS2017, as is or re-thresholded | nothing / benign traffic | 0.36 |
| **Autoencoder retrained on UNSW benign traffic** | **benign traffic only** | **0.914** |
| Supervised model trained on UNSW (reference) | labelled attacks | 0.995 |

### What we learned

- **It catches unknown attacks that behave differently.** Bot and Infiltration, invisible to a
  supervised model that never saw them, are caught 92–100%. The autoencoder beats the
  Isolation Forest everywhere that matters.
- **It cannot see attacks that hide in content.** Web attacks (brute-force logins, XSS, SQL
  injection) are ordinary HTTP connections at the flow level; nothing here catches them
  unseen. Payload-level detectors (like Module 3's phishing model) cover that.
- **Its price is false alarms.** With all families known, it lifts attack recall only from
  99.59% to 99.69% while adding ~2,900 false alarms on 287k benign test flows (1%). That is why
  anomaly alerts stay Low/Medium and form their own queue. A stricter budget helps little:
  at 0.1% FPR it catches 1.5% of attacks (trade-off table in the report).
- **"Normal" is local.** Moved to UNSW-NB15 unchanged, the CIC-trained autoencoder scores
  below chance (0.36): UNSW's normal traffic looks stranger to it than UNSW's attacks. Retrained
  on UNSW's benign traffic alone, which needs no labelling, it reaches 0.914. On a new network,
  this is the detector you can have on day one.

## Module 3: phishing email classifier

`uv run sentinel phishing download` fetches the Phishing Email Curated Datasets (11 public
corpora, [Zenodo 8339691](https://doi.org/10.5281/zenodo.8339691), CC BY 4.0) and
`uv run dvc repro prepare_phishing` turns them into 200,517 deduplicated emails, split so no
sender domain or subject campaign appears in two splits. Label 1 means phishing only in the
Nazario corpora, fraud in the Nigerian ones and spam elsewhere, so results report each kind.

Fine-tuning DeBERTa-v3-small with LoRA needs a GPU and runs on Kaggle:

1. Kaggle → Create → New Dataset → upload the four files in `kaggle/phishing-emails/`
   (train/val/test parquet + README), name it `sentinel-phishing-emails`, keep it private.
2. Kaggle → Create → New Notebook → File → Import Notebook →
   `notebooks/phishing_lora_kaggle.ipynb`.
3. Session options: Accelerator **GPU T4 x2**, Internet **On**; Add Input → your dataset.
4. Save Version → Save & Run All (~75 min on 2× T4), then download
   `phishing_lora_output.zip` from Output and unzip it into `models/phishing/`.

Then `uv run dvc repro build_phishing` (~11 min, CPU) merges the adapter, exports ONNX,
quantizes it, benchmarks every variant and registers `phishing-classifier` in MLflow.
Full tables: [reports/phishing/results.md](reports/phishing/results.md),
[model card](reports/phishing/model_card.md).

**Model.** DeBERTa-v3-small (141M parameters) with LoRA rank 16 on the attention
projections (0.72% of weights trainable), 256 tokens of subject + body, 2 epochs.

**Results (test split, 36,087 emails):**

| Threshold | F1 | False alarms on legitimate mail | Malicious caught | Phishing caught |
|---|---|---|---|---|
| 0.5 | 0.970 | 3.99% | 97.6% | 93.2% |
| **0.989 (deployed: 2% validation budget)** | **0.944** | **0.31%** | **89.7%** | **77.7%** |

PR-AUC 0.997, ROC-AUC 0.997. A TF-IDF + logistic regression baseline on the same split
reaches ROC-AUC 0.979 and cannot get its false alarms below ~2.7%.

**Optimisation** (1,000 test emails, laptop CPU):

| Model | Size | F1 | Verdicts changed | Latency p50 / p95 |
|---|---|---|---|---|
| PyTorch, float32 | — | 0.957 | — | 204 / 252 ms |
| ONNX, float32 | 568 MB | 0.957 | 0 | 130 / 161 ms |
| **ONNX, int8 embeddings (deployed)** | **273 MB** | **0.957** | **2** | **129 / 158 ms** |
| ONNX, int8 everything | 165 MB | 0.872 | 85 | 87 / 113 ms |

What we learned:

- **ONNX alone gives the speed-up** (1.6× lower latency); quantizing the embedding table
  halves the size at no accuracy cost. Full int8 is fastest, but DeBERTa's outlier
  activations shift high probabilities enough to flip 85 of 1,000 verdicts at the deployed
  threshold, so it is not used. The report's target of ≤1 F1 point with ≥2× speed and ~4×
  size is not reachable on this model with dynamic int8.
- **Export bugs a smoke test catches.** Transformers 5 loads the fp16 checkpoint as fp16
  (breaks mixed-precision training, ~100× slower on CPU): load as float32. The legacy ONNX
  exporter mistranslates DeBERTa's masked softmax: exact on unpadded input, wrong on padded
  batches (logits off by up to 2). The `torch.export` exporter is exact; a regression test
  guards it.
- **Mostly spam, little phishing.** Only ~1,500 emails are true phishing, and phishing
  recall is the weakest per kind, so it is reported separately rather than hidden in F1.

`POST /score/phishing` takes `{subject, body, urls[]}` and returns the verdict,
probability, severity, the URLs found, and the sentences whose removal most lowers the
score, e.g. *"Dear customer,"* and *"Urgent: your account has been suspended"* for a
credential-phishing email (~200 ms with explanation).

### URL classifier

`uv run dvc repro train_url` trains character 3–5-gram TF-IDF (on normalised URLs) plus 23
lexical features (length, digit ratio, subdomains, entropy, suspicious TLD, IP host,
shortener, sensitive words, ...) with LightGBM. Data: PhiUSIIL (UCI 967) + Hannousse &
Yahiouche 2021, 242k raw URLs, split by registered domain. ISCX-URL2016, named in the
design, sits behind a registration form, and its open mirrors carry only pre-computed
features that serving could not reproduce. Report:
[reports/phishing/url_results.md](reports/phishing/url_results.md).

| Test URLs | ROC-AUC | False alarms on legitimate | Malicious caught |
|---|---|---|---|
| All (deployed, source-balanced) | 0.999 | 0.00% | 34.4% |
| Hannousse only (realistic legitimate URLs) | 0.979 | 0.09% | 38.6% |
| Trained on PhiUSIIL only, scored on Hannousse | 0.661 | 99.5% | 99.8% |

- **Dataset shortcut.** Every legitimate PhiUSIIL URL is a bare homepage, so a model trained
  on it learns "has a path → malicious": trained on PhiUSIIL alone it flags 99.5% of
  Hannousse's legitimate URLs. Balancing the two sources and setting the threshold on
  Hannousse's realistic legitimate URLs keeps false alarms at 0.09%, but the price is a
  strict threshold that catches ~39% of malicious URLs. It works as a high-confidence link
  checker (`POST /score/url`, with TreeSHAP reasons such as *"suspicious tld = 1"*).
- **Links do not improve email verdicts.** Legitimate email links (newsletters, tracking,
  documents) look unlike the training data's legitimate URLs. On validation, no URL
  threshold let links flag emails while adding under 0.5 points of false alarms, so links
  are shown but do not change the email verdict. A learned text + link combination adds only
  ~1–2 points of malicious mail caught at matched false-alarm rates.

## API

```bash
uv run sentinel serve
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models
curl -X POST http://127.0.0.1:8000/score/flows -H "Content-Type: application/json" \
     -d '{"flows": [{"dst_port": 22, "flow_duration": 5000000, ...}], "explain": true}'
```

`POST /score/flows` takes up to 10,000 flows (canonical feature names; a 422 response lists
any missing ones) and returns family (one of the eight, or "Unknown anomaly"), which detector
raised it, calibrated confidence, attack score, anomaly score, severity, class probabilities
and, for alerts, the top-5 reasons. The API loads `models:/ids-classifier@production` and
`models:/anomaly-detector@production` from MLflow and falls back to `models/ids/` and
`models/anomaly/` (serving supervised verdicts only if no anomaly model exists).

## Model registry

Each training run logs parameters, metrics, figures and the model bundle to MLflow
(experiment `sentinel-ids`, http://localhost:5000) and registers a new `ids-classifier`
version with alias `@staging`. It moves to `@production` only if it was trained on the
full data, beats the current production model on test macro-F1, and keeps benign FPR within
budget.

## Repository layout

```
src/sentinel/
  common/     config (params.yaml + .env), JSON logging
  data/       download, ingest, columns, clean, labels, splits, features, schemas, pipeline, synthetic
  ids/        Module 1: dataset, models/ (linear, gbm, mlp), weights, calibration, decision,
              explain, metrics, bundle, registry, train, tune, imbalance, report
  anomaly/    Module 2: autoencoder, iforest, evaluate, bundle, registry, train, report
  detection/  fusion.py: supervised + anomaly -> one verdict ("Unknown anomaly")
  services/   api.py (FastAPI)
  cli.py      `sentinel` command; each pipeline subcommand is a DVC stage
tests/        unit/, data/ (pipeline on synthetic data), model/ (training, registry, API)
reports/      data_summary.json, ids/ (results, model card, figures, studies)
dvc.yaml      reproducible pipeline; params.yaml holds every tunable
docker-compose.yml   Postgres, MLflow, Redpanda (+ topic init)
```
