[← Back to the README](../README.md)

# Module 6: streaming, MLOps, dashboard and API

Module 6 joins the modules into one running system:

```
replayer --net.flows--> stream detector --alerts.new-->  API (REST + WebSocket) --> React dashboard
(test-split flows,      (micro-batches: fused            /metrics, /stats, /mlops
 real timestamps)        verdict, SHAP once per                |
                         campaign, review flag)           Prometheus --> Grafana
                             |
                         Postgres: alerts + 5% flow sample --> drift watcher (Evidently PSI)
                                                                 --> Prefect retrain flow --> MLflow
```

| Piece | What it does |
|---|---|
| Replayer | `sentinel stream replay --days friday --rate 500` publishes flows to Redpanda (`net.flows`) at a set rate, in time order. Each message carries the flow's ground truth separately from its features, so the detector never sees it. |
| Stream detector | `sentinel stream detect` consumes micro-batches (up to 500 flows or 0.2 s) and scores them with the fused Module 1+2 detector. It stores alerts, publishes them to `alerts.new`, and commits Kafka offsets only after the database write. SHAP reasons are computed once per campaign (source, target and family); later members point at the explained alert. |
| Review flag | The Module 4 defense is now served. A flow passed as benign whose LightGBM and MLP members disagree by more than the validation-chosen gap becomes a Low "Needs review" alert, in the stream and in `POST /score/flows` (`needs_review`, `member_gap`). |
| Drift watcher | `sentinel mlops drift-watch` checks every minute. **Drift:** the last 10 minutes of sampled live flows against benign training flows, as the share of features with Evidently PSI above 0.1. It measures two windows: all flows, and flows the detector passed as benign. **Live performance:** on the labelled flows of the same window, the attack recall and the false-alarm rate (any alert counts, review flags included). A retrain starts after two consecutive checks with benign-window drift of 20% or more, recall below 90%, or false alarms above 4%, at most once an hour. |
| Retraining | Prefect flow `sentinel-retrain` (`sentinel mlops retrain`): labelled live flows join the training split, and LightGBM + MLP are retrained. Validation and test rows identical to a live flow are dropped first. Production is re-scored on the same test rows, and the candidate is promoted only if a paired bootstrap puts its macro-F1 gain above zero, within the false-alarm budget. Every run is logged in the `retrain_runs` table. |
| Metrics | Prometheus scrapes the detector (:9101), drift watcher (:9103) and API (`/metrics`): throughput, batch and scoring latency, end-to-end latency, consumer lag, alerts by family and severity, drift share per window, live recall and false-alarm rate, retrain runs, copilot reports, LLM tokens and guard flags. Grafana (:3000) provisions a 25-panel dashboard (`monitoring/build_dashboard.py`). |
| Dashboard | React 19 + TypeScript + Vite + Tailwind + Recharts, served by the API at `/app/`. Pages: overview (live alert feed over WebSocket), alerts with filters, alert detail with SHAP reasons and related alerts, copilot report (Markdown export), phishing check, model health (drift, retrains, embedded Grafana) and robustness (run a pad-and-delay evasion test live). Each family keeps one colour, validated for colour-blind separation in light and dark mode. |

### Results

**Throughput.** On the laptop CPU, scoring alone sustained about 1,440 flows/s,
consuming as fast as the replayer could publish. Scoring a full 500-flow batch takes
about 46–49 ms (p95 48.6 ms, target under 50 ms). Explaining every alert
with SHAP cut throughput to about 220 flows/s. Explaining once per campaign restored it.

Those numbers are for the detector alone. With everything on one laptop (the replayer
at 500 flows/s, API, drift job, Docker services, and a browser refreshing Grafana every
10 s), a Friday replay averaged **84 ms** of scoring per batch of ~320 flows. 80% of batches
finished under 100 ms, and p95 fell between 0.1 and 0.25 s. The detector kept up with the
replay, but the < 50 ms target holds only when it has the CPU to itself.

**Drift demo** (`scripts/drift_demo.ps1`: Monday's test flows, then Friday's, at 500 flows/s):

| Phase | Drifted share (all flows) | Drifted share (benign window) |
|---|---|---|
| Monday (benign) | 0–1% | 0–1% |
| Friday (DDoS, PortScan, Bot) | 8.5% → 20.7% → 31.7% | 0% |

The demo first ran with the original trigger, all-window drift. Two consecutive checks at
20.7% started a retrain, and the cooldown stopped a second one at 31.7%. But the benign
window stayed at 0% throughout. What drifted was the attack mix (packet sizes, flag counts,
destination ports), not normal traffic. Over the same windows, the detector caught **99.8%**
of the labelled live attacks, and its false alarms stayed at 0.1–3.1%. So the alarm meant
"the traffic looks different", not "the model got worse", and that retrain was wasted work.

The trigger was changed as a result. It now watches the benign window and live performance,
and the all-window share is only displayed. Replaying the demo's 17 checks through the new
rule starts no retrain.

Live false alarms are 1–2% by design, so the 4% budget is double that. Of the labelled benign
flows, 1.55% became "Unknown anomaly" (the autoencoder's threshold allows 1% on validation),
0.11% became review flags, and 0.01% were given a known family. During the demo, which shared
the CPU with a retrain, the live end-to-end latency (replay to stored alert) was p50 0.42 s
and p95 1.8 s.

**Retraining, and a leak it exposed.** The first drift-triggered run trained on 7,870
labelled live flows and reported test macro-F1 0.9957 → 0.9988, so it was promoted. That
number was invalid: the replayer streams the **test** split, so the candidate was scored on the
flows it had just trained on. The fix: before retraining, validation and test rows identical
to a live flow are dropped (here 7,879 test rows and 6 validation rows). Production is then
re-scored on that same reduced test set, and the promotion decision compares the two
(`sentinel.ids.dataset.without_overlap`, `production_test_macro_f1` in MLflow). The leaked
v4 was rolled back.

The corrected re-run scored 0.9988 against production's 0.9957 on the same 360,061 test flows.
Almost all of that gap is one WebAttack flow: 21 of 22 caught before, 22 of 22 after.
Macro-F1 averages over eight families, so one flow in a 22-flow family moves it by about 0.003.

That raised the question of whether "beats production" means anything at this level. So
promotion now also needs a **paired bootstrap** over the test flows, for every training run
and not only retraining
(`ids.metrics.paired_bootstrap_f1`, 2,000 resamples): the 95% interval of the candidate's
macro-F1 gain must lie above zero. Each resample is a multinomial draw over the
(true, candidate, production) cells, so this is exact and takes milliseconds. On this candidate
the gain is 0.0031, with a 95% interval of 0.0001 to 0.0104, and no resample favours
production. The gain is tiny but consistent. Besides the WebAttack flow, the candidate is
slightly better on PortScan (F1 0.9982 → 0.9990; 899 live PortScan flows were in its training
data), BruteForce, DoS and benign; DDoS was unchanged. The bootstrap covers
test-sample noise only, not variation between training seeds.

The candidate passed, but production stays on **v3**. The candidate learned from 7,870
test-split flows, and every later evaluation in this repo scores the full test split: the
copilot's alert database, the robustness page and the replay demo. Deploying it would quietly
inflate all of them. In a real deployment, live traffic is new and this conflict does not arise.

Run it:

```bash
docker compose up -d --wait                        # Postgres, MLflow, Redpanda, Prometheus, Grafana
uv run sentinel serve                              # API + dashboard: http://127.0.0.1:8000/app/
uv run sentinel stream detect                      # in a second terminal
uv run sentinel mlops drift-watch                  # in a third
uv run sentinel stream replay --days monday,friday --rate 500
```

On Windows, `powershell -ExecutionPolicy Bypass -File scripts\drift_demo.ps1` starts the detector
and drift watcher and replays both days. Drift shows on the dashboard and in Grafana, but
with the default trigger no retrain starts, because the model keeps catching Friday's attacks.
To see a retrain, set `mlops.trigger_window: all` in `params.yaml`. To run everything in containers instead:
`docker compose --profile app up -d --build`. The demo replayer:
`docker compose --profile app --profile demo run --rm replayer stream replay --days friday`.
The dashboard is at http://localhost:8000/app/, Grafana at http://localhost:3000 and
Prometheus at http://localhost:9090.

**Ports.** Every host port can be changed in `.env` (`SENTINEL_PORT`, `POSTGRES_PORT`,
`MLFLOW_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`), so Sentinel can run beside another project
that already holds 8000 or 5432. `sentinel serve`, docker compose and the Vite dev proxy all
read `SENTINEL_PORT`; `.env.example` lists the URLs that must follow a changed port.

**Start-up.** The API downloads each registry version once into `data/cache/models` and loads
it from there on later starts. That saves about 10 s per start against a local MLflow; most of
the start-up time is loading the models into memory. The copilot's models (guard, embedder and
re-ranker) load in a background thread while the API already serves scoring and alerts, so
the first investigation is not slow. Two requests for the same alert share one investigation
and one LLM call. The API gives Gemini 20 s and one retry before falling back to the local
model; evaluations keep the patient settings. The live alert feed queries the database off the
event loop, and connections give up after 5 s, so a stopped database cannot freeze the API.

**Analyst feedback.** Every alert page has **✓ Confirmed attack** and **✕ False alarm**
buttons, with an optional corrected family and a note. Every phishing check asks "What is it
really?" (phishing or legitimate). Verdicts go to the `feedback` table, which the copilot
cannot read, and are used three ways:

- **Live precision** on Model health: alert precision overall and per family, email alert
  precision, and phishing that passed as clean. These count only what analysts labelled
  (the latest verdict per alert), so they describe reviewed items, not all traffic.
- **Training labels.** A verdict labels the alert's sampled flows that have no label yet, so
  live traffic gains labels for the retraining flow; a false alarm becomes Benign. Replayed
  flows keep their dataset label.
- **Fresh email test data.** Labelled emails are stored with what was checked and what the
  models said. `sentinel phishing feedback-export` writes them to a parquet file: the
  untouched test data that the look-alike fix needs.

**Access.** Every compose port binds to 127.0.0.1, so nothing is reachable from the network.
The API and dashboard have no login unless `SENTINEL_API_KEY` is set in `.env`. With it set,
they require HTTP Basic auth: any user name, the key as password. The browser asks once and
resends the credentials, so the dashboard and its live feed need no login page; `/health`
and `/metrics` stay open for health checks and Prometheus. `sentinel serve --host 0.0.0.0`
warns if no key is set. Grafana, MLflow, Prometheus and Postgres have no auth of their own
here, so keep them on localhost.

### Real traffic: packet captures

`sentinel stream pcap <capture.pcap | folder>` scores real network traffic instead of the
replayed dataset:

1. **Extract flows.** The flows are computed by DistriNet's corrected CICFlowMeter at
   commit `55f60ad` (29 July 2022). It runs in Docker with no network access. Build it once
   with `docker compose --profile pcap build flowmeter` (`docker/flowmeter/Dockerfile`).
   - **Why that version.** It is the one behind the training data. The dataset page names no
     commit, but its documented changes (Total TCP Duration, ICMP, RST counts, UTC
     microsecond timestamps, the segmentation-offload fix) end in July 2022. The tool's
     source at that commit has exactly the dataset's columns.
   - **Why not the latest.** The newest version (`4dd5319`, 2024) was tried first. It
     renames Total TCP Flow Time and adds retransmission columns, and its 2023 fixes change
     how subflows and the backward initial window are computed. Its output would silently
     shift the models' inputs.
   - **Why not a re-implementation.** It would differ in small ways (flow timeouts, FIN
     handling, active and idle times) that shift the features the same way.
2. **Parse as in training.** The CSVs go through the same parser as the dataset ingest
   (`data.ingest.parse_flow_csv`), with the same rule for undefined rates.
3. **Score.** Every flow is scored by the deployed detector, and a summary is written to
   `reports/pcap/<name>.json`: alerts by family and detector, the busiest sources, and the
   most confident alerts. `--publish` also streams the flows to `net.flows`, so the stream
   detector stores the alerts and they appear in the dashboard.

Captures carry no labels, so the summary reports what the detector thinks, not accuracy.
Expect false alarms on traffic unlike CIC-IDS2017: the cross-network study found that these
models do not transfer between networks.

Path check: 6,000 flows from the Friday CSV (output of the same tool) went through steps 2
and 3. All 82 inputs were mapped, and the verdicts matched the labels: 1,713 PortScan alerts
for 1,715 PortScan flows, 1,052 DDoS for 1,052. Bot was mostly missed (9 of 62), the known
weak spot. The sample includes training-period rows, so this checks the plumbing, not
accuracy.

### Security checks in CI

Three workflows keep the project itself secure:

- **Dependencies (`security.yml`).** `pip-audit` checks every locked Python dependency, and
  `npm audit` checks the dashboard's packages, build tools included. It runs on every push and
  pull request, and weekly, because new advisories appear for unchanged dependencies.
  - The CPU build of PyTorch is not on PyPI, so it is audited under its plain version number.
  - One advisory is ignored, with the reason, until a fix exists: nltk's model-file path
    handling (PYSEC-2026-3740). nltk arrives only through Evidently, and Sentinel never calls
    the affected APIs.
- **Secrets (`security.yml`).** gitleaks scans the whole git history for API keys, tokens
  and private keys.
- **Code scanning (`codeql.yml`).** GitHub's CodeQL analyses the Python service and the
  TypeScript dashboard with the `security-extended` queries. Findings appear under the
  repository's Security tab.
- **Updates (`.github/dependabot.yml`).** Weekly pull requests for Python (uv), npm, GitHub
  Actions and Docker base images. Minor and patch updates are grouped into one pull request per
  ecosystem, and every update must pass CI and the audits before it can be merged.

When the checks were added, the audit found known vulnerabilities in 11 locked packages,
including Pillow (39 advisories), aiohttp, filelock, GitPython, mlflow and Werkzeug, and one in
Vite. All were upgraded to fixed versions; the test suite passes on them, and the MLflow 3.17
client still loads every production model from the 3.16 server. The demo Space stays on
Pillow 12.3 because Gradio 6.31 requires Pillow below 13. The Pillow advisories concern image
decoding, which the demo never does.

### Public demo (Hugging Face Spaces)

`SENTINEL_DEMO=1` turns the API into a read-only public demo (`sentinel/demo.py`):

- **Bundled models only.** It loads the local model bundles and never contacts MLflow, and
  the copilot's models are never loaded.
- **A snapshot instead of Postgres.** `sentinel demo snapshot` copies the alert database into
  a 23 MB SQLite file: one alert per replayed flow (38,521 of the ~324,000 rows that repeated
  replays had written), every alert a stored report refers to, the reports, drift checks
  and retraining runs.
- **Writes and spending are off.** Analyst feedback and the live evasion run return 403.
  "Investigate" opens the stored report for an alert instead of calling an LLM.
- **Rate-limited.** Scoring allows 20 requests a minute per visitor (the first
  `X-Forwarded-For` address behind the Spaces proxy).
- **Recorded, and labelled as such.** A banner says what is recorded and what runs live. The
  live alert feed replays the snapshot's alerts in time order (three a second). The
  throughput chart shows a Prometheus series measured during real replays (peak 507 flows/s).
  Alerts without a stored report link to the one for the same campaign or family.

The phishing check runs live, including blocklists and RDAP lookups. `scripts/stage_hf_space.py`
assembles the Space (code, deployed bundles, policies, snapshot, built dashboard; about 320 MB)
and refuses to include `.env` or Phishing Pot data.

The Space uses the Gradio SDK, because Docker Spaces are paid on some accounts. `app.py`
serves the same FastAPI app with a small Gradio page mounted at `/gradio`. It pins PyTorch
2.13, the newest version ZeroGPU hardware supports, and installs the other dependencies at
their `uv.lock` versions. `deploy/huggingface/PUBLISH.md` has the steps.

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

Alerts and the copilot: `GET /alerts?family=&severity=&source=&src_ip=&since=&limit=` lists
alerts, and `GET /alerts/{id}` returns one with its detector reasons and evidence.
`POST /copilot/investigate/{id}` runs the investigation graph and stores the report; it takes
seconds with Gemini and about a minute on the local model. `GET /reports/{report_id}` returns
a stored report.

Dashboard endpoints: `WS /ws/alerts` pushes new alerts; `GET /stats/overview`,
`/stats/throughput`, `/alerts/{id}/related`, `/mlops/drift`, `/mlops/retrains` and
`/robustness/study` feed the pages; `POST /robustness/run` runs a pad-and-delay evasion test
on test-split attack flows. Feedback: `POST /feedback/alerts/{id}`
(`{"verdict": "confirmed" | "false_alarm", "family", "note"}`), `GET /feedback/alerts/{id}`,
`POST /feedback/email` and `GET /feedback/summary`. `GET /metrics` is the Prometheus endpoint, and `/app/` serves the
built dashboard (`cd frontend && npm ci && npm run build`).

## Model registry

Each training run logs parameters, metrics, figures and the model bundle to MLflow
(experiment `sentinel-ids`, http://localhost:5000) and registers a new `ids-classifier`
version with alias `@staging`. It moves to `@production` only if all of these hold:
- It was trained on the full data.
- It keeps benign FPR within budget.
- It beats the current production model on test macro-F1, with production **re-scored on
  the same test rows**.
- A paired bootstrap over those rows (2,000 resamples, `ids.promotion_bootstrap`) puts the
  95% interval of the gain above zero.

A tie, or a one-flow win in a rare family, is not promoted. The comparison falls back to
production's logged score only if production cannot be loaded or needs features this data
does not have.
