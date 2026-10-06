"""Prometheus metrics for the streaming services (report section 9: Metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

FLOWS = Counter("sentinel_flows_scored_total", "Flows scored by the stream detector")
ALERTS = Counter(
    "sentinel_alerts_total", "Alerts raised by the stream detector", ["family", "severity"]
)
BATCH_SECONDS = Histogram(
    "sentinel_batch_seconds",
    "Time to score and store one micro-batch",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
SCORE_SECONDS = Histogram(
    "sentinel_score_seconds",
    "Model time per micro-batch (supervised + anomaly scoring, before explanations and storage)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
BATCH_SIZE = Histogram(
    "sentinel_batch_size", "Flows per micro-batch", buckets=(1, 10, 50, 100, 250, 500, 1000, 2000)
)
END_TO_END = Histogram(
    "sentinel_end_to_end_seconds",
    "From publish (message timestamp) to stored alert decision",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
LAG = Gauge("sentinel_consumer_lag", "Messages waiting in net.flows for the detector")
REPLAYED = Counter("sentinel_flows_replayed_total", "Flows published by the replayer")
MODEL = Info("sentinel_model", "Model versions the detector is serving")
