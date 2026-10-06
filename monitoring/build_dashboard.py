"""Generate monitoring/grafana/dashboards/sentinel.json (run after editing the panels).

Conventions (dataviz): one measure per panel (no dual axes); headline numbers as stat tiles;
attack families keep a fixed categorical colour in every panel (palette validated for the
dark surface); severity uses the status palette, never a series colour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# One fixed colour per family (dark-surface steps; Grafana's default theme is dark). Families
# that occur together (Fri: DDoS, PortScan, Bot; Thu: PortScan, WebAttack, Infiltration) pass
# the all-pairs check; "Unknown anomaly" is unclassified, so neutral grey.
FAMILY = {
    "DoS": "#3987e5",
    "DDoS": "#d95926",
    "PortScan": "#9085e9",
    "BruteForce": "#c98500",
    "WebAttack": "#d55181",
    "Bot": "#199e70",
    "Infiltration": "#008300",
    "Unknown anomaly": "#898781",
}
SEVERITY = {"Low": "#898781", "Medium": "#fab219", "High": "#ec835a", "Critical": "#d03b3b"}
DS = {"type": "prometheus", "uid": "prometheus"}

_id = 0


def _next_id() -> int:
    global _id
    _id += 1
    return _id


def _overrides(colors: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": hexv}}],
        }
        for name, hexv in colors.items()
    ]


def stat(title: str, expr: str, unit: str, x: int, y: int, w: int = 4, desc: str = "") -> dict[str, Any]:
    return {
        "id": _next_id(),
        "type": "stat",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": 4},
        "targets": [{"refId": "A", "expr": expr, "datasource": DS}],
        "fieldConfig": {
            "defaults": {"unit": unit, "color": {"mode": "fixed", "fixedColor": "text"}},
            "overrides": [],
        },
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "none",
                    "graphMode": "area", "textMode": "value"},
    }


def series(
    title: str,
    targets: list[tuple[str, str]],
    unit: str,
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    stacked: bool = False,
    bars: bool = False,
    colors: dict[str, str] | None = None,
    desc: str = "",
) -> dict[str, Any]:
    return {
        "id": _next_id(),
        "type": "timeseries",
        "title": title,
        "description": desc,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [
            {"refId": chr(65 + i), "expr": e, "legendFormat": legend, "datasource": DS}
            for i, (e, legend) in enumerate(targets)
        ],
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "min": 0,
                "custom": {
                    "drawStyle": "bars" if bars else "line",
                    "lineWidth": 2,
                    "fillOpacity": 80 if bars else 10,
                    "showPoints": "never",
                    "stacking": {"mode": "normal" if stacked else "none"},
                    "axisSoftMin": 0,
                },
            },
            "overrides": _overrides(colors or {}),
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def row(title: str, y: int) -> dict[str, Any]:
    return {"id": _next_id(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


def q(hist: str, quantile: float, by: str = "") -> str:
    group = f"le{', ' + by if by else ''}"
    return f"histogram_quantile({quantile}, sum by ({group}) (rate({hist}_bucket[2m])))"


def build() -> dict[str, Any]:
    panels: list[dict[str, Any]] = []
    y = 0
    panels.append(row("Live traffic", y)); y += 1
    panels += [
        stat("Flows scored / s", "sum(rate(sentinel_flows_scored_total[1m]))", "ops", 0, y),
        stat("Alerts / min", "sum(rate(sentinel_alerts_total[1m])) * 60", "short", 4, y),
        stat("Consumer lag", "sum(sentinel_consumer_lag)", "short", 8, y,
             desc="Flows waiting in net.flows; a rising lag means the detector is falling behind."),
        stat("Batch time p95", q("sentinel_batch_seconds", 0.95), "s", 12, y,
             desc="Score + explain + store one micro-batch."),
        stat("Scoring time p95", q("sentinel_score_seconds", 0.95), "s", 16, y,
             desc="Model time only; the design target is < 50 ms per micro-batch."),
        stat("Drift share", "max(sentinel_drift_share)", "percentunit", 20, y,
             desc="Share of features whose live distribution differs from the training reference."),
    ]
    y += 4
    panels += [
        series("Alerts per minute by family",
               [("sum by (family) (rate(sentinel_alerts_total[1m])) * 60", "{{family}}")],
               "short", 0, y, stacked=True, bars=True, colors=FAMILY),
        series("Alerts per minute by severity",
               [("sum by (severity) (rate(sentinel_alerts_total[1m])) * 60", "{{severity}}")],
               "short", 12, y, stacked=True, bars=True, colors=SEVERITY),
    ]
    y += 8
    panels += [
        series("Flows scored per second", [("sum(rate(sentinel_flows_scored_total[1m]))", "scored"),
                                           ("sum(rate(sentinel_flows_replayed_total[1m]))", "replayed")],
               "ops", 0, y, w=8),
        series("Micro-batch latency", [(q("sentinel_batch_seconds", 0.5), "p50 batch"),
                                       (q("sentinel_batch_seconds", 0.95), "p95 batch"),
                                       (q("sentinel_score_seconds", 0.95), "p95 scoring")],
               "s", 8, y, w=8),
        series("End-to-end latency (publish -> decision)",
               [(q("sentinel_end_to_end_seconds", 0.5), "p50"), (q("sentinel_end_to_end_seconds", 0.95), "p95")],
               "s", 16, y, w=8),
    ]
    y += 8
    panels.append(row("Model health and drift", y)); y += 1
    panels += [
        series("Drift share (features drifted)", [("max(sentinel_drift_share)", "drift share"),
                                                  ("max(sentinel_drift_threshold)", "retrain threshold")],
               "percentunit", 0, y, w=8),
        series("Most drifted features (PSI)",
               [("topk(5, sentinel_drift_feature_psi)", "{{feature}}")], "short", 8, y, w=8),
        series("Retraining runs", [("sum by (outcome) (increase(sentinel_retrain_runs_total[1h]))", "{{outcome}}")],
               "short", 16, y, w=8, bars=True),
    ]
    y += 8
    panels.append(row("Copilot", y)); y += 1
    panels += [
        series("Reports per minute", [("sum by (provider) (rate(sentinel_copilot_reports_total[5m])) * 60", "{{provider}}")],
               "short", 0, y, w=8),
        series("Report latency", [(q("sentinel_copilot_report_seconds", 0.5), "p50"),
                                  (q("sentinel_copilot_report_seconds", 0.95), "p95")], "s", 8, y, w=8),
        series("LLM tokens per minute", [("sum by (provider, direction) (rate(sentinel_llm_tokens_total[5m])) * 60",
                                          "{{provider}} {{direction}}")], "short", 16, y, w=8),
    ]
    y += 8
    panels.append(row("API", y)); y += 1
    panels += [
        series("Requests per second by route",
               [("sum by (route) (rate(sentinel_http_request_seconds_count[1m]))", "{{route}}")], "reqps", 0, y),
        series("Request latency p95 by route", [(q("sentinel_http_request_seconds", 0.95, "route"), "{{route}}")],
               "s", 12, y),
    ]
    return {
        "uid": "sentinel-overview",
        "title": "Sentinel overview",
        "tags": ["sentinel"],
        "timezone": "browser",
        "refresh": "5s",
        "time": {"from": "now-30m", "to": "now"},
        "schemaVersion": 39,
        "panels": panels,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "grafana" / "dashboards" / "sentinel.json"
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
