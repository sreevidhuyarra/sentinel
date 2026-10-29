"""Real traffic in: packet captures -> CICFlowMeter flows -> the detector.

Flows are extracted with the corrected CICFlowMeter (DistriNet's fork) at commit 55f60ad
(July 2022), the version whose features match the training data, so every feature is
computed as the models learned it. Later versions rename Total TCP Flow Time and change the
subflow and backward-window calculations; a re-implementation would differ in small ways
(flow timeouts, FIN handling, active/idle times). Either silently shifts the features. The
tool runs in Docker (docker/flowmeter); this module only calls it and reads its CSVs.

Two ways out:
- `analyze`: score every flow locally and summarise (no Kafka or database needed).
- `flows_to_rows` + `replayer.replay`: publish to net.flows, so the stream detector stores
  the alerts and they appear in the dashboard like replayed traffic.

Captures carry no labels: verdicts are what the detector thinks, not ground truth.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from sentinel.common.logging import get_logger
from sentinel.data.columns import feature_columns
from sentinel.data.ingest import parse_flow_csv

log = get_logger(__name__)

FLOWMETER_IMAGE = "sentinel-flowmeter:55f60ad"
PCAP_SUFFIXES = (".pcap", ".pcapng", ".cap")
META = ("ts", "src_ip", "src_port", "dst_ip", "dst_port", "protocol")
TOP = 10

Run = Callable[..., subprocess.CompletedProcess[str]]


class MissingFeaturesError(ValueError):
    """The CSV lacks inputs the detector needs (a different CICFlowMeter version?)."""


def captures_in(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    found = [p for p in files if p.suffix.lower() in PCAP_SUFFIXES]
    if not found:
        raise FileNotFoundError(f"no {'/'.join(PCAP_SUFFIXES)} files at {path}")
    return found


def extract_flows(
    pcaps: Sequence[Path], out_dir: Path, image: str = FLOWMETER_IMAGE, run: Run = subprocess.run
) -> list[Path]:
    """Run the pinned CICFlowMeter over the captures; returns the CSVs it wrote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sentinel-pcap-") as tmp:
        inp = Path(tmp) / "input"
        inp.mkdir()
        for p in pcaps:  # the tool reads a folder; copy so only these captures are read
            shutil.copy2(p, inp / p.name)
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{inp.resolve()}:/tmp/pcap/input:ro",
            "-v", f"{out_dir.resolve()}:/tmp/pcap/output",
            image, "/tmp/pcap/input", "/tmp/pcap/output",
        ]  # fmt: skip
        log.info("extracting flows from %d capture(s) with %s", len(pcaps), image)
        result = run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"CICFlowMeter failed ({result.returncode}): {result.stderr[-2000:]}"
            )
    csvs = sorted(out_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"CICFlowMeter wrote no CSV to {out_dir}")
    return csvs


def read_capture_csv(path: Path, inputs: Sequence[str]) -> pl.DataFrame:
    """Flows as the detector expects them: canonical names, non-finite rates set to 0 (the
    same rule as the training data's cleaning), every input present."""
    df = parse_flow_csv(path).drop("label", strict=False)
    missing = sorted(set(inputs) - set(df.columns))
    if missing:
        raise MissingFeaturesError(f"{path.name} lacks detector inputs: {missing}")
    floats = [c for c in feature_columns(df) if df.schema[c].is_float()]
    df = df.with_columns(
        [pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c) for c in floats]
    ).with_columns([pl.col(c).fill_null(0) for c in feature_columns(df)])
    return df.with_columns(pl.lit(path.stem).alias("capture"))


def analyze(
    flows: pl.DataFrame, detector: Any, inputs: Sequence[str], batch: int = 5000
) -> dict[str, Any]:
    """Score every flow and summarise: verdicts by family and detector, busiest alert
    sources, and the most confident alerts with their connection details."""
    results: list[dict[str, Any]] = []
    for start in range(0, flows.height, batch):
        part = flows.slice(start, batch)
        results += detector.predict(part.select(inputs).cast(pl.Float64), explain=False)
    meta = flows.select([c for c in META if c in flows.columns]).to_dicts()
    alerts = [
        {**m, **{k: r[k] for k in ("family", "detector", "confidence", "attack_score")}}
        for m, r in zip(meta, results, strict=True)
        if r["is_attack"] or r.get("needs_review")
    ]
    flagged = [r for r in results if r["is_attack"]]
    sources = Counter(str(a.get("src_ip")) for a in alerts)
    top = sorted(alerts, key=lambda a: a.get("attack_score") or 0.0, reverse=True)[:TOP]
    return {
        "flows": len(results),
        "alerts": len(flagged),
        "review_flags": sum(1 for r in results if r.get("needs_review")),
        "alert_share": len(flagged) / len(results) if results else 0.0,
        "by_family": dict(Counter(r["family"] for r in flagged).most_common()),
        "by_detector": dict(Counter(str(r["detector"]) for r in flagged).most_common()),
        "top_sources": [{"src_ip": s, "alerts": n} for s, n in sources.most_common(TOP)],
        "top_alerts": [{k: _plain(v) for k, v in a.items()} for a in top],
        "captures": sorted(flows["capture"].unique().to_list()) if "capture" in flows else [],
    }


def flows_to_rows(flows: pl.DataFrame) -> Iterator[dict[str, Any]]:
    """Rows for `replayer.replay`, in time order. No `label`/`family`: captures are
    unlabelled, so the stream detector stores no ground truth for them."""
    yield from flows.sort("ts", nulls_last=True).iter_rows(named=True)


def _plain(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v
