"""Small synthetic dataset in the corrected CIC-IDS2017 CSV format, for tests and CI.

It reproduces the real files' header, per-day attack schedule, 'Attempted' labels,
non-finite rate values and duplicate rows, so the pipeline exercises every cleaning
path without the 1 GB download. Values are plausible, not realistic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

HEADER = [
    "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd RST Flags", "Bwd RST Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min",
    "Packet Length Max", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max",
    "Idle Min", "ICMP Code", "ICMP Type", "Total TCP Flow Time", "Label", "Attempted Category",
]  # fmt: skip

# (label, dst_port, rows, mean_fwd_packets, mean_duration_us) per day, in time order.
SCHEDULE: dict[str, list[tuple[str, int, int, float, float]]] = {
    "monday": [("BENIGN", 443, 1500, 12, 2e6)],
    "tuesday": [
        ("BENIGN", 443, 900, 12, 2e6),
        ("FTP-Patator", 21, 300, 8, 3e5),
        ("FTP-Patator - Attempted", 21, 20, 2, 1e4),
        ("SSH-Patator", 22, 300, 20, 1e6),
    ],
    "wednesday": [
        ("BENIGN", 80, 900, 12, 2e6),
        ("DoS Hulk", 80, 400, 5, 5e4),
        ("DoS Slowloris", 80, 150, 4, 8e7),
        ("DoS Slowloris - Attempted", 80, 40, 1, 1e3),
        ("Heartbleed", 444, 11, 2000, 1e8),
    ],
    "thursday": [
        ("BENIGN", 443, 900, 12, 2e6),
        ("Web Attack - Brute Force", 80, 70, 10, 5e6),
        ("Web Attack - XSS", 80, 30, 10, 5e6),
        ("Infiltration", 444, 30, 50, 3e7),
        ("Infiltration - Portscan", 445, 300, 1, 50),
    ],
    "friday": [
        ("BENIGN", 443, 900, 12, 2e6),
        ("Botnet", 8080, 120, 6, 4e5),
        ("Portscan", 1024, 500, 1, 60),
        ("DDoS", 80, 500, 3, 2e4),
    ],
}
_DATES = {d: datetime(2017, 7, 3 + i, 9, 0) for i, d in enumerate(SCHEDULE)}


def _block(
    rng: np.random.Generator,
    day: str,
    label: str,
    port: int,
    n: int,
    fwd: float,
    dur: float,
    start: datetime,
) -> dict[str, list[object] | np.ndarray]:
    fwd_pk = rng.poisson(fwd, n) + 1
    bwd_pk = rng.poisson(max(fwd * 0.8, 0.5), n)
    fwd_len = fwd_pk * rng.integers(40, 1400, n)
    bwd_len = bwd_pk * rng.integers(0, 1400, n)
    duration = np.maximum(rng.exponential(dur, n).astype(np.int64), 0)
    duration[rng.random(n) < 0.02] = 0  # zero-duration flows -> infinite rates
    total_pk = fwd_pk + bwd_pk
    with np.errstate(divide="ignore", invalid="ignore"):
        bytes_s = (fwd_len + bwd_len) / (duration / 1e6)
        pk_s = total_pk / (duration / 1e6)
    ts = [start + timedelta(milliseconds=int(x)) for x in np.sort(rng.integers(0, 3_600_000, n))]
    cols: dict[str, list[object] | np.ndarray] = {c: np.zeros(n, dtype=np.int64) for c in HEADER}
    cols.update(
        {
            "Flow ID": [
                f"172.16.0.1-192.168.10.50-{p}-{port}-6" for p in rng.integers(1024, 65535, n)
            ],
            "Src IP": ["172.16.0.1"] * n,
            "Src Port": rng.integers(1024, 65535, n),
            "Dst IP": ["192.168.10.50"] * n,
            "Dst Port": np.full(n, port),
            "Protocol": np.full(n, 6),
            "Timestamp": [t.strftime("%Y-%m-%d %H:%M:%S.%f") for t in ts],
            "Flow Duration": duration,
            "Total Fwd Packet": fwd_pk,
            "Total Bwd packets": bwd_pk,
            "Total Length of Fwd Packet": fwd_len,
            "Total Length of Bwd Packet": bwd_len,
            "Fwd Packet Length Max": fwd_len // fwd_pk + rng.integers(0, 100, n),
            "Fwd Packet Length Mean": fwd_len / fwd_pk,
            "Bwd Packet Length Max": np.where(bwd_pk > 0, bwd_len // np.maximum(bwd_pk, 1), 0),
            "Flow Bytes/s": bytes_s,
            "Flow Packets/s": pk_s,
            "Flow IAT Mean": duration / np.maximum(total_pk - 1, 1),
            "SYN Flag Count": rng.integers(0, 2, n),
            "ACK Flag Count": rng.integers(0, 2, n),
            "FWD Init Win Bytes": rng.choice([-1, 29200, 65535], n),
            "Fwd PSH Flags": np.zeros(n, dtype=np.int64),  # constant column, as in the real data
            "Label": [label] * n,
            "Attempted Category": np.full(n, 0 if label.endswith("Attempted") else -1),
        }
    )
    return cols


def generate(out_dir: Path, seed: int = 0) -> list[Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, next_id = [], 1
    for day, blocks in SCHEDULE.items():
        frames = []
        for i, (label, port, n, fwd, dur) in enumerate(blocks):
            start = _DATES[day] + timedelta(hours=i)
            frames.append(pl.DataFrame(_block(rng, day, label, port, n, fwd, dur, start)))
        df = pl.concat(frames, how="vertical_relaxed").sort("Timestamp")
        df = pl.concat([df, df.head(5)], how="vertical_relaxed")  # exact duplicates
        df = df.with_columns(pl.int_range(next_id, next_id + df.height).alias("id"))
        next_id += df.height
        path = out_dir / f"{day}.csv"
        df.select(HEADER).write_csv(path)
        paths.append(path)
    return paths
