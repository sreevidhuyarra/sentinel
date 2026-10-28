"""Packet-capture input: CICFlowMeter CSV parsing, the Docker call, and the summary."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from sentinel.stream.pcap import (
    MissingFeaturesError,
    analyze,
    captures_in,
    extract_flows,
    flows_to_rows,
    read_capture_csv,
)

# CICFlowMeter's header (the corrected fork), as the training CSVs have it.
HEADER = (
    "Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Total Fwd Packet,"
    "Total Bwd packets,Total Length of Fwd Packet,Total Length of Bwd Packet,"
    "Fwd Packet Length Max,Fwd Packet Length Min,Fwd Packet Length Mean,Fwd Packet Length Std,"
    "Bwd Packet Length Max,Bwd Packet Length Min,Bwd Packet Length Mean,Bwd Packet Length Std,"
    "Flow Bytes/s,Flow Packets/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,"
    "Fwd IAT Total,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Total,Bwd IAT Mean,"
    "Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,"
    "Fwd RST Flags,Bwd RST Flags,Fwd Header Length,Bwd Header Length,Fwd Packets/s,Bwd Packets/s,"
    "Packet Length Min,Packet Length Max,Packet Length Mean,Packet Length Std,"
    "Packet Length Variance,FIN Flag Count,SYN Flag Count,RST Flag Count,PSH Flag Count,"
    "ACK Flag Count,URG Flag Count,CWR Flag Count,ECE Flag Count,Down/Up Ratio,"
    "Average Packet Size,Fwd Segment Size Avg,Bwd Segment Size Avg,Fwd Bytes/Bulk Avg,"
    "Fwd Packet/Bulk Avg,Fwd Bulk Rate Avg,Bwd Bytes/Bulk Avg,Bwd Packet/Bulk Avg,"
    "Bwd Bulk Rate Avg,Subflow Fwd Packets,Subflow Fwd Bytes,Subflow Bwd Packets,"
    "Subflow Bwd Bytes,FWD Init Win Bytes,Bwd Init Win Bytes,Fwd Act Data Pkts,Fwd Seg Size Min,"
    "Active Mean,Active Std,Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,ICMP Code,"
    "ICMP Type,Total TCP Flow Time,Label"
)
N_FEATURES = len(HEADER.split(",")) - 8  # all but flow id, IPs, ports, protocol, time, label


def _row(src: str, port: int, flow_bytes: str) -> str:
    feats = ["3"] * N_FEATURES
    feats[13] = flow_bytes  # Flow Bytes/s: CICFlowMeter writes Infinity for 0-duration flows
    return ",".join(
        [f"{src}-10.0.0.9-{port}-80-6", src, str(port), "10.0.0.9", "80", "6",
         "29/10/2026 03:15:42 PM", *feats, "NeedManualLabel"]
    )  # fmt: skip


@pytest.fixture()
def capture_csv(tmp_path: Path) -> Path:
    p = tmp_path / "lab_Flow.csv"
    p.write_text(
        "\n".join([HEADER, _row("10.0.0.5", 40000, "1200.5"), _row("10.0.0.6", 40001, "Infinity")])
        + "\n"
    )
    return p


INPUTS = ["dst_port", "flow_bytes_s", "syn_flag_count", "total_tcp_flow_time", "icmp_type"]


def test_capture_csv_is_parsed_like_the_training_data(capture_csv: Path) -> None:
    df = read_capture_csv(capture_csv, INPUTS)
    assert df.height == 2 and "label" not in df.columns
    assert df["flow_bytes_s"].to_list() == [1200.5, 0.0]  # Infinity -> 0, as in cleaning
    assert df["ts"].dt.hour().to_list() == [15, 15]  # 12-hour clock with PM
    assert df["src_ip"].to_list() == ["10.0.0.5", "10.0.0.6"]
    assert set(df["capture"]) == {"lab_Flow"}


def test_missing_inputs_are_named(capture_csv: Path) -> None:
    with pytest.raises(MissingFeaturesError, match="not_a_feature"):
        read_capture_csv(capture_csv, [*INPUTS, "not_a_feature"])


def test_extract_flows_runs_the_pinned_tool_offline(tmp_path: Path, capture_csv: Path) -> None:
    pcap = tmp_path / "lab.pcap"
    pcap.write_bytes(b"\xd4\xc3\xb2\xa1")
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":/tmp")[0])
        (out / "lab.pcap_Flow.csv").write_text(capture_csv.read_text())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    csvs = extract_flows([pcap], tmp_path / "flows", image="flowmeter:test", run=fake_run)
    assert [c.name for c in csvs] == ["lab.pcap_Flow.csv"]
    cmd = seen["cmd"]
    assert cmd[:3] == ["docker", "run", "--rm"] and "--network" in cmd and "none" in cmd
    assert "flowmeter:test" in cmd and any(a.endswith(":/tmp/pcap/input:ro") for a in cmd)

    def failing(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "bad capture")

    with pytest.raises(RuntimeError, match="bad capture"):
        extract_flows([pcap], tmp_path / "flows2", run=failing)


def test_captures_in_finds_pcaps(tmp_path: Path) -> None:
    (tmp_path / "a.pcap").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.PCAPNG").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    assert [p.name for p in captures_in(tmp_path)] == ["a.pcap", "b.PCAPNG"]
    with pytest.raises(FileNotFoundError):
        captures_in(tmp_path / "sub" / "missing")


class FakeDetector:
    """Flags flows from 10.0.0.6 as PortScan, reviews nothing."""

    def predict(self, frame: pl.DataFrame, explain: bool) -> list[dict[str, Any]]:
        return [
            {
                "family": "PortScan" if v == 0.0 else "Benign",
                "is_attack": v == 0.0,
                "detector": "supervised" if v == 0.0 else None,
                "confidence": 0.9 if v == 0.0 else None,
                "attack_score": 0.9 if v == 0.0 else 0.01,
                "needs_review": False,
            }
            for v in frame["flow_bytes_s"]
        ]


def test_analyze_summarises_verdicts(capture_csv: Path) -> None:
    flows = read_capture_csv(capture_csv, INPUTS)
    s = analyze(flows, FakeDetector(), INPUTS, batch=1)
    assert (s["flows"], s["alerts"], s["review_flags"]) == (2, 1, 0)
    assert s["by_family"] == {"PortScan": 1} and s["alert_share"] == 0.5
    assert s["top_sources"] == [{"src_ip": "10.0.0.6", "alerts": 1}]
    assert s["top_alerts"][0]["dst_port"] == 80 and s["captures"] == ["lab_Flow"]
    rows = list(flows_to_rows(flows))
    assert len(rows) == 2 and "label" not in rows[0]
