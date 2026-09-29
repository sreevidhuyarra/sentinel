"""Canonical column names for CICFlowMeter flow data.

Canonical names are snake_case versions of the corrected CIC-IDS2017 release header
(e.g. 'Flow Bytes/s' -> 'flow_bytes_s'). The original 2017 release uses different
spellings for some columns; ALIASES maps the known ones onto the canonical names.
"""

from __future__ import annotations

import re

import polars as pl

ALIASES: dict[str, str] = {
    "destination_port": "dst_port",
    "source_port": "src_port",
    "source_ip": "src_ip",
    "destination_ip": "dst_ip",
    "total_fwd_packets": "total_fwd_packet",
    "total_backward_packets": "total_bwd_packets",
    "total_length_of_fwd_packets": "total_length_of_fwd_packet",
    "total_length_of_bwd_packets": "total_length_of_bwd_packet",
    "avg_fwd_segment_size": "fwd_segment_size_avg",
    "avg_bwd_segment_size": "bwd_segment_size_avg",
    "init_win_bytes_forward": "fwd_init_win_bytes",
    "init_win_bytes_backward": "bwd_init_win_bytes",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
}

# Never used as model inputs: identifiers invite shortcut learning (the model memorises
# attacker IPs or row ids), and label-derived columns would leak the target.
ID_COLUMNS = ["row_id", "id", "flow_id", "src_ip", "src_port", "dst_ip", "timestamp", "ts", "day"]
LABEL_COLUMNS = ["label", "family", "attempted", "attempted_category", "split"]
NON_FEATURE_COLUMNS = frozenset(ID_COLUMNS + LABEL_COLUMNS)

# Present in both releases; the schema check requires them.
CORE_FEATURES = [
    "dst_port",
    "flow_duration",
    "total_fwd_packet",
    "total_bwd_packets",
    "total_length_of_fwd_packet",
    "total_length_of_bwd_packet",
    "flow_bytes_s",
    "flow_packets_s",
    "fwd_packet_length_max",
    "bwd_packet_length_max",
    "flow_iat_mean",
    "syn_flag_count",
    "ack_flag_count",
]


def canonical(name: str) -> str:
    s = name.strip().lower().replace("/s", "_s")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return ALIASES.get(s, s)


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename to canonical names and drop exact-duplicate columns (the original release
    repeats 'Fwd Header Length', which pandas/polars suffix as '.1' / '_duplicated_0')."""
    renamed: dict[str, str] = {}
    seen: set[str] = set()
    drop: list[str] = []
    for col in df.columns:
        new = canonical(re.sub(r"(\.\d+|_duplicated_\d+)$", "", col))
        if new in seen:
            drop.append(col)
        else:
            seen.add(new)
            renamed[col] = new
    return df.drop(drop).rename(renamed)


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, t in df.schema.items() if c not in NON_FEATURE_COLUMNS and t.is_numeric()]
