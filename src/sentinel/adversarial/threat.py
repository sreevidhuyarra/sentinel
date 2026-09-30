"""Threat model: what an attacker can change about its own flows, and how.

Goal: evasion, i.e. an attack flow scored Benign. The attacker controls its own packets
and timing but not the victim's replies, the destination port, the protocol or TCP
flags. It can only *add* padding and delay: removing payload or speeding up the network
would break the attack. Two attack spaces follow:

- Feature space (upper bound): any controllable feature may move within an L-inf budget in
  standardised units; timing and own-size features only upwards; everything clipped to
  the range seen in training.
- Problem space (realistic): a flow is padded by a fraction `pad` and delayed by a factor
  `delay`, and every dependent feature (rates, means, ratios) is recomputed consistently.
"""

from __future__ import annotations

import numpy as np
import polars as pl

TIMING = [
    "flow_duration",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_total",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_total",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
    "total_tcp_flow_time",
]
OWN_SIZE = [
    "total_length_of_fwd_packet",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "fwd_segment_size_avg",
    "subflow_fwd_bytes",
    "fwd_bytes_bulk_avg",
]
DEPENDENT = [
    "flow_bytes_s",
    "flow_packets_s",
    "fwd_packets_s",
    "bwd_packets_s",
    "fwd_bulk_rate_avg",
    "packet_length_max",
    "packet_length_mean",
    "packet_length_std",
    "packet_length_variance",
    "average_packet_size",
    "fwd_bwd_bytes_ratio",
]
OWN_TCP = ["fwd_init_win_bytes"]

CONTROLLABLE = TIMING + OWN_SIZE + DEPENDENT + OWN_TCP
INCREASE_ONLY = TIMING + OWN_SIZE


def masks(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(controllable, increase_only) boolean masks over a feature list."""
    ctrl = np.array([c in CONTROLLABLE for c in columns])
    inc = np.array([c in INCREASE_ONLY for c in columns])
    return ctrl, inc


def project(
    z_adv: np.ndarray,
    z: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    controllable: np.ndarray,
    increase_only: np.ndarray,
) -> np.ndarray:
    """Make a feature-space perturbation valid: fixed features restored, padding/delay
    features never below the original, everything within the training range. Features
    are standardised monotone transforms of raw values, so order is preserved."""
    out = np.where(controllable, z_adv, z)
    out = np.where(increase_only, np.maximum(out, z), out)
    return np.clip(out, lo, hi)


def pad_and_delay(frame: pl.DataFrame, pad: float, delay: float) -> pl.DataFrame:
    """Raw flows as they would look if every forward packet carried `pad` x more payload
    and all timing were stretched by `delay` (>= 1).

    Byte-derived statistics are rescaled by the change in total bytes (packet counts are
    unchanged); packet-length std is rescaled the same way, an approximation because the
    per-packet mix is not recorded in the flow summary. pad=0, delay=1 is the identity.
    """
    if pad < 0 or delay < 1:
        raise ValueError("attacker can only add padding (pad >= 0) and delay (delay >= 1)")
    s = 1.0 + pad
    fwd = pl.col("total_length_of_fwd_packet").cast(pl.Float64)
    bwd = pl.col("total_length_of_bwd_packet").cast(pl.Float64)
    old_total = fwd + bwd
    new_total = fwd * s + bwd
    byte_ratio = pl.when(old_total > 0).then(new_total / old_total).otherwise(1.0)
    out = frame.with_columns(
        [(pl.col(c).cast(pl.Float64) * s).alias(c) for c in OWN_SIZE if c in frame.columns]
        + [(pl.col(c).cast(pl.Float64) * delay).alias(c) for c in TIMING if c in frame.columns]
        + [
            (pl.col("flow_bytes_s") * byte_ratio / delay).alias("flow_bytes_s"),
            (pl.col("flow_packets_s") / delay).alias("flow_packets_s"),
            (pl.col("fwd_packets_s") / delay).alias("fwd_packets_s"),
            (pl.col("bwd_packets_s") / delay).alias("bwd_packets_s"),
            (pl.col("fwd_bulk_rate_avg") * s / delay).alias("fwd_bulk_rate_avg"),
            (pl.col("packet_length_mean") * byte_ratio).alias("packet_length_mean"),
            (pl.col("average_packet_size") * byte_ratio).alias("average_packet_size"),
            (pl.col("packet_length_std") * byte_ratio).alias("packet_length_std"),
            (pl.col("packet_length_variance") * byte_ratio**2).alias("packet_length_variance"),
        ]
    )
    # Padding can only raise the overall maximum. The overall minimum is left unchanged:
    # the summary does not record which direction it came from.
    return out.with_columns(
        pl.max_horizontal(frame["packet_length_max"], out["fwd_packet_length_max"]).alias(
            "packet_length_max"
        )
    )
