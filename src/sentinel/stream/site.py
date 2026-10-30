"""Site tuning: adapt alerting to one network from a capture of its normal traffic.

The lab models stay as they are. A site profile, learned from a capture known to be normal,
changes two things for that network only:
- Discovery allowlist: (protocol, destination port) pairs that the baseline sends to
  multicast or broadcast addresses at least MIN_FLOWS times (LLMNR, mDNS, SSDP,
  WS-Discovery...). Flows matching one raise no alert. Only multicast/broadcast destinations
  qualify: those packets stay on the LAN and cannot carry a session to the internet. The
  unicast replies and direct queries on those services between two private addresses are
  covered too, for the same reason.
- Anomaly threshold: raised, never lowered, to the score that `fpr` of the baseline's other
  flows exceed. This is the same rule that set the lab threshold, applied to this network.
Supervised verdicts on other flows are unchanged.

A profile describes a real network, so it stays local (data/site/, git-ignored).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sentinel.anomaly.evaluate import threshold_at_fpr

MIN_FLOWS = 5


def fanout() -> pl.Expr:
    """Destination is multicast (224-239.x) or a broadcast address (ends in .255)."""
    first = pl.col("dst_ip").str.split(".").list.first().cast(pl.Int32, strict=False)
    return first.is_between(224, 239).fill_null(False) | pl.col("dst_ip").str.ends_with(".255")


def _private(col: str) -> pl.Expr:
    """RFC 1918 address (10/8, 172.16/12, 192.168/16)."""
    octets = pl.col(col).str.split(".")
    a = octets.list.get(0, null_on_oob=True).cast(pl.Int32, strict=False)
    b = octets.list.get(1, null_on_oob=True).cast(pl.Int32, strict=False)
    return ((a == 10) | ((a == 172) & b.is_between(16, 31)) | ((a == 192) & (b == 168))).fill_null(
        False
    )


def lan() -> pl.Expr:
    """Both ends are private unicast addresses: the flow stays on the local network."""
    return _private("src_ip") & _private("dst_ip") & ~fanout()


@dataclass
class SiteProfile:
    name: str
    allow: list[tuple[int, int]]  # (protocol, destination port) to multicast/broadcast
    anomaly_threshold: float
    lab_threshold: float
    fpr: float
    baseline_flows: int

    def allowed(self, flows: pl.DataFrame) -> np.ndarray:
        """Discovery to multicast/broadcast on an allowlisted service, or the unicast
        exchange that follows it between two local devices (a device answering from the
        service port, or a direct query to it)."""
        if not self.allow:
            return np.zeros(flows.height, dtype=bool)
        dtype = flows.schema["dst_port"]
        services = pl.DataFrame(
            {"p": [p for p, _ in self.allow], "port": [d for _, d in self.allow]},
            schema={"p": flows.schema["protocol"], "port": dtype},
        ).select(pl.struct("p", "port").alias("s"))["s"]

        def on(port: str) -> pl.Expr:
            return pl.struct(pl.col("protocol").alias("p"), pl.col(port).alias("port")).is_in(
                services
            )

        rule = (fanout() & on("dst_port")) | (lan() & (on("src_port") | on("dst_port")))
        return flows.select(rule).to_series().to_numpy()

    def apply(self, flows: pl.DataFrame, results: Sequence[dict[str, Any]]) -> dict[str, int]:
        """Clear the alerts this profile suppresses (in place); count them by reason."""
        allowed = self.allowed(flows)
        counts = {"allowlist": 0, "threshold": 0}
        for i, r in enumerate(results):
            if not r["is_attack"]:
                continue
            if allowed[i]:
                reason = "allowlist"
            elif r["detector"] == "anomaly" and r["anomaly_score"] < self.anomaly_threshold:
                reason = "threshold"
            else:
                continue
            counts[reason] += 1
            r.update(is_attack=False, family="Benign", detector=None, site_suppressed=reason)
        return counts

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SiteProfile:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        cfg["allow"] = [tuple(p) for p in cfg["allow"]]
        return cls(**cfg)


def compare(
    profile: SiteProfile, flows: pl.DataFrame, results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Alerts before and after the profile. With a `label` column (the lab data), also the
    attack recall it costs, overall and per family, and the benign false-alarm rate."""
    before = np.array([r["is_attack"] for r in results], dtype=bool)
    after_results = [dict(r) for r in results]
    suppressed = profile.apply(flows, after_results)
    after = np.array([r["is_attack"] for r in after_results], dtype=bool)
    n = max(len(results), 1)
    out: dict[str, Any] = {
        "flows": len(results),
        "alerts_before": int(before.sum()),
        "alerts_after": int(after.sum()),
        "alert_share_before": float(before.sum() / n),
        "alert_share_after": float(after.sum() / n),
        "suppressed": suppressed,
    }
    if "family" in flows.columns:
        fam = flows["family"].to_numpy()
        attack, benign = fam != "Benign", fam == "Benign"
        out |= {
            "recall_before": float(before[attack].mean()) if attack.any() else None,
            "recall_after": float(after[attack].mean()) if attack.any() else None,
            "benign_fpr_before": float(before[benign].mean()) if benign.any() else None,
            "benign_fpr_after": float(after[benign].mean()) if benign.any() else None,
            "recall_by_family": {
                str(f): {
                    "flows": int((fam == f).sum()),
                    "before": float(before[fam == f].mean()),
                    "after": float(after[fam == f].mean()),
                }
                for f in sorted(set(fam[attack]))
            },
        }
    return out


def learn(
    name: str,
    flows: pl.DataFrame,
    results: Sequence[dict[str, Any]],
    lab_threshold: float,
    fpr: float,
) -> SiteProfile:
    """Fit a profile to a capture of normal traffic and the detector's verdicts on it."""
    allow = (
        flows.filter(fanout())
        .group_by("protocol", "dst_port")
        .len()
        .filter(pl.col("len") >= MIN_FLOWS)
        .sort("protocol", "dst_port")
    )
    profile = SiteProfile(
        name=name,
        allow=[(int(p), int(d)) for p, d in allow.select("protocol", "dst_port").rows()],
        anomaly_threshold=lab_threshold,
        lab_threshold=lab_threshold,
        fpr=fpr,
        baseline_flows=flows.height,
    )
    rest = ~profile.allowed(flows)
    scores = np.array([r["anomaly_score"] for r in results], dtype=float)[rest]
    if scores.size:
        profile.anomaly_threshold = max(lab_threshold, threshold_at_fpr(scores, fpr))
    return profile
