"""Site profile: discovery allowlist and per-network anomaly threshold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from sentinel.stream.site import MIN_FLOWS, SiteProfile, compare, learn


def _flows(rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dst_ip": [d for d, _, _ in rows],
            "protocol": [p for _, p, _ in rows],
            "dst_port": [port for _, _, port in rows],
        }
    )


def _result(score: float, threshold: float = 1.0, family: str = "Benign") -> dict[str, Any]:
    anomalous = score >= threshold and family == "Benign"
    return {
        "is_attack": anomalous or family != "Benign",
        "family": "Unknown anomaly" if anomalous else family,
        "detector": "anomaly" if anomalous else ("supervised" if family != "Benign" else None),
        "anomaly_score": score,
    }


def test_learn_allowlists_repeated_lan_discovery_only() -> None:
    rows = [("239.255.255.250", 17, 1900)] * MIN_FLOWS  # SSDP multicast
    rows += [("192.168.1.255", 17, 137)] * MIN_FLOWS  # NetBIOS broadcast
    rows += [("224.0.0.252", 17, 5355)] * (MIN_FLOWS - 1)  # too rare to learn
    rows += [("8.8.8.8", 17, 53)] * 20  # unicast never qualifies
    flows = _flows(rows)
    p = learn("t", flows, [_result(0.1)] * flows.height, lab_threshold=1.0, fpr=0.01)
    assert p.allow == [(17, 137), (17, 1900)]


def test_threshold_only_rises() -> None:
    flows = _flows([("10.0.0.2", 6, 443)] * 100)
    low = learn("t", flows, [_result(0.01)] * 100, lab_threshold=1.0, fpr=0.01)
    assert low.anomaly_threshold == 1.0
    noisy = [_result(float(i)) for i in range(100)]  # scores 0..99
    high = learn("t", flows, noisy, lab_threshold=1.0, fpr=0.1)
    assert 89 < high.anomaly_threshold <= 90


def test_apply_suppresses_allowlisted_and_weak_anomalies_only() -> None:
    p = SiteProfile("t", [(17, 1900)], 5.0, 1.0, 0.01, 10)
    flows = _flows(
        [
            ("239.255.255.250", 17, 1900),
            ("10.0.0.2", 6, 443),
            ("10.0.0.3", 6, 443),
            ("10.0.0.4", 6, 22),
        ]
    )
    results = [
        _result(0.1, family="PortScan"),  # allowlisted service: suppressed
        _result(2.0),  # anomaly under the site threshold: suppressed
        _result(9.0),  # strong anomaly: kept
        _result(0.1, family="Brute Force"),  # supervised, not allowlisted: kept
    ]
    assert p.apply(flows, results) == {"allowlist": 1, "threshold": 1}
    assert [r["is_attack"] for r in results] == [False, False, True, True]
    assert results[0]["site_suppressed"] == "allowlist"


def test_compare_reports_recall_cost(tmp_path: Path) -> None:
    p = SiteProfile("t", [], 5.0, 1.0, 0.01, 10)
    flows = _flows([("10.0.0.1", 6, 80)] * 3).with_columns(
        pl.Series("family", ["DoS", "DoS", "Benign"])
    )
    results = [_result(9.0), _result(2.0), _result(3.0)]  # all anomaly alerts
    c = compare(p, flows, results)
    assert (c["alerts_before"], c["alerts_after"]) == (3, 1)
    assert (c["recall_before"], c["recall_after"]) == (1.0, 0.5)
    assert (c["benign_fpr_before"], c["benign_fpr_after"]) == (1.0, 0.0)
    assert all(r["is_attack"] for r in results)  # the caller's verdicts are untouched
    path = tmp_path / "site.json"
    p.save(path)
    assert SiteProfile.load(path) == p
