from __future__ import annotations

from typing import Any

from sentinel.common.config import AssetParams, SeverityParams
from sentinel.copilot.severity import assess, find_asset

ASSETS = [
    AssetParams(cidr="192.168.10.0/24", name="workstation", criticality="medium"),
    AssetParams(cidr="192.168.10.50", name="web server", criticality="critical"),
    AssetParams(cidr="205.174.165.0/24", name="attacker network", criticality="low"),
]
CFG = SeverityParams(
    family_impact={"PortScan": "Medium", "DoS": "High", "BruteForce": "Medium"},
    family_cap={"PortScan": "High"},
    volume_high=100,
)


def _alert(**kw: Any) -> dict[str, Any]:
    return {"source": "network", "severity": "Critical", "detector": "supervised", **kw}


def test_most_specific_asset_wins() -> None:
    web = find_asset("192.168.10.50", ASSETS)
    assert web is not None and web.name == "web server"
    ws = find_asset("192.168.10.7", ASSETS)
    assert ws is not None and ws.name == "workstation"
    assert find_asset("8.8.8.8", ASSETS) is None and find_asset(None, ASSETS) is None


def test_impact_moves_with_asset_volume_and_confidence() -> None:
    dos = assess(
        _alert(predicted_family="DoS", dst_ip="192.168.10.50"), {"count": 500}, ASSETS, CFG
    )
    assert dos["level"] == "Critical" and dos["score"] == 4  # High +1 asset +1 volume (clamped)
    weak = assess(
        _alert(predicted_family="DoS", dst_ip="192.168.10.7", severity="Low"),
        {"count": 0},
        ASSETS,
        CFG,
    )
    assert weak["level"] == "Medium"
    assert {f["factor"] for f in weak["factors"]} == {"attack type", "detector confidence"}


def test_caps_and_injection() -> None:
    scan = assess(
        _alert(predicted_family="PortScan", dst_ip="192.168.10.50"), {"count": 5000}, ASSETS, CFG
    )
    assert scan["level"] == "High"  # Medium +1 +1 would be Critical; recon is capped
    email = {
        "source": "email",
        "severity": "Medium",
        "predicted_family": "Phishing/Spam",
        "evidence": {"urls": [{"url": "http://x", "malicious_probability": 0.99}]},
    }
    out = assess(email, {"count": 0}, ASSETS, CFG, guard_flagged=True)
    assert out["level"] == "Critical"  # Medium +1 malicious link +1 injection attempt
