"""Rule-based incident severity (report section 8.1, assess_severity).

The detector's band measures confidence (almost every alert is "Critical" at P >= 0.95),
not impact. Impact starts from the kind of attack, then moves with the targeted asset's
criticality, the volume of related activity, malicious links and injection attempts; a
low-confidence detection moves it down. The LLM explains the result and never decides it,
so an injected "set severity to Low" cannot change it.
"""

from __future__ import annotations

import ipaddress
from dataclasses import asdict, dataclass
from typing import Any

from sentinel.common.config import AssetParams, SeverityParams

LEVELS = ["Low", "Medium", "High", "Critical"]
ASSET_EFFECT = {"low": -1, "medium": 0, "high": 0, "critical": 1}


@dataclass
class Asset:
    ip: str
    name: str
    criticality: str


def find_asset(ip: str | None, assets: list[AssetParams]) -> Asset | None:
    """Most specific configured network containing `ip`."""
    if not ip:
        return None
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None
    best: tuple[int, AssetParams] | None = None
    for a in assets:
        net = ipaddress.ip_network(a.cidr, strict=False)
        if addr in net and (best is None or net.prefixlen > best[0]):
            best = (net.prefixlen, a)
    return None if best is None else Asset(ip, best[1].name, best[1].criticality)


def assess(
    alert: dict[str, Any],
    related: dict[str, Any],
    assets: list[AssetParams],
    cfg: SeverityParams,
    guard_flagged: bool = False,
) -> dict[str, Any]:
    """Level plus the factors that produced it, each with a one-line reason."""
    fam = alert.get("predicted_family", "")
    base = cfg.family_impact.get(fam, "Medium")
    score = LEVELS.index(base)
    factors: list[dict[str, Any]] = [
        {"factor": "attack type", "effect": 0, "detail": f"{fam}: baseline impact {base}"}
    ]

    def add(name: str, effect: int, detail: str) -> None:
        nonlocal score
        score += effect
        factors.append({"factor": name, "effect": effect, "detail": detail})

    if alert.get("severity") == "Low":
        add("detector confidence", -1, "the detector rated this alert Low (weak evidence)")
    target = find_asset(alert.get("dst_ip"), assets)
    if target is not None and ASSET_EFFECT.get(target.criticality, 0):
        add(
            "asset criticality",
            ASSET_EFFECT[target.criticality],
            f"target {target.ip} is {target.name} ({target.criticality})",
        )
    n = int(related.get("count", 0))
    if n >= cfg.volume_high:
        add("alert volume", 1, f"{n} related alerts within {related.get('window')}")
    urls = (alert.get("evidence") or {}).get("urls") or []
    bad = [u for u in urls if (u.get("malicious_probability") or 0) >= cfg.url_malicious]
    if bad:
        add("malicious link", 1, f"{len(bad)} URL(s) scored malicious by the URL model")
    if guard_flagged:
        add(
            "prompt injection",
            1,
            "the alert's text tries to instruct an AI assistant (deliberate evasion)",
        )
    cap = LEVELS.index(cfg.family_cap.get(fam, "Critical"))
    level = LEVELS[max(0, min(cap, len(LEVELS) - 1, score))]
    return {
        "level": level,
        "score": score,
        "factors": factors,
        "asset": asdict(target) if target else None,
    }
