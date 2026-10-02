"""Gold set for the copilot evaluation (report section 8.3).

Labels are assigned per dataset sub-label, not per alert: every "SSH-Patator" flow is the
same attack tool doing the same thing, so it gets the same ATT&CK techniques. `primary`
is the best answer; `acceptable` lists every technique an analyst could defend (ATT&CK
often has a parent and a sub-technique that both fit). Precision@1 counts the copilot's
first technique as correct if it is acceptable.

The mapping follows the CIC-IDS2017 attack descriptions (tool and behaviour per label);
review it before quoting the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sqlalchemy import Engine, select

from sentinel.db.schema import alert_truth, alerts


@dataclass(frozen=True)
class GoldLabel:
    primary: str
    acceptable: frozenset[str]
    cves: frozenset[str] = field(default_factory=frozenset)
    note: str = ""


def _g(primary: str, *others: str, cves: tuple[str, ...] = (), note: str = "") -> GoldLabel:
    return GoldLabel(primary, frozenset({primary, *others}), frozenset(cves), note)


GOLD: dict[str, GoldLabel] = {
    "Portscan": _g("T1046", "T1595", "T1595.001", "T1595.002", note="nmap scans from outside"),
    "Infiltration - Portscan": _g("T1046", note="internal scan from the compromised host"),
    "DDoS": _g("T1498.001", "T1498", "T1499.002", "T1499", note="LOIC HTTP flood, many sources"),
    "DoS Hulk": _g("T1499.002", "T1499", "T1499.003", note="unique-URL HTTP GET flood"),
    "DoS GoldenEye": _g("T1499.002", "T1499", "T1499.003", note="HTTP keep-alive flood"),
    "DoS Slowloris": _g("T1499.002", "T1499", "T1499.003", note="slow headers exhaust connections"),
    "DoS Slowloris - Attempted": _g("T1499.002", "T1499", "T1499.003"),
    "DoS Slowhttptest": _g("T1499.002", "T1499", "T1499.003", note="slow body / read attack"),
    "DoS Slowhttptest - Attempted": _g("T1499.002", "T1499", "T1499.003"),
    "FTP-Patator": _g("T1110.001", "T1110", note="FTP password guessing"),
    "SSH-Patator": _g("T1110.001", "T1110", note="SSH password guessing"),
    "Web Attack - Brute Force": _g("T1110.001", "T1110", note="web login form brute force"),
    "Web Attack - SQL Injection": _g("T1190", note="SQL injection against the web app"),
    "Web Attack - XSS": _g("T1190", "T1059.007", "T1189", note="stored XSS (Selenium-driven)"),
    "Botnet": _g("T1071.001", "T1071", "T1571", note="Ares bot C2 over HTTP to port 8080"),
    "Botnet - Attempted": _g("T1071.001", "T1071", "T1571"),
    "Infiltration": _g(
        "T1204.002", "T1204", "T1105", "T1071.001", "T1071", note="malicious file, Metasploit"
    ),
    "Infiltration - Attempted": _g("T1204.002", "T1204", "T1105"),
    "Heartbleed": _g("T1190", "T1212", cves=("CVE-2014-0160",), note="OpenSSL heartbeat leak"),
    "phishing": _g(
        "T1566.002",
        "T1566",
        "T1566.001",
        "T1598",
        "T1598.002",
        "T1598.003",
        note="credential phishing email",
    ),
    "fraud": _g(
        "T1598.003", "T1598", "T1566", "T1566.002", "T1534", note="advance-fee / 419 fraud email"
    ),
}

# Alerts per sub-label in the gold set (fewer if fewer exist).
PER_LABEL = {
    "Portscan": 8,
    "Infiltration - Portscan": 6,
    "DDoS": 8,
    "DoS Hulk": 7,
    "DoS GoldenEye": 6,
    "DoS Slowloris": 6,
    "DoS Slowhttptest": 6,
    "FTP-Patator": 8,
    "SSH-Patator": 8,
    "Web Attack - Brute Force": 6,
    "Web Attack - SQL Injection": 3,
    "Web Attack - XSS": 4,
    "Botnet": 8,
    "Botnet - Attempted": 2,
    "Infiltration": 5,
    "Heartbleed": 3,
    "phishing": 8,
    "fraud": 6,
}


def build_gold_set(engine: Engine, seed: int) -> list[dict[str, Any]]:
    """Stratified sample of alert IDs with their labels and checkable key facts."""
    rng = np.random.default_rng(seed)
    q = (
        select(
            alerts.c.id,
            alerts.c.source,
            alerts.c.src_ip,
            alerts.c.dst_ip,
            alerts.c.dst_port,
            alerts.c.predicted_family,
            alert_truth.c.label,
        )
        .join(alert_truth, alert_truth.c.alert_id == alerts.c.id)
        .order_by(alerts.c.id)
    )
    with engine.connect() as conn:
        rows = [dict(r._mapping) for r in conn.execute(q)]
    by_label: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    out = []
    for label, n in PER_LABEL.items():
        pool = by_label.get(label, [])
        for i in sorted(rng.choice(len(pool), min(n, len(pool)), replace=False)) if pool else []:
            r = pool[int(i)]
            g = GOLD[label]
            facts = (
                [x for x in (r["src_ip"], r["dst_ip"]) if x] + [str(r["dst_port"])]
                if r["source"] == "network"
                else []
            )
            out.append(
                {
                    "alert_id": int(r["id"]),
                    "label": label,
                    "predicted_family": r["predicted_family"],
                    "primary": g.primary,
                    "acceptable": sorted(g.acceptable),
                    "cves": sorted(g.cves),
                    "key_facts": facts,
                }
            )
    return out
