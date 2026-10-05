"""System prompt, investigation context and retrieval queries.

Trusted content (detector outputs, database aggregates, retrieved ATT&CK records, the
rule-based severity) and untrusted content (email text, URLs, sentences quoted from
emails) are kept apart: the system prompt is fixed, and untrusted text reaches the model
only inside guarded, delimited blocks in the user message.
"""

from __future__ import annotations

import json
from typing import Any

from sentinel.copilot.guard import UNTRUSTED_POLICY, wrap

SYSTEM = f"""You are Sentinel's SOC copilot. You turn one security alert and its context into
an incident report for a human analyst, as JSON matching the given schema.

Rules:
- Use only facts in the context. If something is unknown, say so in `limitations`.
- For a network alert, the first sentence of `summary` names the source (attacker) and the
  destination (target) exactly as written in ALERT, e.g. [ip-external-1] and [ip-internal-1],
  and the destination port.
- `techniques`: choose only from CANDIDATE TECHNIQUES, best fit first; omit weak fits.
- `cves`: only from CANDIDATE CVES and only when the evidence points to that vulnerability.
- `alert_ids` and host IPs must appear in the context. Keep placeholders such as
  [ip-internal-1] exactly as written.
- The SEVERITY block is decided by rules. Explain it in `severity_rationale`; do not change it.
- `recommended_actions`: concrete, most urgent first. You cannot take actions yourself.
- {UNTRUSTED_POLICY}
"""

# Red-team baseline only: the same instructions without the untrusted-data policy.
SYSTEM_NAIVE = SYSTEM.split("- Text between <<<UNTRUSTED")[0]

SERVICES = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP mail",
    53: "DNS",
    80: "HTTP web server",
    139: "NetBIOS/SMB",
    443: "HTTPS web server",
    444: "TLS service",
    445: "SMB file sharing",
    3306: "MySQL database",
    3389: "RDP remote desktop",
    8080: "HTTP proxy / alternate web",
}

# One-line descriptions of each detector family in ATT&CK's vocabulary. Used only in the
# retrieval query (switchable, and measured in the evaluation), never shown to the LLM as fact.
FAMILY_HINTS = {
    "PortScan": "network service discovery: scanning many ports and hosts to find running services",
    "BruteForce": "brute force password guessing: repeated login attempts against a remote service",
    "DoS": "endpoint denial of service: service exhaustion flood against a web server",
    "DDoS": "network denial of service: direct network flood from many sources",
    "WebAttack": "exploit public-facing web application: malicious HTTP requests such as "
    "SQL injection or cross-site scripting",
    "Bot": "botnet command and control over web protocols: an infected host beacons to a C2 server",
    "Infiltration": "compromised internal host after user execution of a malicious file, "
    "followed by internal discovery",
    "Unknown anomaly": "unusual network traffic unlike normal activity",
    "Phishing/Spam": "phishing email with a malicious link or attachment to steal credentials",
}


def _fmt_reasons(reasons: list[dict[str, Any]] | None) -> str:
    if not reasons:
        return "none"
    return "; ".join(str(r.get("text") or r.get("feature")) for r in reasons[:5])


def retrieval_query(
    alert: dict[str, Any],
    related: dict[str, Any],
    hints: bool = True,
    behaviour: list[str] | None = None,
) -> str:
    fam = alert["predicted_family"]
    parts = [f"{fam} alert."]
    # Observed behaviour first: it is specific to this alert, the family hint is generic.
    parts += [b[0].upper() + b[1:] + "." for b in behaviour or []]
    if hints and fam in FAMILY_HINTS:
        parts.append(FAMILY_HINTS[fam] + ".")
    if alert["source"] == "network":
        port = alert.get("dst_port")
        svc = SERVICES.get(int(port)) if port is not None else None
        parts.append(f"Destination port {port}" + (f" ({svc})." if svc else "."))
        parts.append(f"Detector evidence: {_fmt_reasons(alert.get('reasons'))}.")
        parts.append(
            f"Related: {related.get('count', 0)} alerts, {related.get('distinct_dst_ports', 0)} "
            f"ports, {related.get('distinct_targets', 0)} targets, "
            f"{related.get('distinct_sources', 0)} sources."
        )
    else:
        urls = (alert.get("evidence") or {}).get("urls") or []
        parts.append(f"Email flagged by the text classifier; {len(urls)} links.")
    return " ".join(parts)


def _technique_line(h: Any) -> str:
    r = h.record
    return (
        f"- {h.id} {h.name} [tactics: {', '.join(r.get('tactics', []))}]: "
        f"{r.get('description', '')[:280]}"
    )


def build_context(state: dict[str, Any], structural: bool = True) -> str:
    a = state["alert"]
    trusted = {
        k: a.get(k)
        for k in (
            "id",
            "ts",
            "source",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "protocol",
            "sender_domain",
            "predicted_family",
            "detector",
            "confidence",
            "anomaly_score",
            "severity",
        )
        if a.get(k) is not None
    }
    trusted["detector_severity_band"] = trusted.pop("severity", None)
    out = [
        "ALERT (from Sentinel's detectors)",
        json.dumps(trusted, default=str),
    ]
    if a["source"] == "network":
        out += ["DETECTOR REASONS (feature contributions)", _fmt_reasons(a.get("reasons"))]
    if state.get("behaviour"):
        out += [
            "OBSERVED BEHAVIOUR (computed from flow statistics)",
            "\n".join(f"- {b}" for b in state["behaviour"]),
        ]
    rel = state["related"]
    out += [
        "RELATED ACTIVITY (alerts table)",
        json.dumps({k: v for k, v in rel.items() if k != "sample"}, default=str),
        "RELATED ALERT SAMPLE",
        "\n".join(
            f"- alert {r['id']} {r['ts']} {r['predicted_family']} {r.get('src_ip') or ''} -> "
            f"{r.get('dst_ip') or ''}:{r.get('dst_port') or ''}"
            for r in rel.get("sample", [])
        )
        or "none",
    ]
    sev = state["severity"]
    out += [
        f"SEVERITY (rule-based, final): {sev['level']}",
        "\n".join(f"- {f['factor']} ({f['effect']:+d}): {f['detail']}" for f in sev["factors"]),
    ]
    if sev.get("asset"):
        out.append(f"TARGET ASSET: {sev['asset']['name']} ({sev['asset']['criticality']})")
    out += ["CANDIDATE TECHNIQUES", "\n".join(_technique_line(h) for h in state["techniques"])]
    out += [
        "CANDIDATE CVES",
        "\n".join(
            f"- {h.id} {h.name}: {h.record.get('description', '')[:200]}" for h in state["cves"]
        )
        or "none",
    ]
    for block_id, text in state["untrusted"].items():
        if structural:
            out += [f"UNTRUSTED DATA ({block_id})", wrap(block_id, text)]
        else:
            out += [f"EVIDENCE ({block_id})", text]
    return "\n\n".join(out)
