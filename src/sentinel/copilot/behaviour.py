"""Plain-language behaviour descriptions of a network alert, computed from flow statistics.

Flow features carry no ATT&CK vocabulary, so retrieval used to depend on the detector's
family name alone. These sentences describe what the traffic *did* (fan-out, fan-in,
regularity, sizes, durations) in the words ATT&CK uses, so the copilot can map alerts the
family name is too coarse for ("WebAttack", "Unknown anomaly") and explain its choice.

The rules are generic and use only the alert and its surrounding activity, never a dataset
label. Thresholds are parameters, tuned on the development alert set only.
"""

from __future__ import annotations

import ipaddress
from typing import Any

from sentinel.common.config import BehaviourParams

AUTH_PORTS = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    110: "POP3",
    143: "IMAP",
    445: "SMB",
    3389: "RDP",
    5900: "VNC",
}
WEB_PORTS = {80: "HTTP", 443: "HTTPS", 8080: "HTTP", 8443: "HTTPS"}
TLS_PORTS = {443, 444, 465, 636, 993, 995, 8443}
# Ports an internal host normally uses towards the internet; anything else is worth a note.
STANDARD_OUTBOUND = {22, 25, 53, 80, 123, 443, 465, 587, 993, 995}


def _private(ip: str | None) -> bool | None:
    try:
        return ipaddress.ip_address(ip).is_private if ip else None
    except ValueError:
        return None


def describe(alert: dict[str, Any], act: dict[str, Any], cfg: BehaviourParams) -> list[str]:
    """Behaviour sentences, most specific first; empty when nothing stands out."""
    if not act:
        return []
    out: list[str] = []  # what this connection (and its source -> target pair) did
    context: list[str] = []  # what the same source did elsewhere in the window
    port = alert.get("dst_port")
    src_internal, dst_internal = _private(alert.get("src_ip")), _private(alert.get("dst_ip"))
    fo, fi, pair, flow = act["from_source"], act["to_target"], act["pair"], act["flow"]
    window = act["window_minutes"]

    # A scan is this connection's behaviour only if the pair itself spans many ports; a
    # single connection from a host that also scans is something else (e.g. its C2 link),
    # and the scan is context.
    probe = pair["ports"] >= max(2, cfg.scan_ports // 2)
    if fo["ports"] >= cfg.scan_ports:
        scan = (
            f"one source probed {fo['ports']} different ports on {fo['hosts']} host(s) within "
            f"{window} minutes, the pattern of a port scan for network service discovery"
        )
        if probe:
            out.append(scan)
        else:
            who = "This internal source" if src_internal else "This source"
            context.append(
                f"{who} also probed {fo['ports']} different ports on {fo['hosts']} host(s) "
                f"within {window} minutes (port scanning elsewhere; this connection is not a probe)"
            )
    elif fo["hosts"] >= cfg.sweep_hosts:
        sweep = (
            f"one source contacted {fo['hosts']} different hosts within {window} minutes "
            "(a network sweep for remote system discovery)"
        )
        (out if probe else context).append(sweep)

    fwd = pair.get("fwd_bytes")
    alike = fwd is not None and fwd["cv"] <= cfg.alike_cv
    scanning = fo["ports"] >= cfg.scan_ports
    flooded = fi["alerts"] >= cfg.flood_alerts and pair["ports"] <= 2 and not scanning
    if port in AUTH_PORTS and pair["alerts"] >= cfg.repeat_min and pair["ports"] <= 2 and alike:
        out.append(
            f"{pair['alerts']} repeated short connections from one source to the "
            f"{AUTH_PORTS[port]} login service with near-identical sizes: automated "
            "password guessing (brute force)"
        )
    elif port in WEB_PORTS and pair["alerts"] >= cfg.web_repeat_min and alike and not flooded:
        out.append(
            f"{pair['alerts']} near-identical {WEB_PORTS[port]} requests from one source to the "
            "same web server at a steady pace: automated form submission such as a login "
            "brute force"
        )

    # Login traffic is brute force, not a flood, however many connections it makes.
    if flooded and port not in AUTH_PORTS:
        who = (
            f"from {fi['sources']} different sources (distributed)"
            if fi["sources"] >= cfg.flood_sources
            else "from a single source"
        )
        if port in WEB_PORTS:
            # Requests to one web service exhaust that service; "flood" alone reads as a
            # volumetric network flood (a different ATT&CK technique).
            out.append(
                f"{fi['alerts']} {WEB_PORTS[port]} connections to the web service on one "
                f"server within {window} minutes {who}: application requests sent to exhaust "
                "the web service (service exhaustion, endpoint denial of service)"
            )
        else:
            out.append(
                f"{fi['alerts']} connections to one target within {window} minutes {who}: "
                "a volumetric flood saturating the target (network denial of service)"
            )

    # This flow's own statistics (pair means are skewed by a few extreme flows).
    fwd_b = flow.get("total_length_of_fwd_packet") or 0
    bwd_b = flow.get("total_length_of_bwd_packet") or 0
    dur_s = (flow.get("flow_duration") or 0) / 1e6
    pps = flow.get("flow_packets_s")
    if (
        dur_s >= cfg.slow_seconds
        and pps is not None
        and pps <= cfg.slow_pps
        and bwd_b <= cfg.slow_bwd_bytes
        # A slow-rate attack targets our server; a quiet outbound session is something else.
        and dst_internal is not False
    ):
        out.append(
            f"a long-lived connection ({dur_s:.0f} s) that sends very few packets and gets "
            "almost no response: a slow-rate attack holding server connections open "
            "(service exhaustion)"
        )

    gap, cv = pair.get("interval_median_s"), pair.get("interval_cv")
    if (
        pair["alerts"] >= cfg.beacon_min
        and gap is not None
        and gap >= cfg.beacon_min_interval_s
        and cv is not None
        and cv <= cfg.beacon_cv
        and src_internal
        and dst_internal is False  # beaconing calls out; regular inbound traffic is not C2
    ):
        out.append(
            f"regular connections about every {gap:.0f} s from an internal host to the same "
            f"external host on port {port}: periodic beaconing typical of command and control "
            "over an application-layer protocol"
        )

    if (
        src_internal
        and dst_internal is False
        and port is not None
        and port not in STANDARD_OUTBOUND
    ):
        out.append(
            f"an internal host connecting out to an external host on non-standard port {port}"
        )

    if port in TLS_PORTS and bwd_b >= cfg.leak_bytes and bwd_b >= cfg.leak_ratio * max(fwd_b, 1):
        out.append(
            f"a TLS service returned {bwd_b:,.0f} bytes for {fwd_b:,.0f} bytes requested: the "
            "server sent far more data than asked for (possible memory disclosure by exploiting "
            "a public-facing service)"
        )

    if port in WEB_PORTS and pair["alerts"] <= cfg.few_requests and fwd_b >= cfg.large_request:
        out.append(
            f"a few {WEB_PORTS[port]} requests carrying large request bodies ({fwd_b:,.0f} "
            "bytes): possible injection or exploit attempt against a public-facing web application"
        )
    return out + context
