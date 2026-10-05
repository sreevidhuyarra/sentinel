from __future__ import annotations

from typing import Any

from sentinel.common.config import BehaviourParams
from sentinel.copilot.behaviour import describe

CFG = BehaviourParams()


def _act(**kw: Any) -> dict[str, Any]:
    pair = {
        "alerts": 1,
        "ports": 1,
        "interval_median_s": None,
        "interval_cv": None,
        "fwd_bytes": None,
        "bwd_bytes": None,
        "duration_us": None,
        "fwd_packets": None,
        "packets_per_s": None,
    }
    pair.update(kw.pop("pair", {}))
    return {
        "window_minutes": 30,
        "from_source": {"alerts": 1, "hosts": 1, "ports": 1, **kw.pop("from_source", {})},
        "to_target": {"alerts": 1, "sources": 1, **kw.pop("to_target", {})},
        "pair": pair,
        "flow": kw.pop("flow", {}),
    }


def _alert(port: int, src: str = "172.16.0.1", dst: str = "192.168.10.50") -> dict[str, Any]:
    return {"source": "network", "src_ip": src, "dst_ip": dst, "dst_port": port}


def test_scan_and_sweep() -> None:
    out = describe(_alert(80), _act(from_source={"ports": 900, "hosts": 1}), CFG)
    assert any("port scan" in s for s in out)
    out = describe(_alert(445), _act(from_source={"ports": 1, "hosts": 40}), CFG)
    assert any("network sweep" in s for s in out)


def test_password_guessing_needs_alike_repeated_connections() -> None:
    alike = {"alerts": 300, "ports": 1, "fwd_bytes": {"mean": 400.0, "cv": 0.1}}
    assert any("password guessing" in s for s in describe(_alert(22), _act(pair=alike), CFG))
    varied = {**alike, "fwd_bytes": {"mean": 400.0, "cv": 2.0}}
    assert not any("password" in s for s in describe(_alert(22), _act(pair=varied), CFG))
    web = describe(_alert(80), _act(pair=alike), CFG)
    assert any("login brute force" in s for s in web)
    # Under a flood, identical requests are the flood, not a login brute force.
    flooded = describe(_alert(80), _act(pair=alike, to_target={"alerts": 9000}), CFG)
    assert not any("login" in s for s in flooded) and any("flood" in s for s in flooded)
    # Login traffic is never called a flood, however many connections it makes.
    ssh = describe(_alert(22), _act(pair=alike, to_target={"alerts": 9000}), CFG)
    assert not any("flood" in s for s in ssh)


def test_flood_slow_beacon_leak() -> None:
    flood = describe(_alert(80), _act(to_target={"alerts": 5000, "sources": 40}), CFG)
    assert any("distributed" in s and "flood" in s for s in flood)
    # A port scan hitting one host many times is a scan, not a flood.
    scan = _act(from_source={"ports": 900}, to_target={"alerts": 5000}, pair={"ports": 900})
    assert not any("flood" in s for s in describe(_alert(80), scan, CFG))
    slow = {"flow_duration": 9e7, "flow_packets_s": 0.5, "total_length_of_bwd_packet": 0.0}
    assert any("slow-rate" in s for s in describe(_alert(80), _act(flow=slow), CFG))
    answered = {**slow, "total_length_of_bwd_packet": 180_000.0}  # a long but normal session
    assert not any("slow-rate" in s for s in describe(_alert(80), _act(flow=answered), CFG))
    beacon = {"alerts": 30, "interval_median_s": 60.0, "interval_cv": 0.1}
    out = describe(_alert(8080, "192.168.10.9", "205.174.165.73"), _act(pair=beacon), CFG)
    assert any("beaconing" in s and "external" in s for s in out)
    assert any("non-standard port 8080" in s for s in out)
    inbound = describe(_alert(80), _act(pair=beacon), CFG)  # attacker -> internal web server
    assert not any("beaconing" in s for s in inbound)
    leak = {"total_length_of_fwd_packet": 300.0, "total_length_of_bwd_packet": 64_000.0}
    assert any("memory disclosure" in s for s in describe(_alert(444), _act(flow=leak), CFG))


def test_quiet_traffic_and_emails_say_nothing() -> None:
    assert describe(_alert(443), _act(), CFG) == []
    assert describe({"source": "email"}, {}, CFG) == []
