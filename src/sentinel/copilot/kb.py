"""Knowledge base for retrieval: MITRE ATT&CK Enterprise (STIX 2.1) and CISA KEV.

The design names an NVD snapshot for CVEs; the full feed is 250k+ records, most never
exploited. CISA's Known Exploited Vulnerabilities catalog (~1.7k CVEs, one public JSON)
is the subset an incident responder acts on, and it links each entry to NVD.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from sentinel.common.logging import get_logger

log = get_logger(__name__)

ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack.json"
)
KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

_CITATION = re.compile(r"\s*\(Citation:[^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_TAGS = re.compile(r"</?code>")
_WS = re.compile(r"\s+")


def clean(text: str | None) -> str:
    if not text:
        return ""
    text = _MD_LINK.sub(r"\1", _CITATION.sub("", text))
    return _WS.sub(" ", _TAGS.sub("", text)).strip()


def download(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for url, name in ((ATTACK_URL, "enterprise-attack.json"), (KEV_URL, "kev.json")):
        path = raw_dir / name
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with path.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        out.append(path)
    return out


def _external_id(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return str(ref["external_id"])
    return None


def _active(obj: dict[str, Any]) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def parse_attack(path: Path) -> pl.DataFrame:
    """One row per active technique / sub-technique, with mitigations and detections."""
    objs = json.loads(path.read_text(encoding="utf-8"))["objects"]
    by_id = {o["id"]: o for o in objs}
    mitigations: dict[str, list[str]] = {}
    detections: dict[str, list[str]] = {}
    for r in objs:
        if r["type"] != "relationship" or not _active(r):
            continue
        src, tgt = by_id.get(r["source_ref"]), by_id.get(r["target_ref"])
        if src is None or tgt is None or not _active(src):
            continue
        if r["relationship_type"] == "mitigates":
            mitigations.setdefault(tgt["id"], []).append(src["name"])
        elif r["relationship_type"] == "detects":
            analytics = [
                clean(by_id[a].get("description"))
                for a in src.get("x_mitre_analytic_refs", [])
                if a in by_id
            ]
            detections.setdefault(tgt["id"], []).append(
                src["name"] + (": " + " ".join(analytics[:2]) if analytics else "")
            )
    rows = []
    for o in objs:
        if o["type"] != "attack-pattern" or not _active(o):
            continue
        tid = _external_id(o)
        if tid is None:
            continue
        rows.append(
            {
                "id": tid,
                "name": o["name"],
                "parent": tid.split(".")[0] if "." in tid else None,
                "tactics": [p["phase_name"] for p in o.get("kill_chain_phases", [])],
                "platforms": o.get("x_mitre_platforms", []),
                "description": clean(o.get("description")),
                "detection": " ".join(detections.get(o["id"], []))[:2000],
                "mitigations": sorted(set(mitigations.get(o["id"], []))),
                "url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}",
            }
        )
    df = pl.DataFrame(rows).sort("id")
    names = dict(zip(df["id"], df["name"], strict=True))
    # "Brute Force: Password Guessing", as ATT&CK itself displays sub-techniques.
    return df.with_columns(
        pl.struct("id", "name", "parent")
        .map_elements(
            lambda s: f"{names.get(s['parent'], '')}: {s['name']}" if s["parent"] else s["name"],
            return_dtype=pl.String,
        )
        .alias("full_name")
    )


def parse_kev(path: Path) -> pl.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "id": v["cveID"],
            "name": v["vulnerabilityName"],
            "vendor": v["vendorProject"],
            "product": v["product"],
            "description": clean(v["shortDescription"]),
            "date_added": v["dateAdded"],
            "ransomware": v.get("knownRansomwareCampaignUse") == "Known",
            "cwes": v.get("cwes", []),
            "url": f"https://nvd.nist.gov/vuln/detail/{v['cveID']}",
        }
        for v in data["vulnerabilities"]
    ]
    return pl.DataFrame(rows).sort("id")


def build(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tech = parse_attack(raw_dir / "enterprise-attack.json")
    cves = parse_kev(raw_dir / "kev.json")
    tech.write_parquet(out_dir / "techniques.parquet")
    cves.write_parquet(out_dir / "cves.parquet")
    log.info("knowledge base: %d techniques, %d CVEs", len(tech), len(cves))
    return {"techniques": len(tech), "cves": len(cves)}
