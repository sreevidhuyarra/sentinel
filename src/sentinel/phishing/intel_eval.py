"""Measure the blocklist checks on legitimate mail and decide how far they may be trusted.

The same rules as the sender checks (sentinel.phishing.sender_eval): a check may raise an
alert by itself if it fires on at most 0.1% of legitimate validation emails, may mark an
email suspicious at up to 2%, and is ignored above that.

- Blocklists are measured against a snapshot of the feeds (their download dates go in the
  report). The corpora are years older than the feeds, so a legitimate domain that has since
  expired and been re-registered by a phisher counts as a false alarm: the measured rate is
  an upper bound.
- Domain age is not measured: RDAP gives today's registration, and the corpora's domains
  are from 2000-2020, so "new at the time of the email" cannot be rebuilt for them. It only
  marks an email suspicious.
- Phishing Pot hits are hindsight (today's feed against older emails) and are reported
  as such, not as a detection rate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.common.config import Params
from sentinel.common.logging import get_logger
from sentinel.phishing.intel import Intel
from sentinel.phishing.sender import parse_sender
from sentinel.phishing.sender_eval import (
    ALERT_MAX,
    BUNDLE_DIR,
    POLICY_PATH,
    POT_CACHE,
    REPORTS,
    REVIEW_MAX,
    URL_BUNDLE_DIR,
    _auth_text,
    _pct,
    _rates,
    corpus_frame,
    corpus_senders,
    findings_of,
    verdicts,
)

log = get_logger(__name__)

INTEL_POLICY_PATH = Path("models/phishing/intel_policy.json")
MEASURED = ("listed_url", "listed_domain")
UNMEASURED = {"new_domain": "review"}
INTEL_CACHE = Path("data/cache/intel")


def intel_findings(intel: Intel, df: pd.DataFrame) -> list[set[str]]:
    out = []
    for fr, rt, a, urls in zip(df["from_"], df["reply_to"], df["auth"], df["urls"], strict=True):
        who = parse_sender(fr, rt, _auth_text(a)) if fr else None
        rep = intel.check(who, list(urls)[:20])
        out.append({f.check for f in rep.findings})
    return out


def _rate(found: list[set[str]], mask: np.ndarray, check: str) -> float:
    n = int(mask.sum())
    c: Counter[str] = Counter(x for f, m in zip(found, mask, strict=True) if m for x in f)
    return c[check] / n if n else float("nan")


def run(params: Params) -> dict[str, Any]:
    from sentinel.phishing.export import PhishingOnnxModel
    from sentinel.phishing.sender import load_policy
    from sentinel.phishing.urls import UrlModel

    intel = Intel(params.resolve(INTEL_CACHE), rdap=False)
    intel.refresh_feeds()
    if not intel.feeds_loaded:
        raise RuntimeError("no blocklist could be downloaded or found in data/cache/intel")

    p = params.phishing
    ph = PhishingOnnxModel.load(params.resolve(BUNDLE_DIR))
    um = UrlModel.load(params.resolve(URL_BUNDLE_DIR))
    thr = (ph.threshold, p.suspicious_threshold, um.threshold, um.email_threshold or 1.1)
    sender_policy: dict[str, str] = dict(load_policy(params.resolve(POLICY_PATH)))

    senders = corpus_senders(params)
    val = corpus_frame(params, "val", senders)
    test = corpus_frame(params, "test", senders)
    pot_path = params.resolve(POT_CACHE)
    pot = pd.read_parquet(pot_path) if pot_path.exists() else None

    legit_val = (val["kind"] == "legitimate").to_numpy()
    val_intel, test_intel = intel_findings(intel, val), intel_findings(intel, test)
    val_sender, test_sender = findings_of(val), findings_of(test)

    # Role per check, from legitimate validation mail only.
    legit_rate = {c: _rate(val_intel, legit_val, c) for c in MEASURED}
    policy: dict[str, str] = {
        c: "alert" if r <= ALERT_MAX else "review" if r <= REVIEW_MAX else "ignore"
        for c, r in legit_rate.items()
    } | UNMEASURED

    def groups(d: pd.DataFrame) -> dict[str, np.ndarray]:
        kind = d["kind"].to_numpy()
        return {k: kind == k for k in ("legitimate", "phishing", "fraud", "spam")}

    results: dict[str, Any] = {}
    for name, d, sf, inf in (
        ("test", test, test_sender, test_intel),
        ("val", val, val_sender, val_intel),
    ):
        both = [a | b for a, b in zip(sf, inf, strict=True)]
        out: dict[str, Any] = {}
        for config, found, pol in (
            ("text + link + sender", sf, sender_policy),
            ("+ blocklists", both, sender_policy | policy),
        ):
            mal, sus = verdicts(d, found, pol, *thr, use_sender=True)
            out[config] = {g: _rates(mal, sus, m) for g, m in groups(d).items()}
        results[name] = out

    check_rates: dict[str, dict[str, float]] = {}
    for label, found, mask in (
        ("legitimate val", val_intel, legit_val),
        ("legitimate test", test_intel, (test["kind"] == "legitimate").to_numpy()),
        ("phishing (Nazario) test", test_intel, (test["kind"] == "phishing").to_numpy()),
    ):
        check_rates[label] = {c: _rate(found, mask, c) for c in MEASURED}
    if pot is not None:
        pot_found = intel_findings(intel, pot)
        for split in ("dev", "test"):
            mask = (pot["split"] == split).to_numpy()
            check_rates[f"phishing pot {split} (hindsight)"] = {
                c: _rate(pot_found, mask, c) for c in MEASURED
            }

    summary: dict[str, Any] = {
        "policy": policy,
        "limits": {"alert_max": ALERT_MAX, "review_max": REVIEW_MAX},
        "feeds": intel.status()["feeds"],
        "check_rates": check_rates,
        "results": results,
        "counts": {
            "legitimate val": int(legit_val.sum()),
            "legitimate test": int((test["kind"] == "legitimate").sum()),
        },
    }
    path = params.resolve(INTEL_POLICY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"policy": policy, "rates_on_legitimate_val": legit_rate}, indent=2),
        encoding="utf-8",
    )
    reports = params.resolve(REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "intel_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (reports / "intel_results.md").write_text(report(summary), encoding="utf-8")
    return summary


def report(s: dict[str, Any]) -> str:
    lines = [
        "# Blocklists and domain age\n",
        "Each email's sender, Reply-To and link domains are looked up in phishing blocklists "
        "(downloaded; nothing about the email is sent out). Roles follow the sender-check "
        "rules: **alert** at most 0.1% of legitimate validation emails, **review** at most "
        "2%. Domain age (RDAP) is review-only: it cannot be measured on these corpora.\n",
        "Feed snapshot used:\n",
    ]
    lines += [
        f"- {n}: {f['hosts']:,} hosts, downloaded {f['updated']}" for n, f in s["feeds"].items()
    ]
    lines += [
        "",
        "## How often each check fires\n",
        "| check | role | " + " | ".join(s["check_rates"]) + " |",
        "|---|---|" + "---|" * len(s["check_rates"]),
    ]
    for c in MEASURED:
        lines.append(
            f"| {c} | {s['policy'][c]} | "
            + " | ".join(_pct(r[c]) for r in s["check_rates"].values())
            + " |"
        )
    lines += [
        f"| new_domain | {s['policy']['new_domain']} | not measured |"
        + " |" * (len(s["check_rates"]) - 1),
        "",
        "Phishing Pot columns are hindsight: today's feeds against emails sent months or "
        "years earlier. They show overlap, not what the check would have caught at the time.\n",
        "## Test results\n",
        "Share flagged **malicious** and **malicious or suspicious**, before and after adding "
        "the blocklist checks to the text, link and sender checks.\n",
        "| emails | n | malicious | + blocklists | malicious or suspicious | + blocklists |",
        "|---|---|---|---|---|---|",
    ]
    t = s["results"]["test"]
    for g in t["text + link + sender"]:
        a, b = t["text + link + sender"][g], t["+ blocklists"][g]
        lines.append(
            f"| {g} | {a['n']} | {_pct(a['malicious'])} | {_pct(b['malicious'])} | "
            f"{_pct(a['malicious_or_suspicious'])} | {_pct(b['malicious_or_suspicious'])} |"
        )
    lines.append("")
    return "\n".join(lines)
