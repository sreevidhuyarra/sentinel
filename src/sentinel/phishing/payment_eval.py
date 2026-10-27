"""Measure the payment checks and decide how far they may be trusted.

- The wording checks are review-only by design (see sentinel.phishing.payment): the corpora
  hold few genuine invoices, so their legitimate false-alarm rate would not carry over to a
  finance inbox. Their rates are still reported.
- `payment_from_risky_sender` follows the sender-check rule: alert if it fires on at most
  0.1% of legitimate validation emails, review up to 2%. Risky = any From-line finding except
  no_address (domain age and blocklists need lookups and are left out here).
- No corpus here is invoice fraud (the "fraud" kind is advance-fee scams), so catch rates are
  reported for what exists, not as an invoice-fraud detection rate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from sentinel.common.config import Params
from sentinel.common.logging import get_logger
from sentinel.phishing.payment import ESCALATED, NOT_RISKY, PAYMENT_CHECKS, check_payment
from sentinel.phishing.sender_eval import (
    ALERT_MAX,
    BUNDLE_DIR,
    POLICY_PATH,
    POT_CACHE,
    REPORTS,
    REVIEW_MAX,
    URL_BUNDLE_DIR,
    _pct,
    _rates,
    corpus_frame,
    corpus_senders,
    findings_of,
    verdicts,
)

log = get_logger(__name__)

PAYMENT_POLICY_PATH = Path("models/phishing/payment_policy.json")
CHECKS = (*PAYMENT_CHECKS, ESCALATED)


def payment_findings(texts: list[str], sender_found: list[set[str]]) -> list[set[str]]:
    out = []
    for text, sf in zip(texts, sender_found, strict=True):
        found = {f.check for f in check_payment(text)}
        if found and sf - NOT_RISKY:
            found.add(ESCALATED)
        out.append(found)
    return out


def _rate(found: list[set[str]], mask: np.ndarray, check: str) -> float:
    n = int(mask.sum())
    hits = sum(1 for f, m in zip(found, mask, strict=True) if m and check in f)
    return hits / n if n else float("nan")


def _with_text(params: Params, df: pd.DataFrame, split: str) -> pd.DataFrame:
    texts = (
        pl.read_parquet(params.resolve(params.phishing.processed_dir) / "emails.parquet")
        .filter(pl.col("split") == split)
        .select("id", "text")
        .to_pandas()
    )
    return df.merge(texts, on="id")


def run(params: Params) -> dict[str, Any]:
    from sentinel.phishing.export import PhishingOnnxModel
    from sentinel.phishing.sender import load_policy
    from sentinel.phishing.urls import UrlModel

    p = params.phishing
    ph = PhishingOnnxModel.load(params.resolve(BUNDLE_DIR))
    um = UrlModel.load(params.resolve(URL_BUNDLE_DIR))
    thr = (ph.threshold, p.suspicious_threshold, um.threshold, um.email_threshold or 1.1)
    sender_policy: dict[str, str] = dict(load_policy(params.resolve(POLICY_PATH)))

    senders = corpus_senders(params)
    frames = {s: _with_text(params, corpus_frame(params, s, senders), s) for s in ("val", "test")}
    pot_path = params.resolve(POT_CACHE)
    if pot_path.exists():
        frames["pot"] = pd.read_parquet(pot_path)
    sender_found = {k: findings_of(d) for k, d in frames.items()}
    pay_found = {
        k: payment_findings(d["text"].tolist(), sender_found[k]) for k, d in frames.items()
    }

    legit_val = (frames["val"]["kind"] == "legitimate").to_numpy()
    esc_rate = _rate(pay_found["val"], legit_val, ESCALATED)
    policy = {c: "review" for c in PAYMENT_CHECKS} | {
        ESCALATED: "alert"
        if esc_rate <= ALERT_MAX
        else "review"
        if esc_rate <= REVIEW_MAX
        else "ignore"
    }

    masks: dict[str, tuple[str, np.ndarray]] = {
        "legitimate val": ("val", legit_val),
        "legitimate test": ("test", (frames["test"]["kind"] == "legitimate").to_numpy()),
        "phishing (Nazario) test": ("test", (frames["test"]["kind"] == "phishing").to_numpy()),
        "fraud (advance-fee) test": ("test", (frames["test"]["kind"] == "fraud").to_numpy()),
    }
    if "pot" in frames:
        for split in ("dev", "test"):
            masks[f"phishing pot {split}"] = ("pot", (frames["pot"]["split"] == split).to_numpy())
    check_rates = {
        label: {c: _rate(pay_found[k], m, c) for c in CHECKS} for label, (k, m) in masks.items()
    }

    def configs(k: str) -> dict[str, Any]:
        d, sf, pf = frames[k], sender_found[k], pay_found[k]
        both = [a | b for a, b in zip(sf, pf, strict=True)]
        out = {}
        for config, found, pol in (
            ("text + link + sender", sf, sender_policy),
            ("+ payment checks", both, sender_policy | policy),
        ):
            mal, sus = verdicts(d, found, pol, *thr, use_sender=True)
            if k == "pot":
                out[config] = {
                    s: _rates(mal, sus, (d["split"] == s).to_numpy()) for s in ("dev", "test")
                }
            else:
                kind = d["kind"].to_numpy()
                out[config] = {
                    g: _rates(mal, sus, kind == g)
                    for g in ("legitimate", "phishing", "fraud", "spam")
                }
        return out

    summary: dict[str, Any] = {
        "policy": policy,
        "limits": {"alert_max": ALERT_MAX, "review_max": REVIEW_MAX},
        "check_rates": check_rates,
        "results": {k: configs(k) for k in frames if k != "val"},
    }
    path = params.resolve(PAYMENT_POLICY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"policy": policy, "rates_on_legitimate_val": check_rates["legitimate val"]},
            indent=2,
        ),
        encoding="utf-8",
    )
    reports = params.resolve(REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "payment_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (reports / "payment_results.md").write_text(report(summary), encoding="utf-8")
    return summary


def report(s: dict[str, Any]) -> str:
    cols = list(s["check_rates"])
    lines = [
        "# Payment and invoice-fraud checks\n",
        "Wording checks (new bank details, gift cards, wire transfers, payment urgency) only "
        "mark an email for review: the corpora hold few genuine invoices, so their "
        "legitimate false-alarm rate would not carry over to a finance inbox. A payment "
        "request together with a From-line warning sign is measured: alert at most 0.1% of "
        "legitimate validation emails, review at most 2%.\n",
        "None of these corpora is invoice fraud (the fraud kind is advance-fee scams), so "
        "the catch rates below are not an invoice-fraud detection rate.\n",
        "## How often each check fires\n",
        "| check | role | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for c in CHECKS:
        lines.append(
            f"| {c} | {s['policy'][c]} | "
            + " | ".join(_pct(s["check_rates"][k][c]) for k in cols)
            + " |"
        )
    lines += [
        "",
        "## Test results\n",
        "| emails | n | malicious | + payment | malicious or suspicious | + payment |",
        "|---|---|---|---|---|---|",
    ]
    for k, res in s["results"].items():
        base, new = res["text + link + sender"], res["+ payment checks"]
        for g in base:
            name = f"Phishing Pot {g}" if k == "pot" else g
            a, b = base[g], new[g]
            lines.append(
                f"| {name} | {a['n']} | {_pct(a['malicious'])} | {_pct(b['malicious'])} | "
                f"{_pct(a['malicious_or_suspicious'])} | {_pct(b['malicious_or_suspicious'])} |"
            )
    lines.append("")
    return "\n".join(lines)
