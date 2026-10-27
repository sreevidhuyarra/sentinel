"""Measure the sender checks and decide how far each may be trusted.

Data
- The email corpora's validation and test splits (legitimate, phishing, fraud and spam, with
  the text model's Kaggle-run scores). Only the From line survives in these CSVs, so they
  can measure the address checks but not Reply-To or SPF/DKIM/DMARC.
- Phishing Pot (github.com/rf-peixoto/phishing_pot, CC BY-NC 4.0): real phishing emails
  received from about 2022 on, as raw .eml with headers. Split by sample number (arrival
  order): the earlier half is for development, the later half is the held-out test.
  Emails whose text also appears in the corpora are dropped.

Policy (fixed before any Phishing Pot result was seen)
- A check may raise an alert by itself if it fires on at most `alert_max` of legitimate
  validation emails (default 0.1%), and all such checks together add at most
  `combined_max` (0.5%) false alarms beyond the text and link models.
- A check that fires on at most `review_max` (2%) only marks an email suspicious.
- Header checks (Reply-To, SPF, DKIM, DMARC) cannot be measured on legitimate mail here, so
  they only mark an email suspicious.
- Everything else is reported but ignored.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from sentinel.common.config import Params
from sentinel.common.logging import get_logger
from sentinel.phishing.data import MIN_TEXT_CHARS, read_corpus
from sentinel.phishing.sender import (
    HEADER_CHECKS,
    check_sender,
    parse_eml,
    parse_sender,
)
from sentinel.phishing.text import dedup_key, extract_urls, model_input

log = get_logger(__name__)

POT_URL = "https://github.com/rf-peixoto/phishing_pot"
POT_COMMIT = "2933c0341b1248678c1e246b36d352add2edf0f1"
POT_ZIP = Path("data/raw/phishing_pot") / f"phishing_pot-{POT_COMMIT[:7]}.zip"
POT_CACHE = Path("data/interim/phishing_pot.parquet")
POLICY_PATH = Path("models/phishing/sender_policy.json")
BUNDLE_DIR = Path("models/phishing/bundle")
URL_BUNDLE_DIR = Path("models/phishing/url_bundle")
REPORTS = Path("reports/phishing")
ALERT_MAX, REVIEW_MAX, COMBINED_MAX = 0.001, 0.02, 0.005
SCORE_CHUNK = 500
CHECKS = (
    "brand_impersonation",
    "address_in_name",
    "lookalike_domain",
    "brand_in_domain",
    "punycode_domain",
    "freemail_role",
    "suspicious_tld",
    "random_address",
    "no_address",
    "reply_to_elsewhere",
    "dmarc_fail",
    "spf_fail",
    "dkim_fail",
)


# -- data ----------------------------------------------------------------------------------


def corpus_senders(params: Params) -> dict[str, str]:
    """Text key -> raw From line, rebuilt from the corpus CSVs (the processed data keeps
    only the sender's domain)."""
    p = params.phishing
    out: dict[str, str] = {}
    for f in sorted(params.resolve(p.raw_dir).glob("*.csv")):
        df = read_corpus(f)
        df = df[df["sender"].fillna("").str.strip() != ""]
        for s, b, sender in zip(df["subject"], df["body"], df["sender"], strict=True):
            out.setdefault(dedup_key(model_input(s, b, p.max_chars)), str(sender))
    return out


def corpus_frame(params: Params, split: str, senders: dict[str, str]) -> pd.DataFrame:
    from sentinel.phishing.url_train import _email_frame
    from sentinel.phishing.urls import UrlModel

    url = UrlModel.load(params.resolve(URL_BUNDLE_DIR))
    df = _email_frame(params, url, split)
    texts = (
        pl.read_parquet(params.resolve(params.phishing.processed_dir) / "emails.parquet")
        .filter(pl.col("split") == split)
        .select("id", "source", "text")
        .to_pandas()
    )
    df = df.merge(texts, on="id")
    df["from_"] = [senders.get(dedup_key(t), "") for t in df["text"]]
    df["reply_to"], df["auth"] = "", ""
    return df.drop(columns=["text"])


def _sample_number(name: str) -> int:
    m = re.search(r"sample-(\d+)\.eml$", name)
    return int(m.group(1)) if m else -1


def load_pot(params: Params, zip_path: Path, cache: Path, corpus_keys: set[str]) -> pd.DataFrame:
    """Phishing Pot emails parsed straight from the zip (attachments never decoded),
    deduplicated, without texts seen in the corpora, scored by the text and link models."""
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted((n for n in z.namelist() if n.endswith(".eml")), key=_sample_number)
        for name in names:
            try:
                e = parse_eml(z.read(name))
            except Exception as exc:  # a malformed message must not stop the run
                log.warning("skipping %s: %s", name, exc)
                continue
            text = model_input(e.subject, e.body, params.phishing.max_chars)
            rows.append(
                {
                    "sample": _sample_number(name),
                    "text": text,
                    "urls": extract_urls(f"{e.subject}\n{e.body}"),
                    "from_": e.sender.from_,
                    "reply_to": e.sender.reply_to,
                    "auth": json.dumps(e.sender.auth),
                }
            )
    df = pd.DataFrame(rows)
    n_parsed = len(df)
    df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
    df["key"] = df["text"].map(dedup_key)
    df = df.drop_duplicates("key")
    n_unique = len(df)
    df = df[~df["key"].isin(corpus_keys)].reset_index(drop=True)
    log.info(
        "phishing pot: %d parsed, %d unique, %d not in the corpora", n_parsed, n_unique, len(df)
    )
    df["split"] = np.where(df["sample"] <= df["sample"].median(), "dev", "test")
    df["label"], df["kind"], df["source"] = 1, "phishing", "phishing_pot"
    df = _score_pot(params, df)
    df.attrs = {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["key"]).assign(n_parsed=n_parsed, n_unique=n_unique).to_parquet(cache)
    return pd.read_parquet(cache)


def _score_pot(params: Params, df: pd.DataFrame) -> pd.DataFrame:
    from sentinel.phishing.export import PhishingOnnxModel
    from sentinel.phishing.urls import UrlModel

    # ~4 emails/s on a laptop CPU (~1 h): scored in chunks saved as they finish, so an
    # interrupted run resumes where it stopped.
    ph = PhishingOnnxModel.load(params.resolve(BUNDLE_DIR))
    parts = params.resolve(POT_CACHE).with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    probs = []
    for start in range(0, len(df), SCORE_CHUNK):
        part = parts / f"text-{start:06d}-{len(df)}.npy"
        if not part.exists():
            log.info("scoring emails %d-%d of %d", start, start + SCORE_CHUNK, len(df))
            chunk = df["text"].iloc[start : start + SCORE_CHUNK].tolist()
            np.save(part, ph.predict_proba(chunk))
        probs.append(np.load(part))
    df["prob"] = np.concatenate(probs) if probs else np.zeros(0)
    url = UrlModel.load(params.resolve(URL_BUNDLE_DIR))
    flat = [(i, u) for i, urls in enumerate(df["urls"]) for u in list(urls)[:20]]
    best = np.full(len(df), -1.0)
    if flat:
        for (i, _), p in zip(flat, url.predict_proba([u for _, u in flat]), strict=True):
            best[i] = max(best[i], p)
    df["max_url_prob"] = best
    return df


# -- checks and policy ---------------------------------------------------------------------


def findings_of(df: pd.DataFrame) -> list[set[str]]:
    return [
        {f.check for f in check_sender(parse_sender(fr, rt, _auth_text(a)))}
        for fr, rt, a in zip(df["from_"], df["reply_to"], df["auth"], strict=True)
    ]


def _auth_text(a: str) -> str:
    """parse_sender wants a header string; the cache stores the parsed dict as JSON."""
    if not a:
        return ""
    return " ".join(f"{k}={v}" for k, v in json.loads(a).items())


def rates(found: list[set[str]], mask: np.ndarray) -> dict[str, float]:
    n = int(mask.sum())
    c: Counter[str] = Counter(x for f, m in zip(found, mask, strict=True) if m for x in f)
    return {k: (c[k] / n if n else float("nan")) for k in CHECKS}


def choose_policy(
    legit_found: list[set[str]],
    already_alerted: np.ndarray,
    alert_max: float = ALERT_MAX,
    review_max: float = REVIEW_MAX,
    combined_max: float = COMBINED_MAX,
) -> tuple[dict[str, str], dict[str, float]]:
    """check -> "alert" / "review" / "ignore", from legitimate validation emails only."""
    n = len(legit_found)
    rate = rates(legit_found, np.ones(n, bool))
    policy = {}
    extra = np.zeros(n, bool)
    for check in sorted(CHECKS, key=lambda k: rate[k]):
        if check in HEADER_CHECKS:
            policy[check] = "review"
            continue
        fires = np.array([check in f for f in legit_found])
        new_extra = extra | (fires & ~already_alerted)
        if rate[check] <= alert_max and new_extra.mean() <= combined_max:
            policy[check], extra = "alert", new_extra
        elif rate[check] <= review_max:
            policy[check] = "review"
        else:
            policy[check] = "ignore"
    return policy, rate


def verdicts(
    df: pd.DataFrame,
    found: list[set[str]],
    policy: dict[str, str],
    text_thr: float,
    review_thr: float,
    url_thr: float,
    url_email_thr: float,
    use_sender: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """(malicious, suspicious) per email, by the same rules as POST /score/phishing."""
    prob, url = df["prob"].to_numpy(), df["max_url_prob"].to_numpy()
    alert = np.array([any(policy.get(c) == "alert" for c in f) for f in found]) & use_sender
    review = np.array([any(policy.get(c) == "review" for c in f) for f in found]) & use_sender
    malicious = (prob >= text_thr) | (url >= url_email_thr) | alert
    suspicious = ~malicious & ((prob >= review_thr) | (url >= url_thr) | review)
    return malicious, suspicious


# -- run -----------------------------------------------------------------------------------


def run(params: Params, zip_path: Path | None = None) -> dict[str, Any]:
    from sentinel.phishing.export import PhishingOnnxModel
    from sentinel.phishing.urls import UrlModel

    p = params.phishing
    ph = PhishingOnnxModel.load(params.resolve(BUNDLE_DIR))
    um = UrlModel.load(params.resolve(URL_BUNDLE_DIR))
    text_thr, review_thr = ph.threshold, p.suspicious_threshold
    url_thr, url_email_thr = um.threshold, um.email_threshold or 1.1

    senders = corpus_senders(params)
    val = corpus_frame(params, "val", senders)
    test = corpus_frame(params, "test", senders)
    corpus_keys = set(
        pl.read_parquet(params.resolve(p.processed_dir) / "emails.parquet")["text"]
        .map_elements(dedup_key, return_dtype=pl.String)
        .to_list()
    )
    pot = load_pot(
        params, params.resolve(zip_path or POT_ZIP), params.resolve(POT_CACHE), corpus_keys
    )
    log.info("corpus emails: val %d, test %d; phishing pot %d", len(val), len(test), len(pot))

    has = {
        name: (d["from_"].str.strip() != "").to_numpy()
        for name, d in (("val", val), ("test", test))
    }
    legit_val = (val["label"] == 0).to_numpy() & has["val"]
    val_found, test_found, pot_found = findings_of(val), findings_of(test), findings_of(pot)
    base_val, _ = verdicts(
        val, val_found, {}, text_thr, review_thr, url_thr, url_email_thr, use_sender=False
    )
    policy, legit_rate = choose_policy(
        [f for f, m in zip(val_found, legit_val, strict=True) if m], base_val[legit_val]
    )

    def groups(d: pd.DataFrame, has_sender: np.ndarray) -> dict[str, np.ndarray]:
        kind = d["kind"].to_numpy()
        out = {
            "legitimate": (kind == "legitimate") & has_sender,
            "phishing (Nazario)": (kind == "phishing") & has_sender,
            "fraud": (kind == "fraud") & has_sender,
            "spam": (kind == "spam") & has_sender,
        }
        return out

    pot_dev = (pot["split"] == "dev").to_numpy()
    check_rates = {
        "legitimate val": legit_rate,
        "phishing (Nazario) val": rates(
            val_found, (val["kind"] == "phishing").to_numpy() & has["val"]
        ),
        "phishing pot dev": rates(pot_found, pot_dev),
        "legitimate test": rates(
            test_found, (test["kind"] == "legitimate").to_numpy() & has["test"]
        ),
        "phishing pot test": rates(pot_found, ~pot_dev),
    }

    results: dict[str, Any] = {}
    for split_name, d, found, g in (
        ("test", test, test_found, groups(test, has["test"])),
        ("val", val, val_found, groups(val, has["val"])),
    ):
        results[split_name] = {}
        for config, use in (("text + link", False), ("text + link + sender", True)):
            mal, sus = verdicts(d, found, policy, text_thr, review_thr, url_thr, url_email_thr, use)
            results[split_name][config] = {name: _rates(mal, sus, m) for name, m in g.items()}
    for split_name, m in (("dev", pot_dev), ("test", ~pot_dev)):
        sub = pot[m].reset_index(drop=True)
        found = [f for f, keep in zip(pot_found, m, strict=True) if keep]
        out = {}
        for config, use, pol in (
            ("text + link", False, policy),
            ("text + link + sender (From line only)", True, _from_only(policy)),
            ("text + link + sender (with headers)", True, policy),
        ):
            mal, sus = verdicts(sub, found, pol, text_thr, review_thr, url_thr, url_email_thr, use)
            out[config] = _rates(mal, sus, np.ones(len(sub), bool))
        results[f"phishing pot {split_name}"] = out

    summary = {
        "policy": policy,
        "limits": {"alert_max": ALERT_MAX, "review_max": REVIEW_MAX, "combined_max": COMBINED_MAX},
        "thresholds": {
            "text": text_thr,
            "review": review_thr,
            "url": url_thr,
            "url_email": url_email_thr,
        },
        "check_rates": check_rates,
        "results": results,
        "counts": {
            "legitimate val with sender": int(legit_val.sum()),
            "legitimate test with sender": int(
                ((test["kind"] == "legitimate") & has["test"]).sum()
            ),
            "phishing pot parsed": int(pot["n_parsed"].iloc[0]) if len(pot) else 0,
            "phishing pot unique": int(pot["n_unique"].iloc[0]) if len(pot) else 0,
            "phishing pot dev": int(pot_dev.sum()),
            "phishing pot test": int((~pot_dev).sum()),
        },
        "source": {"phishing_pot": POT_URL, "commit": POT_COMMIT, "license": "CC BY-NC 4.0"},
    }
    policy_path = params.resolve(POLICY_PATH)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps({"policy": policy, "rates_on_legitimate_val": legit_rate}, indent=2),
        encoding="utf-8",
    )
    reports = params.resolve(REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "sender_results.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    (reports / "sender_results.md").write_text(report(summary), encoding="utf-8")
    return summary


def _from_only(policy: dict[str, str]) -> dict[str, str]:
    return {k: ("ignore" if k in HEADER_CHECKS else v) for k, v in policy.items()}


def _rates(mal: np.ndarray, sus: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    n = int(mask.sum())
    return {
        "n": n,
        "malicious": float(mal[mask].mean()) if n else float("nan"),
        "malicious_or_suspicious": float((mal | sus)[mask].mean()) if n else float("nan"),
    }


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.2f}%"


def report(s: dict[str, Any]) -> str:
    lines = [
        "# Sender checks\n",
        "Who an email claims to be from versus where it came from. Checks are measured on "
        "legitimate validation mail and given a role: **alert** (may flag an email alone), "
        "**review** (marks it suspicious) or **ignore**. Header checks (Reply-To, SPF, DKIM, "
        "DMARC) need the raw .eml; the corpora keep only the From line, so their false-alarm "
        "rate on legitimate mail is unmeasured and they are limited to review.\n",
        f"Modern phishing: [Phishing Pot]({POT_URL}) at commit `{POT_COMMIT[:7]}` "
        "(CC BY-NC 4.0), split by arrival order; the later half is the held-out test.\n",
        "## Policy and how often each check fires\n",
        "| check | role | legitimate val | phishing (Nazario) val | Phishing Pot dev "
        "| legitimate test | Phishing Pot test |",
        "|---|---|---|---|---|---|---|",
    ]
    cr = s["check_rates"]
    for c in CHECKS:
        lines.append(
            f"| {c} | {s['policy'].get(c, 'ignore')} | "
            + " | ".join(
                _pct(cr[k][c])
                for k in (
                    "legitimate val",
                    "phishing (Nazario) val",
                    "phishing pot dev",
                    "legitimate test",
                    "phishing pot test",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Test results\n",
        "Share flagged **malicious** (an alert) and **malicious or suspicious** (an alert or "
        "a review). Corpus emails count only when they have a From line.\n",
        "| emails | n | text + link: malicious | + sender: malicious | text + link: "
        "malicious or suspicious | + sender: malicious or suspicious |",
        "|---|---|---|---|---|---|",
    ]
    t = s["results"]["test"]
    for g in t["text + link"]:
        a, b = t["text + link"][g], t["text + link + sender"][g]
        lines.append(
            f"| {g} | {a['n']} | {_pct(a['malicious'])} | {_pct(b['malicious'])} | "
            f"{_pct(a['malicious_or_suspicious'])} | {_pct(b['malicious_or_suspicious'])} |"
        )
    lines += [
        "",
        "Phishing Pot (modern phishing, raw .eml):\n",
        "| split | configuration | n | malicious | malicious or suspicious |",
        "|---|---|---|---|---|",
    ]
    for split in ("phishing pot dev", "phishing pot test"):
        for config, r in s["results"][split].items():
            lines.append(
                f"| {split.removeprefix('phishing pot ')} | {config} | {r['n']} | "
                f"{_pct(r['malicious'])} | {_pct(r['malicious_or_suspicious'])} |"
            )
    lines.append("")
    return "\n".join(lines)
