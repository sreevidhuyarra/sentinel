"""URL classifier: character n-gram TF-IDF + hand-built lexical features -> LightGBM.

Data: PhiUSIIL Phishing URL Dataset (Prasad & Chandra 2024, UCI 967, CC BY 4.0) and the
Web page phishing detection dataset (Hannousse & Yahiouche 2021, CC BY 4.0), both with raw
URLs. ISCX-URL2016, named in the design report, sits behind a registration form and its
open mirrors ship only pre-computed features, which serving could not reproduce.

Known shortcut: every legitimate PhiUSIIL URL is a bare homepage. URLs are therefore
normalised (lowercase, no scheme, no leading "www.", no trailing "/") before the n-gram
features, the split is grouped by registered domain, and a cross-dataset test (train on
PhiUSIIL only, score Hannousse) measures how much the model relies on dataset style.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

_SECOND_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or"}
# Top-level domains over-represented in abuse reports (heuristic, not a verdict).
SUSPICIOUS_TLDS = frozenset(
    [
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
        "xyz",
        "top",
        "work",
        "click",
        "zip",
        "review",
        "country",
        "kim",
        "men",
        "loan",
        "date",
        "racing",
        "win",
        "download",
        "stream",
        "gdn",
        "mom",
        "bid",
        "trade",
        "party",
        "science",
        "icu",
        "buzz",
        "rest",
        "fit",
        "cyou",
        "cfd",
        "sbs",
    ]
)
SHORTENERS = frozenset(
    [
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "t.co",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "cutt.ly",
        "rb.gy",
        "tiny.cc",
        "shorturl.at",
    ]
)
SENSITIVE = (
    "login",
    "signin",
    "verify",
    "secure",
    "account",
    "update",
    "bank",
    "confirm",
    "password",
    "webscr",
    "paypal",
    "wallet",
    "billing",
    "unlock",
    "support",
)

LEXICAL = [
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "subdomain_count",
    "dot_count",
    "hyphen_count",
    "digit_ratio",
    "special_char_count",
    "has_at_sign",
    "host_is_ip",
    "is_https",
    "has_port",
    "path_depth",
    "query_param_count",
    "url_entropy",
    "host_entropy",
    "suspicious_tld",
    "sensitive_word_count",
    "is_shortener",
    "punycode",
    "longest_token",
    "host_digit_count",
]


def _split(url: str) -> Any:
    url = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.IGNORECASE):
        url = "http://" + url
    try:
        return urlsplit(url)
    except ValueError:
        return urlsplit("http://invalid")


def host_of(url: str) -> str:
    return (_split(url).hostname or "").lower().strip(".")


def registered_domain(url: str) -> str:
    host = host_of(url)
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def normalize(url: str) -> str:
    """Form used for n-gram features: lowercase, no scheme, no leading www., no trailing /."""
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url.strip().lower())
    u = re.sub(r"^www\d?\.", "", u)
    return u.rstrip("/")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def lexical_features(url: str) -> list[float]:
    p = _split(url)
    host = (p.hostname or "").lower()
    labels = [x for x in host.split(".") if x]
    try:
        ipaddress.ip_address(host)
        is_ip = 1.0
    except ValueError:
        is_ip = 0.0
    low = url.lower()
    try:
        has_port = float(p.port is not None)
    except ValueError:
        has_port = 1.0
    tokens = re.split(r"[/?.=&_\-]+", low)
    return [
        len(url),
        len(host),
        len(p.path),
        len(p.query),
        max(len(labels) - 2, 0),
        low.count("."),
        low.count("-"),
        sum(ch.isdigit() for ch in url) / max(len(url), 1),
        sum(low.count(c) for c in "@~%=&?!*$"),
        float("@" in url),
        is_ip,
        float(low.startswith("https://")),
        has_port,
        p.path.count("/"),
        len([q for q in p.query.split("&") if q]),
        _entropy(low),
        _entropy(host),
        float(bool(labels) and labels[-1] in SUSPICIOUS_TLDS),
        sum(low.count(w) for w in SENSITIVE),
        float(registered_domain(url) in SHORTENERS or host in SHORTENERS),
        float("xn--" in host),
        max((len(t) for t in tokens), default=0),
        sum(ch.isdigit() for ch in host),
    ]


def lexical_matrix(urls: list[str]) -> np.ndarray:
    return np.asarray([lexical_features(u) for u in urls], dtype=np.float64)


@dataclass
class UrlModel:
    vectorizer: Any  # sklearn TfidfVectorizer (char n-grams on normalized URLs)
    booster: Any  # lightgbm.Booster
    threshold: float
    metadata: dict[str, Any]
    # Threshold a link must reach to make an *email* malicious in the combined verdict,
    # tuned on the email validation split; None means links do not change email verdicts.
    email_threshold: float | None = None

    def features(self, urls: list[str]) -> Any:
        from scipy import sparse

        tfidf = self.vectorizer.transform([normalize(u) for u in urls])
        return sparse.hstack([tfidf, sparse.csr_matrix(lexical_matrix(urls))], format="csr")

    @property
    def feature_names(self) -> list[str]:
        return [f"contains '{g}'" for g in self.vectorizer.get_feature_names_out()] + LEXICAL

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        if not urls:
            return np.zeros(0)
        return np.asarray(self.booster.predict(self.features(urls)), dtype=np.float64)

    def explain(self, url: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Top features pushing this URL toward malicious (TreeSHAP, log-odds units)."""
        raw = self.booster.predict(self.features([url]), pred_contrib=True)
        # Sparse input makes LightGBM return contributions as a sparse matrix.
        contrib = np.asarray(raw.toarray() if hasattr(raw, "toarray") else raw)[0, :-1]
        names = self.feature_names
        lex = dict(zip(LEXICAL, lexical_features(url), strict=True))
        out = []
        for i in np.argsort(-contrib)[:top_k]:
            if contrib[i] <= 0:
                break
            name = names[i]
            text = f"{name.replace('_', ' ')} = {lex[name]:g}" if name in lex else f"URL {name}"
            out.append({"feature": name, "contribution": float(contrib[i]), "text": text})
        return out

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        self.booster.save_model(str(directory / "lightgbm.txt"))
        cfg = {
            "threshold": self.threshold,
            "email_threshold": self.email_threshold,
            "lexical_features": LEXICAL,
            "metadata": self.metadata,
        }
        (directory / "url.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> UrlModel:
        import joblib
        import lightgbm as lgb

        cfg = json.loads((directory / "url.json").read_text(encoding="utf-8"))
        if cfg["lexical_features"] != LEXICAL:
            raise ValueError("URL bundle was built with a different lexical feature set")
        return cls(
            vectorizer=joblib.load(directory / "vectorizer.joblib"),
            booster=lgb.Booster(model_file=str(directory / "lightgbm.txt")),
            threshold=cfg["threshold"],
            metadata=cfg.get("metadata", {}),
            email_threshold=cfg.get("email_threshold"),
        )
