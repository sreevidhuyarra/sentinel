from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.phishing.url_train import fit, metrics, split, threshold_at_budget
from sentinel.phishing.urls import (
    LEXICAL,
    UrlModel,
    lexical_features,
    normalize,
    registered_domain,
)


@pytest.mark.parametrize(
    ("url", "norm"),
    [
        ("HTTPS://WWW.Example.com/", "example.com"),
        ("http://www2.shop.example.co.uk/path/", "shop.example.co.uk/path"),
        ("example.com/login", "example.com/login"),
    ],
)
def test_normalize(url: str, norm: str) -> None:
    assert normalize(url) == norm


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://mail.google.com/x", "google.com"),
        ("http://a.b.example.co.uk", "example.co.uk"),
        ("http://192.168.1.10:8080/login", "192.168.1.10"),
        ("paypal.com.secure-login.xyz/verify", "secure-login.xyz"),
    ],
)
def test_registered_domain(url: str, domain: str) -> None:
    assert registered_domain(url) == domain


def test_lexical_features() -> None:
    f = dict(
        zip(
            LEXICAL,
            lexical_features("http://192.168.1.10:8080/secure/login.php?a=1&b=2"),
            strict=True,
        )
    )
    assert len(f) == len(LEXICAL)
    assert f["host_is_ip"] == 1 and f["has_port"] == 1 and f["is_https"] == 0
    assert f["query_param_count"] == 2 and f["sensitive_word_count"] >= 2
    g = dict(zip(LEXICAL, lexical_features("https://bit.ly/abc"), strict=True))
    assert g["is_shortener"] == 1 and g["is_https"] == 1
    h = dict(zip(LEXICAL, lexical_features("http://free-gift.tk"), strict=True))
    assert h["suspicious_tld"] == 1
    assert lexical_features("http://[::bad")  # malformed input must not raise


def _synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        if i % 2:
            rows.append(
                (f"http://secure-login-{i}.xyz/account/verify.php?id={rng.integers(1e9)}", 1)
            )
        else:
            rows.append((f"https://www.company{i}.com/about", 0))
    df = pd.DataFrame(rows, columns=["url", "label"])
    df["source"] = "synthetic"
    df["domain"] = df["url"].map(registered_domain)
    return df


def test_fit_threshold_save_load_explain(tmp_path: Path) -> None:
    df = split(_synthetic(), seed=0)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    assert set(parts["train"]["domain"]).isdisjoint(parts["test"]["domain"])
    model = fit(parts["train"], parts["val"], max_features=2000, seed=0)
    val_p = model.predict_proba(parts["val"]["url"].tolist())
    model.threshold = threshold_at_budget(val_p[parts["val"]["label"].to_numpy() == 0], 0.05)
    m = metrics(
        parts["test"]["label"].to_numpy(),
        model.predict_proba(parts["test"]["url"].tolist()),
        model.threshold,
    )
    assert m["roc_auc"] > 0.95

    model.save(tmp_path / "b")
    again = UrlModel.load(tmp_path / "b")
    urls = parts["test"]["url"].tolist()[:20]
    assert np.allclose(model.predict_proba(urls), again.predict_proba(urls))
    reasons = again.explain("http://secure-login-7.xyz/account/verify.php?id=3")
    assert reasons and all(r["contribution"] > 0 for r in reasons)
    assert again.predict_proba([]).shape == (0,)
