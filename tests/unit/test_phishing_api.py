"""/score/phishing end to end with a tiny, fully local ONNX bundle (no downloads)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from sentinel.phishing.explain import explain, segments
from sentinel.phishing.export import PhishingOnnxModel, export_onnx
from sentinel.phishing.url_train import fit
from sentinel.phishing.urls import UrlModel
from sentinel.services.api import create_app

WORDS = [
    "urgent",
    "verify",
    "your",
    "account",
    "password",
    "bank",
    "click",
    "here",
    "now",
    "meeting",
    "agenda",
    "attached",
    "notes",
    "thanks",
    "url",
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        DebertaV2Config,
        DebertaV2ForSequenceClassification,
        PreTrainedTokenizerFast,
    )

    root = tmp_path_factory.mktemp("phishing_bundle")
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        **{w: i + 4 for i, w in enumerate(WORDS)},
    }
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(root / "tokenizer")
    torch.manual_seed(0)
    cfg = DebertaV2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        position_buckets=16,
        pos_att_type=["p2c", "c2p"],
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        position_biased_input=False,
        type_vocab_size=0,
        num_labels=2,
    )
    export_onnx(
        DebertaV2ForSequenceClassification(cfg).eval(), tokenizer, root / "model.onnx", max_len=32
    )
    # Threshold 0 flags every email, so the explanation path always runs.
    (root / "phishing.json").write_text(
        json.dumps({"model_file": "model.onnx", "max_len": 32, "threshold": 0.0})
    )
    return root


@pytest.fixture(scope="module")
def url_model() -> UrlModel:
    rows = [
        (f"http://secure-login-{i}.xyz/account/verify.php?id={i * 7919}", 1)
        if i % 2
        else (f"https://www.company{i}.com/about", 0)
        for i in range(400)
    ]
    df = pd.DataFrame(rows, columns=["url", "label"])
    model = fit(df.iloc[:300], df.iloc[300:], max_features=2000, seed=0)
    model.threshold = 0.5
    model.email_threshold = 0.5
    return model


@pytest.fixture(scope="module")
def client(bundle: Path, url_model: UrlModel) -> Iterator[TestClient]:
    # The flow endpoints are not exercised here, so a placeholder stands in for the IDS bundle.
    app = create_app(
        bundle=object(),  # type: ignore[arg-type]
        phishing=PhishingOnnxModel.load(bundle),
        phishing_source="test",
        url=url_model,
        url_source="url-test",
    )
    with TestClient(app) as c:
        yield c


def test_segments_split_sentences_and_lines() -> None:
    assert segments("Hello there. Click here now!\nThanks") == [
        "Hello there.",
        "Click here now!",
        "Thanks",
    ]


def test_predict_proba_is_order_independent(bundle: Path) -> None:
    m = PhishingOnnxModel.load(bundle)
    texts = [
        "urgent verify your password",
        "meeting agenda attached",
        "click here now bank account url",
    ]
    p = m.predict_proba(texts)
    assert abs(p[1] - m.predict_proba([texts[1]])[0]) < 1e-5
    assert ((p >= 0) & (p <= 1)).all()


def test_explain_returns_at_most_three_positive_reasons(bundle: Path) -> None:
    m = PhishingOnnxModel.load(bundle)
    reasons = explain(
        m, "Urgent. Verify your account. Click here now. Thanks. Meeting notes attached."
    )
    assert len(reasons) <= 3 and all(r["contribution"] > 0 for r in reasons)


def test_score_phishing_endpoint(client: TestClient) -> None:
    r = client.post(
        "/score/phishing",
        json={
            "subject": "Urgent",
            "body": "Verify your account at http://bank.example/login now. Thanks.",
            "urls": ["http://extra.example"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["is_malicious"] is True and body["severity"] in ("Medium", "High")
    assert body["urls"] == ["http://bank.example/login", "http://extra.example"]
    assert body["model_version"] == "test"
    assert client.get("/health").json()["phishing_model_loaded"] is True


def test_combined_verdict_scores_links(client: TestClient) -> None:
    body = client.post(
        "/score/phishing",
        json={
            "subject": "Hi",
            "body": "See http://secure-login-9.xyz/account/verify.php?id=1 and "
            "https://www.company8.com/about",
        },
    ).json()
    by_url = {r["url"]: r for r in body["url_results"]}
    assert by_url["http://secure-login-9.xyz/account/verify.php?id=1"]["is_malicious"] is True
    assert by_url["https://www.company8.com/about"]["is_malicious"] is False
    assert "url" in body["flagged_by"] and body["url_model_version"] == "url-test"
    assert by_url["http://secure-login-9.xyz/account/verify.php?id=1"]["reasons"]


def test_score_url_endpoint(client: TestClient) -> None:
    r = client.post(
        "/score/url", json={"urls": ["http://secure-login-3.xyz/account/verify.php?id=2"]}
    )
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["is_malicious"] is True and res["reasons"]
    assert client.get("/health").json()["url_model_loaded"] is True
    assert client.post("/score/url", json={"urls": []}).status_code == 422


def test_score_phishing_validation_and_unavailable(client: TestClient) -> None:
    assert client.post("/score/phishing", json={"subject": "x", "body": ""}).status_code == 422
    app = create_app(bundle=object())  # type: ignore[arg-type]
    with TestClient(app) as c:
        assert c.post("/score/phishing", json={"body": "hello"}).status_code == 503
        assert c.post("/score/url", json={"urls": ["http://a.b"]}).status_code == 503
