"""ONNX export, quantization and build helpers for the phishing classifier.

Uses a tiny randomly-initialised DeBERTa-v2 (same relative-attention setup as v3-small)
so no download is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from sentinel.phishing.build import split_metrics, threshold_at_budget
from sentinel.phishing.export import export_onnx, quantize_int8
from sentinel.phishing.report import model_card, results_markdown


class _FakeTokenizer:
    """Maps words to ids; enough for export_onnx's sample batch."""

    def __call__(self, texts: list[str], **kw: Any) -> dict[str, torch.Tensor]:
        ids = [[1] + [5 + (hash(w) % 90) for w in t.split()] + [2] for t in texts]
        n = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [0] * (n - len(x)) for x in ids])
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}


@pytest.fixture(scope="module")
def tiny_model() -> Any:
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

    torch.manual_seed(0)
    cfg = DebertaV2Config(
        vocab_size=100,
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
    return DebertaV2ForSequenceClassification(cfg).eval()


def test_onnx_export_is_exact_on_padded_batches(tiny_model: Any, tmp_path: Path) -> None:
    """Regression test: the legacy TorchScript exporter produced graphs that were exact on
    unpadded input but wrong on padded batches (logit errors up to ~2)."""
    path = export_onnx(tiny_model, _FakeTokenizer(), tmp_path / "m.onnx", max_len=32)
    rng = np.random.default_rng(0)
    lengths = [5, 12, 20, 32]
    ids = np.zeros((4, 32), dtype=np.int64)
    for i, n in enumerate(lengths):
        ids[i, :n] = rng.integers(3, 100, n)
    mask = (ids != 0).astype(np.int64)
    onnx_logits = ort.InferenceSession(str(path)).run(
        ["logits"], {"input_ids": ids, "attention_mask": mask}
    )[0]
    with torch.no_grad():
        ref = np.stack(
            [
                tiny_model(
                    input_ids=torch.tensor(ids[i : i + 1, :n]),
                    attention_mask=torch.tensor(mask[i : i + 1, :n]),
                )
                .logits[0]
                .numpy()
                for i, n in enumerate(lengths)
            ]
        )
    assert np.abs(onnx_logits - ref).max() < 1e-4

    q = quantize_int8(path, tmp_path / "q.onnx", mode="embeddings")
    q_logits = ort.InferenceSession(str(q)).run(
        ["logits"], {"input_ids": ids, "attention_mask": mask}
    )[0]
    assert q_logits.shape == ref.shape
    with pytest.raises(ValueError, match="unknown quantization mode"):
        quantize_int8(path, tmp_path / "x.onnx", mode="bogus")


def _preds(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    kind = np.where(y == 0, "legitimate", rng.choice(["phishing", "fraud", "spam"], n))
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "label": y,
            "kind": kind,
            "source": rng.choice(["A", "B"], n),
            "prob": np.clip(0.75 * y + rng.normal(0.15, 0.2, n), 0, 1),
        }
    )


def test_threshold_at_budget_respects_false_alarm_budget() -> None:
    val = _preds()
    t = threshold_at_budget(val, 0.05)
    legit = val.loc[val["label"] == 0, "prob"]
    assert (legit >= t).mean() <= 0.05


def test_split_metrics_and_reports() -> None:
    df = _preds()
    m = split_metrics(df, 0.5)
    assert 0 < m["f1"] <= 1 and set(m["flagged_by_kind"]) == {
        "fraud",
        "legitimate",
        "phishing",
        "spam",
    }
    bench = {
        "pytorch_fp32": {
            "size_mb": None,
            "emails_per_s": 8.0,
            "p50_ms": 90.0,
            "p95_ms": 150.0,
            "f1": 0.97,
            "roc_auc": 0.99,
        },
        "onnx_int8_embeddings": {
            "size_mb": 273.0,
            "emails_per_s": 7.8,
            "p50_ms": 95.0,
            "p95_ms": 160.0,
            "f1": 0.97,
            "roc_auc": 0.99,
            "verdicts_differ_from_pytorch": 1,
            "max_prob_diff": 0.01,
        },
    }
    results = {
        "threshold": 0.9,
        "fpr_budget": 0.02,
        "deployed": "onnx_int8_embeddings",
        "accuracy": {"test": m, "val": m},
        "accuracy_at_0.5": {"test": m, "val": m},
        "benchmark_sample": 1000,
        "benchmark": bench,
        "training": {"minutes": 68.5, "gpus": ["Tesla T4"]},
    }
    md = results_markdown(results)
    assert "(deployed)" in md and "Before and after" in md
    assert "Model card: phishing-classifier" in model_card(results)
    json.dumps(results)
