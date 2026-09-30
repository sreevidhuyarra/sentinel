"""Kaggle LoRA adapter -> merged model -> ONNX (fp32) -> dynamic int8 ONNX, plus the
ONNX Runtime inference wrapper used for serving (no PyTorch needed at inference time)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BASE_MODEL = "microsoft/deberta-v3-small"
OPSET = 17


def merge_adapter(adapter_dir: Path, base_model: str = BASE_MODEL) -> Any:
    """Base model (float32) with the LoRA weights folded in; an ordinary transformers model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model, dtype=torch.float32, num_labels=2
    )
    return PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().eval()


def export_onnx(model: Any, tokenizer: Any, path: Path, max_len: int, dynamo: bool = True) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    # Trace with a *padded* batch. The tracer records one code path, and transformers skips
    # the attention mask when it is all ones; tracing an unpadded example bakes that shortcut
    # in, so padded batches at serving time would treat padding as text.
    sample = tokenizer(
        ["Verify your account at [URL]", "Meeting moved to 3pm, agenda attached as before."],
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )
    assert int(sample["attention_mask"].min()) == 0, "export sample must contain padding"

    class Wrapper(torch.nn.Module):
        """Exposes only input_ids / attention_mask and returns logits."""

        def __init__(self, m: Any) -> None:
            super().__init__()
            self.m = m

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            logits: torch.Tensor = self.m(input_ids=input_ids, attention_mask=attention_mask).logits
            return logits

    args = (sample["input_ids"], sample["attention_mask"])
    with torch.no_grad():
        if dynamo:
            # torch.export-based exporter. The legacy TorchScript exporter mistranslates
            # DeBERTa's masked softmax: its graphs are exact on unpadded inputs but wrong on
            # padded batches (logit errors up to ~2).
            batch = torch.export.Dim("batch", min=1, max=1024)
            seq = torch.export.Dim("seq", min=2, max=max_len)
            onnx_program = torch.onnx.export(
                Wrapper(model),
                args,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids": {0: batch, 1: seq},
                    "attention_mask": {0: batch, 1: seq},
                },
                opset_version=OPSET,
                dynamo=True,
            )
            assert onnx_program is not None
            onnx_program.save(str(path), external_data=False)
        else:
            torch.onnx.export(
                Wrapper(model),
                args,
                str(path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"},
                },
                opset_version=OPSET,
                dynamo=False,
            )
    return path


def quantize_int8(src: Path, dst: Path, mode: str = "embeddings") -> Path:
    """Dynamic int8 quantization.

    mode="embeddings": only the 128k x 768 token-embedding table (Gather), ~70% of the
    weights. Measured on this model: 2.1x smaller, identical ROC-AUC and F1.
    mode="full": every MatMul as well (activations quantized on the fly): 3.4x smaller and
    1.6x faster on CPU, but DeBERTa's outlier activations cost 1-2 F1 points.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # The dynamo exporter stores intermediate shape annotations (value_info) that clash with
    # the quantizer's own shape inference; they are optional, so drop them and let it infer.
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    stripped = dst.with_name(dst.stem + ".noshapes.onnx")
    onnx.save(model, str(stripped))
    if mode not in ("embeddings", "full"):
        raise ValueError(f"unknown quantization mode {mode!r}")
    kwargs: dict[str, Any] = {"op_types_to_quantize": ["Gather"]} if mode == "embeddings" else {}
    try:
        quantize_dynamic(str(stripped), str(dst), weight_type=QuantType.QInt8, **kwargs)
    finally:
        stripped.unlink(missing_ok=True)
    return dst


@dataclass
class PhishingOnnxModel:
    """Tokenizer + ONNX Runtime session. `predict_proba` returns P(malicious) per email text."""

    session: Any
    tokenizer: Any
    max_len: int
    threshold: float
    metadata: dict[str, Any]

    @classmethod
    def load(
        cls, bundle_dir: Path, model_file: str | None = None, threads: int | None = None
    ) -> PhishingOnnxModel:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        cfg = json.loads((bundle_dir / "phishing.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        session = ort.InferenceSession(
            str(bundle_dir / (model_file or cfg["model_file"])),
            opts,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = AutoTokenizer.from_pretrained(bundle_dir / "tokenizer")
        return cls(session, tokenizer, cfg["max_len"], cfg["threshold"], cfg.get("metadata", {}))

    def logits(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="np"
        )
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        return np.asarray(self.session.run(["logits"], feeds)[0], dtype=np.float64)

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        # Sorting by length keeps padding (wasted compute) small inside each batch.
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(texts), batch_size):
            idx = order[i : i + batch_size]
            z = self.logits([texts[j] for j in idx])
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            out.append((idx, e[:, 1] / e.sum(axis=1)))
        prob = np.empty(len(texts))
        for idx, p in out:
            prob[idx] = p
        return prob
