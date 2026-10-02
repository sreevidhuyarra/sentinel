"""Train and evaluate the injection guard's classifier (report section 8.2).

Two candidates, chosen on validation like every other model in Sentinel:
- `embed_lr`: frozen bge-small sentence embeddings + logistic regression (seconds to train);
- `deberta`: DeBERTa-v3-small fine-tuned end to end on CPU (the design's choice).
Both learn from segments (the unit the guard scores); both are judged on whole documents
through `Guard.scan`, with the threshold set on validation for <= `max_fpr` false alarms
on clean documents. Test is scored once, per source, rules-only vs classifier vs both.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from sentinel.common.config import Params
from sentinel.common.logging import get_logger
from sentinel.copilot import guard_data
from sentinel.copilot.guard import Guard, segments

log = get_logger(__name__)

DEBERTA = "microsoft/deberta-v3-small"


def training_segments(docs: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Segment-level examples: payload pieces are positive, everything else negative."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for r in docs.iter_rows(named=True):
        if r["source"] != "synthetic_email":
            rows.append({"text": r["text"][:1000], "label": r["label"]})
            continue
        payload_segs = set(segments(r["payload"])) if r["payload"] else set()
        rows += [{"text": s, "label": 1} for s in payload_segs]
        others = [s for s in segments(r["text"]) if s not in payload_segs]
        rows += [{"text": s, "label": 0} for s in rng.sample(others, min(3, len(others)))]
    return pl.DataFrame(rows).unique("text", keep="first", maintain_order=True)


# -- candidate 1: embeddings + logistic regression -----------------------------------------


class EmbedLR:
    name = "embed_lr"

    def __init__(self, c: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression

        from sentinel.copilot.rag import load_embedder

        self._embed = load_embedder()
        self._lr = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")

    def _x(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embed.encode(texts, batch_size=64, normalize_embeddings=True))

    def fit(self, texts: list[str], y: list[int]) -> EmbedLR:
        self._lr.fit(self._x(texts), y)
        return self

    def __call__(self, texts: Any) -> list[float]:
        return [float(p) for p in self._lr.predict_proba(self._x(list(texts)))[:, 1]]

    def save(self, out: Path) -> None:
        import joblib

        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._lr, out / "embed_lr.joblib")

    @classmethod
    def load(cls, path: Path) -> EmbedLR:
        import joblib

        obj = cls()
        obj._lr = joblib.load(path / "embed_lr.joblib")
        return obj


# -- candidate 2: fine-tuned DeBERTa-v3-small ------------------------------------------------


class DebertaGuard:
    name = "deberta"

    def __init__(self, model: Any, tokenizer: Any, max_len: int = 128) -> None:
        self.model, self.tokenizer, self.max_len = model, tokenizer, max_len

    @classmethod
    def load(cls, path: Path) -> DebertaGuard:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32)
        return cls(model.eval(), AutoTokenizer.from_pretrained(path))

    @classmethod
    def train(
        cls,
        texts: list[str],
        y: list[int],
        epochs: int,
        lr: float,
        batch_size: int,
        max_len: int,
        seed: int,
    ) -> DebertaGuard:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(seed)
        tok = AutoTokenizer.from_pretrained(DEBERTA)
        model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA, num_labels=2, dtype=torch.float32
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        n = len(texts)
        steps = epochs * ((n + batch_size - 1) // batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=steps, pct_start=0.1
        )
        pos = sum(y) / n
        weight = torch.tensor([0.5 / (1 - pos), 0.5 / pos], dtype=torch.float32)
        rng = np.random.default_rng(seed)
        model.train()
        step = 0
        for epoch in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                enc = tok(
                    [texts[i] for i in idx],
                    truncation=True,
                    max_length=max_len,
                    padding=True,
                    return_tensors="pt",
                )
                logits = model(**enc).logits
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([y[i] for i in idx]), weight=weight
                )
                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += float(loss) * len(idx)
                step += 1
                if step % 100 == 0:
                    log.info("guard deberta step %d/%d loss %.4f", step, steps, float(loss))
            log.info("guard deberta epoch %d mean loss %.4f", epoch, total / n)
        return cls(model.eval(), tok, max_len)

    @torch.inference_mode()
    def __call__(self, texts: Any, batch_size: int = 32) -> list[float]:
        texts = list(texts)
        out: list[float] = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=self.max_len,
                padding=True,
                return_tensors="pt",
            )
            out += torch.softmax(self.model(**enc).logits, -1)[:, 1].tolist()
        return out

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)


# -- evaluation ------------------------------------------------------------------------------


def _doc_scores(clf: Any, texts: list[str]) -> list[float]:
    """Max classifier score over a document's segments (+ whole text when short)."""
    g = Guard(clf, threshold=2.0, use_rules=False)  # threshold > 1: collect, never flag
    out = []
    for t in texts:
        segs = segments(t)
        pieces = [*segs, t] if len(segs) > 1 and len(t) <= 600 else segs
        out.append(max((x or 0.0 for x in g._scores(pieces)), default=0.0))
    return out


def choose_threshold(
    scores: np.ndarray, y: np.ndarray, rule_flag: np.ndarray, max_fpr: float
) -> float:
    """Lowest threshold whose combined (rules OR classifier) false-alarm rate <= max_fpr."""
    neg = y == 0
    for t in np.unique(np.concatenate([scores, [1.0]])):
        fpr = float(((scores[neg] >= t) | rule_flag[neg]).mean())
        if fpr <= max_fpr:
            return float(t)
    return 1.0


def evaluate(docs: pl.DataFrame, clf: Any, threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    texts, y = docs["text"].to_list(), np.array(docs["label"].to_list())
    rules = np.array(Guard(None).flags(texts))
    clf_only = np.array(_doc_scores(clf, texts)) >= threshold if clf else np.zeros(len(y), bool)
    both = rules | clf_only
    sources = docs["source"].to_numpy()
    for name, flag in (("rules", rules), ("classifier", clf_only), ("rules+classifier", both)):
        row: dict[str, Any] = {}
        for src in ["all", *sorted(set(sources))]:
            m = np.ones(len(y), bool) if src == "all" else sources == src
            pos, neg = m & (y == 1), m & (y == 0)
            tp, fp = int((flag & pos).sum()), int((flag & neg).sum())
            row[src] = {
                "recall": tp / max(int(pos.sum()), 1),
                "fpr": fp / max(int(neg.sum()), 1),
                "precision": tp / max(tp + fp, 1),
                "n_pos": int(pos.sum()),
                "n_neg": int(neg.sum()),
            }
        out[name] = row
    return out


def run(params: Params, reports: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    g = params.copilot.guard
    reports = reports or params.resolve(Path("reports/copilot"))
    out_dir = out_dir or params.resolve(Path("models/copilot/guard"))
    deepset = params.resolve(g.deepset_dir)
    if not list((deepset / "data").glob("*.parquet")):
        guard_data.download_deepset(deepset)
    emails = pl.read_parquet(params.resolve(params.phishing.processed_dir) / "emails.parquet")
    docs = guard_data.build(
        emails.select("text", "split"),
        params.resolve(g.deepset_dir),
        params.seed,
        {"train": g.n_train, "val": g.n_val, "test": g.n_test},
    )
    train = training_segments(docs.filter(pl.col("split") == "train"), params.seed)
    val, test = docs.filter(pl.col("split") == "val"), docs.filter(pl.col("split") == "test")
    log.info(
        "guard data: %d train segments, %d val docs, %d test docs", len(train), len(val), len(test)
    )
    X, y = train["text"].to_list(), train["label"].to_list()

    candidates: dict[str, Any] = {}
    t0 = time.perf_counter()
    candidates["embed_lr"] = EmbedLR().fit(X, y)
    secs = {"embed_lr": time.perf_counter() - t0}
    if g.train_deberta:
        t0 = time.perf_counter()
        candidates["deberta"] = DebertaGuard.train(
            X, y, g.epochs, g.lr, g.batch_size, g.max_len, params.seed
        )
        secs["deberta"] = time.perf_counter() - t0

    yv = np.array(val["label"].to_list())
    rules_v = np.array(Guard(None).flags(val["text"].to_list()))
    selection: dict[str, Any] = {}
    for name, clf in candidates.items():
        sv = np.array(_doc_scores(clf, val["text"].to_list()))
        thr = choose_threshold(sv, yv, rules_v, g.max_fpr)
        flag = rules_v | (sv >= thr)
        selection[name] = {
            "threshold": thr,
            "val_recall": float(flag[yv == 1].mean()),
            "val_fpr": float(flag[yv == 0].mean()),
            "train_seconds": secs[name],
        }
        log.info("guard %s: %s", name, selection[name])
    best = max(selection, key=lambda n: (selection[n]["val_recall"], -selection[n]["val_fpr"]))
    clf, thr = candidates[best], selection[best]["threshold"]

    t0 = time.perf_counter()
    test_metrics = evaluate(test, clf, thr)
    per_doc_ms = 1000 * (time.perf_counter() - t0) / len(test) / 2
    clf.save(out_dir)
    (out_dir / "guard.json").write_text(
        json.dumps({"classifier": best, "threshold": thr, "max_fpr": g.max_fpr}, indent=2)
    )
    results = {
        "data": {
            "train_segments": len(train),
            "val_docs": len(val),
            "test_docs": len(test),
            "test_by_source": test.group_by("source", "label").len().sort("source", "label").rows(),
        },
        "selection": selection,
        "chosen": best,
        "test": test_metrics,
        "ms_per_document": per_doc_ms,
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "guard_results.json").write_text(json.dumps(results, indent=2))
    from sentinel.copilot.report import write_markdown

    write_markdown(reports)
    return results
