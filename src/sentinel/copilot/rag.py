"""Hybrid retrieval over ATT&CK techniques and KEV CVEs (report section 8.1, map_attack).

BM25 and dense embeddings (FAISS inner product on normalised vectors) each rank the
corpus; reciprocal-rank fusion merges them and a cross-encoder re-ranks the head. The LLM
later chooses techniques only from what this returns, so it cannot cite an ID that does
not exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from sentinel.common.logging import get_logger

log = get_logger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# bge-small expects this prefix on queries (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "be",
        "may",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "at",
        "into",
        "via",
        "use",
        "used",
        "using",
        "such",
        "other",
        "than",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def technique_text(r: dict[str, Any]) -> str:
    return (
        f"{r['id']} {r['full_name']}. Tactics: {', '.join(r['tactics'])}. {r['description']} "
        f"Detection: {r['detection'][:600]}"
    )


def cve_text(r: dict[str, Any]) -> str:
    return f"{r['id']} {r['name']}. {r['vendor']} {r['product']}. {r['description']}"


@dataclass
class Hit:
    kind: str  # "technique" or "cve"
    id: str
    name: str
    score: float  # cross-encoder score (higher is better)
    record: dict[str, Any]


class Retriever:
    def __init__(
        self,
        docs: pl.DataFrame,
        embeddings: np.ndarray,
        embed_model: Any = None,
        rerank_model: Any = None,
    ) -> None:
        import faiss
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.records = docs.to_dicts()
        self.texts = docs["text"].to_list()
        self.kinds = np.array(docs["kind"].to_list())
        self._bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings.astype(np.float32))
        self._embed, self._rerank = embed_model, rerank_model

    # -- building ---------------------------------------------------------------------------

    @staticmethod
    def build_docs(knowledge_dir: Path) -> pl.DataFrame:
        tech = pl.read_parquet(knowledge_dir / "techniques.parquet")
        cves = pl.read_parquet(knowledge_dir / "cves.parquet")
        rows = [
            {
                "kind": "technique",
                "id": r["id"],
                "name": r["full_name"],
                "text": technique_text(r),
                "record": json.dumps(r),
            }
            for r in tech.to_dicts()
        ] + [
            {
                "kind": "cve",
                "id": r["id"],
                "name": r["name"],
                "text": cve_text(r),
                "record": json.dumps(r),
            }
            for r in cves.to_dicts()
        ]
        return pl.DataFrame(rows)

    @classmethod
    def build(cls, knowledge_dir: Path, out_dir: Path) -> Retriever:
        docs = cls.build_docs(knowledge_dir)
        embed = load_embedder()
        emb = embed.encode(
            docs["text"].to_list(),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        docs.write_parquet(out_dir / "docs.parquet")
        np.save(out_dir / "embeddings.npy", emb.astype(np.float32))
        (out_dir / "meta.json").write_text(
            json.dumps({"embed_model": EMBED_MODEL, "rerank_model": RERANK_MODEL, "n": len(docs)})
        )
        log.info("retrieval index: %d documents, dim %d", len(docs), emb.shape[1])
        return cls(docs, emb, embed)

    @classmethod
    def load(cls, out_dir: Path, rerank: bool = True) -> Retriever:
        docs = pl.read_parquet(out_dir / "docs.parquet")
        emb = np.load(out_dir / "embeddings.npy")
        return cls(docs, emb, load_embedder(), load_reranker() if rerank else None)

    # -- querying ---------------------------------------------------------------------------

    def _rank_bm25(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)))
        scores[~mask] = -np.inf
        top = np.argsort(-scores)[:n]
        return [int(i) for i in top if np.isfinite(scores[i]) and scores[i] > 0]

    def _rank_dense(self, query: str, mask: np.ndarray, n: int) -> list[int]:
        if self._embed is None:
            return []
        q = self._embed.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False
        )
        _, idx = self._index.search(q.astype(np.float32), min(len(self.texts), n * 4))
        return [int(i) for i in idx[0] if i >= 0 and mask[i]][:n]

    def search(
        self,
        query: str,
        kind: str | None = "technique",
        k: int = 5,
        pool: int = 30,
        rerank_top: int = 20,
    ) -> list[Hit]:
        """Hybrid top-k: `pool` candidates per ranker are fused; the top `rerank_top` re-ranked."""
        mask = np.ones(len(self.texts), bool) if kind is None else self.kinds == kind
        fused: dict[int, float] = {}
        for ranking in (self._rank_bm25(query, mask, pool), self._rank_dense(query, mask, pool)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        head = sorted(fused, key=lambda i: -fused[i])[: rerank_top if self._rerank else pool]
        if not head:
            return []
        if self._rerank is not None:
            scores = np.asarray(
                self._rerank.predict(
                    [(query, self.texts[i]) for i in head], show_progress_bar=False
                )
            )
        else:
            scores = np.array([fused[i] for i in head])
        order = np.argsort(-scores)[:k]
        return [
            Hit(
                kind=self.records[head[j]]["kind"],
                id=self.records[head[j]]["id"],
                name=self.records[head[j]]["name"],
                score=float(scores[j]),
                record=json.loads(self.records[head[j]]["record"]),
            )
            for j in order
        ]

    def get(self, doc_id: str) -> dict[str, Any] | None:
        for r in self.records:
            if r["id"] == doc_id:
                return dict(json.loads(r["record"]))
        return None


def _cached_first(factory: Any, name: str, **kwargs: Any) -> Any:
    """Load from the local Hugging Face cache; go online only on the first run."""
    try:
        return factory(name, device="cpu", local_files_only=True, **kwargs)
    except OSError:
        return factory(name, device="cpu", **kwargs)


def load_embedder() -> Any:
    from sentence_transformers import SentenceTransformer

    return _cached_first(SentenceTransformer, EMBED_MODEL)


def load_reranker() -> Any:
    from sentence_transformers import CrossEncoder

    return _cached_first(CrossEncoder, RERANK_MODEL, max_length=256)
