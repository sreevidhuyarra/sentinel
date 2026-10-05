"""Assemble a Copilot from configuration (API, CLI and evaluation share this)."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.common.config import Params, Settings, get_settings
from sentinel.common.logging import get_logger
from sentinel.copilot.graph import Copilot
from sentinel.copilot.guard import Classifier, Guard
from sentinel.copilot.rag import Retriever
from sentinel.copilot.tools import AlertTools
from sentinel.db.schema import make_engine
from sentinel.llm.base import LLMProvider
from sentinel.llm.factory import build_provider

log = get_logger(__name__)

RAG_DIR = Path("models/copilot/rag")
GUARD_DIR = Path("models/copilot/guard")


def load_guard(path: Path) -> Guard:
    """The trained guard (rules + chosen classifier), or rules only if none was trained."""
    cfg_path = path / "guard.json"
    if not cfg_path.exists():
        log.warning("no trained guard classifier at %s; using rules only", path)
        return Guard(None)
    cfg = json.loads(cfg_path.read_text())
    clf: Classifier
    if cfg["classifier"] == "deberta":
        from sentinel.copilot.guard_train import DebertaGuard

        clf = DebertaGuard.load(path)
    else:
        from sentinel.copilot.guard_train import EmbedLR

        clf = EmbedLR.load(path)
    # Guards trained before pair scoring have a threshold chosen without it.
    return Guard(clf, threshold=float(cfg["threshold"]), pairs=bool(cfg.get("pairs", False)))


def build_copilot(
    params: Params,
    settings: Settings | None = None,
    llm: LLMProvider | str | None = "config",
    read_only: bool = True,
    db_url: str | None = None,
    **kwargs: bool,
) -> Copilot:
    settings = settings or get_settings()
    url = db_url or (settings.postgres_ro_url if read_only else settings.postgres_url)
    provider: LLMProvider | None
    if llm == "config":
        try:
            provider = build_provider(params, settings)
        except ValueError as exc:
            log.warning("no LLM provider (%s); reports will use the template", exc)
            provider = None
    else:
        provider = llm if not isinstance(llm, str) else None
    return Copilot(
        tools=AlertTools(make_engine(url)),
        retriever=Retriever.load(params.resolve(RAG_DIR)),
        llm=provider,
        guard=load_guard(params.resolve(GUARD_DIR)),
        cfg=params.copilot,
        **kwargs,
    )
