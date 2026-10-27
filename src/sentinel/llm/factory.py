"""Build the configured provider chain: cache -> (throttled Gemini) -> Ollama."""

from __future__ import annotations

from sentinel.common.config import CopilotLLMParams, Params, Settings, get_settings
from sentinel.common.logging import get_logger
from sentinel.llm.base import LLMProvider
from sentinel.llm.cache import CachedProvider, FallbackProvider, ThrottledProvider
from sentinel.llm.gemini import GeminiProvider
from sentinel.llm.ollama import OllamaProvider

log = get_logger(__name__)


def _one(
    name: str, cfg: CopilotLLMParams, params: Params, settings: Settings, interactive: bool
) -> LLMProvider:
    path = params.resolve(cfg.cache_path)
    if name == "gemini":
        key = settings.gemini_api_key or ""
        if interactive:
            gem = GeminiProvider(
                key,
                cfg.gemini_model,
                timeout=cfg.interactive_timeout_s,
                retries=cfg.interactive_retries,
                backoff=2.0,
                max_quota_wait=cfg.interactive_max_wait_s,
            )
            throttled = ThrottledProvider(
                gem, path, cfg.gemini_rpm, cfg.gemini_rpd, max_wait=cfg.interactive_max_wait_s
            )
        else:
            gem = GeminiProvider(key, cfg.gemini_model)
            throttled = ThrottledProvider(gem, path, cfg.gemini_rpm, cfg.gemini_rpd)
        return CachedProvider(throttled, path)
    if name == "ollama":
        return CachedProvider(OllamaProvider(cfg.ollama_model, settings.ollama_url), path)
    raise ValueError(f"unknown LLM provider {name!r} (gemini or ollama)")


def build_provider(
    params: Params,
    settings: Settings | None = None,
    order: list[str] | None = None,
    interactive: bool = False,
) -> LLMProvider:
    """Providers in `order` (default: LLM_PROVIDERS env, else params); skips unusable ones.

    Each provider gets its own cache entry, so switching providers never serves one
    model's answer as another's. `interactive` (the API) gives up on Gemini sooner.
    """
    settings = settings or get_settings()
    cfg = params.copilot.llm
    names = order or (
        settings.llm_providers.split(",") if settings.llm_providers else cfg.providers
    )
    chain: list[LLMProvider] = []
    for name in (n.strip() for n in names if n.strip()):
        if name == "gemini" and not settings.gemini_api_key:
            log.warning("GEMINI_API_KEY not set; skipping Gemini")
            continue
        chain.append(_one(name, cfg, params, settings, interactive))
    return chain[0] if len(chain) == 1 else FallbackProvider(chain)


def build_judge(
    params: Params, settings: Settings | None = None, order: list[str] | None = None
) -> LLMProvider:
    """The faithfulness judge: Gemini's judge model if a key is set, else the local model.

    With Ollama as both generator and judge, the judge differs by prompt only (the design
    allows "a different model or prompt"); the evaluation report says which was used.
    """
    settings = settings or get_settings()
    cfg = params.copilot.llm
    path = params.resolve(cfg.cache_path)
    chain: list[LLMProvider] = []
    for name in order or ["gemini", "ollama"]:
        if name == "gemini" and settings.gemini_api_key:
            gem = GeminiProvider(settings.gemini_api_key, cfg.judge_model)
            chain.append(
                CachedProvider(ThrottledProvider(gem, path, cfg.judge_rpm, cfg.judge_rpd), path)
            )
        elif name == "ollama":
            chain.append(
                CachedProvider(OllamaProvider(cfg.ollama_model, settings.ollama_url), path)
            )
    if not chain:
        raise ValueError("no judge provider available")
    return chain[0] if len(chain) == 1 else FallbackProvider(chain)
