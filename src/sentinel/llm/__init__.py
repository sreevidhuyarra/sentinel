"""LLM access for the copilot: one interface, Gemini by default, Ollama offline."""

from sentinel.llm.base import LLMError, LLMProvider, LLMResponse, QuotaError, parse_json

__all__ = ["LLMError", "LLMProvider", "LLMResponse", "QuotaError", "parse_json"]
