"""The LLMProvider interface every copilot call goes through."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """A provider could not answer (network, bad response, refused)."""


class QuotaError(LLMError):
    """Rate or daily quota reached; the fallback chain moves on to the next provider.

    `daily` is True only when the provider says the per-day quota is spent, so a
    per-minute limit does not write off the rest of the day.
    """

    def __init__(self, message: str, daily: bool = False) -> None:
        super().__init__(message)
        self.daily = daily


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        return parse_json(self.text)


class LLMProvider(Protocol):
    name: str
    model: str
    local: bool  # True if prompts never leave the machine

    def generate(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """One completion. With `schema` (JSON Schema), the answer is a JSON document."""
        ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json(text: str) -> Any:
    """Parse a JSON answer, tolerating markdown fences and prose around the object."""
    s = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise LLMError(f"answer is not valid JSON: {text[:200]!r}") from None


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve `$ref`/`$defs` (Pydantic output) into one self-contained schema.

    Structured-output APIs accept only a subset of JSON Schema; a flat schema without
    references, titles or defaults is the most portable form.
    """
    defs = schema.get("$defs", {})
    drop = ("$defs", "title", "default")

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].split("/")[-1]])
        out = {}
        for k, v in node.items():
            if k in drop:
                continue
            # `properties` maps field names to schemas: a field may itself be called "title".
            out[k] = {name: walk(s) for name, s in v.items()} if k == "properties" else walk(v)
        return out

    flat: dict[str, Any] = walk(schema)
    return flat
