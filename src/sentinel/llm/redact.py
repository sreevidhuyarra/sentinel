"""Reversible redaction of identifying values before a prompt leaves the machine.

IP addresses, email addresses and phone numbers become stable placeholders such as
`[ip-internal-1]`, so the model can still tell hosts apart and reason about internal vs
external traffic; `restore` puts the real values back into the answer.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any

_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
)
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}(?![\w:])")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]?\d{3,4}(?!\w)")
_TOKEN = re.compile(r"\[(?:ip-(?:internal|external)|email|phone)-\d+\]")


def _ip_kind(value: str) -> str:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return "external"
    return "internal" if ip.is_private or ip.is_loopback or ip.is_link_local else "external"


class Redactor:
    """One mapping per investigation, so a host keeps its placeholder across prompts."""

    def __init__(self) -> None:
        self.forward: dict[str, str] = {}
        self.backward: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def _token(self, value: str, kind: str) -> str:
        if value not in self.forward:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            tok = f"[{kind}-{self._counts[kind]}]"
            self.forward[value], self.backward[tok] = tok, value
        return self.forward[value]

    def redact(self, text: str) -> str:
        text = _EMAIL.sub(lambda m: self._token(m.group(), "email"), text)
        text = _IPV4.sub(lambda m: self._token(m.group(), f"ip-{_ip_kind(m.group())}"), text)
        text = _IPV6.sub(lambda m: self._token(m.group(), f"ip-{_ip_kind(m.group())}"), text)
        return _PHONE.sub(lambda m: self._token(m.group(), "phone"), text)

    def restore(self, value: Any) -> Any:
        """Put real values back into a string or any JSON-like structure."""
        if isinstance(value, str):
            return _TOKEN.sub(lambda m: self.backward.get(m.group(), m.group()), value)
        if isinstance(value, list):
            return [self.restore(v) for v in value]
        if isinstance(value, dict):
            return {k: self.restore(v) for k, v in value.items()}
        return value
