"""Email text normalisation shared by data preparation, training and serving.

The model sees `subject + body` with HTML removed and every URL replaced by the token
[URL]; URLs are extracted separately for the URL model. Serving must call the same
functions, so the text the model is scored on matches what it was trained on.
"""

from __future__ import annotations

import html
import re

URL_TOKEN = "[URL]"

_URL = re.compile(r"""(?:https?://|www\.)[^\s<>"'()\[\]{}]+""", re.IGNORECASE)
_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r"""href\s*=\s*["']?([^"'\s>]+)""", re.IGNORECASE)
_TAG = re.compile(r"<[^>]{1,2000}>", re.DOTALL)
_SPACE = re.compile(r"[ \t\r\f\v\u00a0\u200b]+")  # incl. &nbsp; and zero-width space
_BLANK_LINES = re.compile(r"\n\s*\n+")
_EMAIL_DOMAIN = re.compile(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_TRAILING_PUNCT = ".,;:!?'\")]}>"

# Second-level labels under which registrations happen (example.co.uk -> example.co.uk).
_SECOND_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or"}


def extract_urls(text: str) -> list[str]:
    """URLs from plain text and from HTML href attributes, in order, without duplicates."""
    found = [
        m.rstrip(_TRAILING_PUNCT)
        for m in _HREF.findall(text)
        if m.lower().startswith(("http", "www"))
    ]
    found += [m.group(0).rstrip(_TRAILING_PUNCT) for m in _URL.finditer(text)]
    seen: set[str] = set()
    out = []
    for u in found:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def strip_html(text: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", text)
    text = re.sub(r"<\s*(br|/p|/div|/tr|/li)\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    return html.unescape(_TAG.sub(" ", text))


def clean_text(text: str | None) -> str:
    """HTML removed, URLs replaced by [URL], whitespace collapsed (newlines kept)."""
    if not text:
        return ""
    text = strip_html(text)
    text = _URL.sub(URL_TOKEN, text)
    text = _SPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def model_input(subject: str | None, body: str | None, max_chars: int = 3000) -> str:
    """The exact string the classifier is trained and served on."""
    s, b = clean_text(subject), clean_text(body)
    return f"{s}\n\n{b[:max_chars]}".strip()


def dedup_key(text: str) -> str:
    """Case- and whitespace-insensitive form used to find duplicate emails."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def sender_domain(sender: str | None) -> str | None:
    """Registered domain of the From address ('Bob <bob@mail.paypal.com>' -> 'paypal.com')."""
    if not sender:
        return None
    m = _EMAIL_DOMAIN.search(sender)
    if not m:
        return None
    parts = m.group(1).lower().strip(".").split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])
