"""Payment and invoice-fraud wording (business email compromise).

An invoice scam reads like ordinary business mail, so the text model passes it. What gives it
away is the request (send money, to new bank details, urgently, or in gift cards) together
with who is asking. These checks find the request; `escalate` pairs it with a sender or
domain finding.

The wording checks only mark an email for review. The legitimate corpora hold few genuine
invoices or bank-detail notices (their false-alarm rate there is near zero), while a real
accounts-payable inbox receives them daily, so that rate would not carry over. A payment
request from a risky sender is rarer in both, and its role is measured
(sentinel.phishing.payment_eval).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from sentinel.phishing.sender import Finding, Severity

# Each check: pattern over subject + body, severity, and the analyst-facing reason.
_CHECKS: dict[str, tuple[re.Pattern[str], Severity, str]] = {
    "payment_change": (
        re.compile(
            r"\b(?:new|updated?|changed?|different|alternate)\s+(?:bank|banking|remittance|"
            r"payment|wire)\s+(?:details|information|info|instructions|account)"
            r"|\b(?:change|update)\s+(?:of|to|in|our|the|my)\s+(?:our\s+|the\s+|my\s+)?"
            r"(?:bank|banking|remittance)\s+(?:details|information|account)",
            re.I,
        ),
        "high",
        "asks to pay to new or changed bank details",
    ),
    "gift_card_request": (
        re.compile(r"\b(?:buy|purchase|get|pick up)\s+(?:\w+\s+){0,4}gift\s*cards?", re.I),
        "high",
        "asks for gift cards to be bought",
    ),
    "wire_request": (
        re.compile(
            r"\b(?:process|make|send|arrange|initiate|complete)\s+(?:a\s+|the\s+|an\s+|this\s+)?"
            r"(?:wire|bank)\s+transfer",
            re.I,
        ),
        "medium",
        "asks for a wire or bank transfer",
    ),
    "payment_urgency": (
        re.compile(
            r"\b(?:overdue|outstanding|past[- ]due|unpaid)\s+(?:invoice|invoices|payment|balance)"
            r"|\b(?:payment|transfer|remittance|pay\s+(?:the|this|it|now))\b[^.\n]{0,60}?"
            r"\b(?:today|immediately|asap|urgent(?:ly)?|right away|by end of (?:the )?day|"
            r"within 24 hours)\b",
            re.I,
        ),
        "medium",
        "presses for a payment (overdue or due immediately)",
    ),
}
PAYMENT_CHECKS = tuple(_CHECKS)
ESCALATED = "payment_from_risky_sender"
# Sender or domain findings that make a payment request worth an alert. no_address is too
# common on legitimate mailing-list mail to count.
NOT_RISKY = frozenset({"no_address"})


def check_payment(text: str) -> list[Finding]:
    """Payment-request findings in the subject and body (model_input text or raw)."""
    out = []
    for check, (pattern, severity, reason) in _CHECKS.items():
        m = pattern.search(text)
        if m:
            quote = " ".join(m.group(0).split())
            out.append(Finding(check, severity, f'{reason}: "{quote}"'))
    return out


def escalate(payment: list[Finding], other_checks: Iterable[str]) -> Finding | None:
    """One finding when a payment request comes with a sender or domain finding."""
    risky = [c for c in other_checks if c not in NOT_RISKY]
    if not payment or not risky:
        return None
    signs = ", ".join(c.replace("_", " ") for c in dict.fromkeys(risky))
    return Finding(
        ESCALATED,
        "high",
        f"a payment request from a sender with warning signs ({signs}): verify by phone, "
        "using a number you already have",
    )
