"""Payment and invoice-fraud wording checks."""

from __future__ import annotations

import pytest

from sentinel.phishing.payment import ESCALATED, check_payment, escalate


@pytest.mark.parametrize(
    ("text", "check"),
    [
        ("Please note our new bank details for all future payments.", "payment_change"),
        ("We have updated banking information, see below.", "payment_change"),
        ("Following a change of our bank account, use the IBAN below.", "payment_change"),
        ("I need you to buy six Apple gift cards for a client today.", "gift_card_request"),
        ("Can you process a wire transfer before noon?", "wire_request"),
        ("Please find attached the overdue invoice.", "payment_urgency"),
        ("Complete the payment today to avoid a late fee.", "payment_urgency"),
        ("The transfer must go out immediately.", "payment_urgency"),
    ],
)
def test_payment_requests_are_found(text: str, check: str) -> None:
    assert check in {f.check for f in check_payment(text)}


@pytest.mark.parametrize(
    "text",
    [
        "Pay attention to the agenda for today's meeting.",
        "Business Wire press release: the company announced today a new plant.",
        "Thanks for the gift card at the party!",
        "Our bank is closed today for the holiday.",
        "The account details are in the shared drive.",
    ],
)
def test_ordinary_wording_is_not_a_payment_request(text: str) -> None:
    assert check_payment(text) == []


def test_finding_quotes_the_matched_words() -> None:
    [f] = check_payment("Kindly settle the OUTSTANDING   invoice.")
    assert f.text.endswith('"OUTSTANDING invoice"')


def test_escalation_needs_both_a_request_and_a_risky_sender() -> None:
    pay = check_payment("Please complete the payment today.")
    assert escalate(pay, []) is None
    assert escalate(pay, ["no_address"]) is None  # too common on legitimate mail
    assert escalate([], ["freemail_role"]) is None
    found = escalate(pay, ["freemail_role", "new_domain", "freemail_role"])
    assert found is not None and found.check == ESCALATED
    assert "(freemail role, new domain)" in found.text and "verify by phone" in found.text
