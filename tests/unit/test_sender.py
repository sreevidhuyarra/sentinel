"""Sender checks: what they flag, what they must not flag, and how the policy is chosen."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.phishing.sender import (
    _one_edit,
    check_sender,
    parse_auth,
    parse_eml,
    parse_sender,
    role_of,
)
from sentinel.phishing.sender_eval import choose_policy


def checks(from_: str, reply_to: str = "", auth: str = "") -> set[str]:
    return {f.check for f in check_sender(parse_sender(from_, reply_to, auth))}


@pytest.mark.parametrize(
    ("from_", "expected"),
    [
        ("PayPal <service@paypal.com>", set()),
        ("PayPal <service@paypal.de>", set()),  # the brand's own name on another TLD
        ("Amazon.com <store-news@amazon.com>", set()),
        ("Apple iCards <erb@mac.com>", set()),
        ("Microsoft <account-security-noreply@accountprotection.microsoft.com>", set()),
        ("Scott Norton <nortonsm@yahoo.com>", set()),  # a surname, not the brand
        ('"otg@heise.de (via RT)" <bugs@perl.org>', set()),  # a relay, not spoofing
        ("Forbes <forbesdaily+403097.1675479@news.forbesdigital.com>", set()),
        ("PayPal <service@paypa1.com>", {"brand_impersonation", "lookalike_domain"}),
        ("Norton Support <help@n0rton-secure.com>", {"brand_impersonation", "lookalike_domain"}),
        ("Chase <alerts@chase-secure-login.com>", {"brand_impersonation", "brand_in_domain"}),
        ('"service@paypal.com" <a@evil.ru>', {"brand_impersonation", "address_in_name"}),
        ("Microsoft Account Team <msacct@gmail.com>", {"brand_impersonation", "freemail_role"}),
        ("Billing Department <payments.dept@gmail.com>", {"freemail_role"}),
        ("x <a@login.paypal.com.secure-check.ru>", {"brand_in_domain"}),
        ("x <a@xn--pypal-4ve.com>", {"punycode_domain"}),
        ("Alice <alice@cheap-deals.top>", {"suspicious_tld"}),
        ("undisclosed", {"no_address"}),
    ],
)
def test_address_checks(from_: str, expected: set[str]) -> None:
    assert checks(from_) == expected


def test_header_checks_need_reply_to_or_auth_results() -> None:
    assert (
        checks("Bob <bob@acme.com>", "Bob <bob@acme.com>", "spf=pass dkim=pass dmarc=pass") == set()
    )
    found = checks("Bob <bob@acme.com>", "bob.acme@gmail.com", "spf=softfail dkim=fail dmarc=fail")
    assert found == {"reply_to_elsewhere", "spf_fail", "dkim_fail", "dmarc_fail"}
    # Header checks still run when the From line has no usable address.
    assert checks("undisclosed", "", "dmarc=fail") == {"no_address", "dmarc_fail"}


def test_parse_auth_and_one_edit() -> None:
    assert parse_auth("mx.example; spf=pass smtp.mailfrom=a.com; dkim=none; dmarc=fail") == {
        "spf": "pass",
        "dkim": "none",
        "dmarc": "fail",
    }
    assert _one_edit("paypal", "paypa") and _one_edit("paypal", "papyal")
    assert not _one_edit("paypal", "paypal") and not _one_edit("paypal", "pyapla")


def test_unmeasured_header_checks_never_alert() -> None:
    assert role_of("dmarc_fail", {"dmarc_fail": "alert"}) == "review"
    assert role_of("dmarc_fail", {"dmarc_fail": "ignore"}) == "ignore"
    assert role_of("brand_impersonation", {"brand_impersonation": "alert"}) == "alert"
    assert role_of("brand_impersonation", {}) == "review"  # no policy file: review only


EML = (
    b"Received: from mail.attacker.example\r\n"
    b"Authentication-Results: mx.recipient.example; spf=fail smtp.mailfrom=paypal.com;"
    b" dkim=none; dmarc=fail header.from=paypal.com\r\n"
    b"Authentication-Results: forged.example; spf=pass; dmarc=pass\r\n"
    b"From: =?utf-8?q?PayPal?= <service@paypal.com>\r\n"
    b"Reply-To: refunds.paypal@gmail.com\r\n"
    b"Subject: Your account is limited\r\n"
    b"Date: Tue, 01 Oct 2024 10:00:00 +0000\r\n"
    b"MIME-Version: 1.0\r\n"
    b'Content-Type: multipart/mixed; boundary="b"\r\n\r\n'
    b"--b\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n"
    b"Please confirm your details at http://paypal-check.example/login\r\n"
    b"--b\r\nContent-Type: application/octet-stream; name=x.exe\r\n"
    b"Content-Transfer-Encoding: base64\r\n\r\nTVqQAAMAAAAEAAAA\r\n--b--\r\n"
)


def test_parse_eml_reads_headers_and_text_but_not_attachments() -> None:
    e = parse_eml(EML)
    assert e.subject == "Your account is limited"
    assert "confirm your details" in e.body and "TVqQ" not in e.body
    s = e.sender
    assert s.address == "service@paypal.com" and s.display_name == "PayPal"
    assert s.reply_to == "refunds.paypal@gmail.com"
    # Only the topmost Authentication-Results (the recipient's own server) is trusted.
    assert s.auth == {"spf": "fail", "dkim": "none", "dmarc": "fail"}
    found = {f.check for f in check_sender(s)}
    assert found == {"reply_to_elsewhere", "spf_fail", "dmarc_fail"}  # From line itself is fine


def test_policy_from_legitimate_rates() -> None:
    n = 10_000
    found: list[set[str]] = [set() for _ in range(n)]
    for i in range(5):  # 0.05%: may alert
        found[i].add("brand_impersonation")
    for i in range(100):  # 1%: review only
        found[i].add("freemail_role")
    for i in range(500):  # 5%: ignored
        found[i].add("random_address")
    for i in range(3):
        found[i].add("dmarc_fail")
    policy, rate = choose_policy(found, np.zeros(n, bool))
    assert policy["brand_impersonation"] == "alert" and rate["brand_impersonation"] == 0.0005
    assert policy["freemail_role"] == "review" and policy["random_address"] == "ignore"
    assert policy["dmarc_fail"] == "review" and policy["punycode_domain"] == "alert"
    # The combined cap: checks that each pass alone stop being alerts once together they
    # would add more than 0.5% false alarms.
    many = [{f"c{i % 8}"} if i < 80 else set() for i in range(n)]
    import sentinel.phishing.sender_eval as se

    old = se.CHECKS
    se.CHECKS = tuple(f"c{i}" for i in range(8))
    try:
        capped, _ = choose_policy(many, np.zeros(n, bool), combined_max=0.005)
    finally:
        se.CHECKS = old
    assert sum(r == "alert" for r in capped.values()) == 5  # 5 x 0.1% = 0.5%
