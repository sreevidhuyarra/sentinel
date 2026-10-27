"""Blocklist and domain-age checks with a fake fetcher (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sentinel.phishing.intel import FEEDS, RDAP_URL, Intel, norm_url, sent_at
from sentinel.phishing.sender import parse_sender

HOSTS = "# comment\nevil-login.com\nsites.google.com\nphish.example.org\n"
URLS = "http://himorhis.square.site/\nhttps://bad.example.net/pay/now\n"
SENT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _rdap(registered: str) -> bytes:
    return json.dumps(
        {"events": [{"eventAction": "registration", "eventDate": f"{registered}T00:00:00Z"}]}
    ).encode()


class FakeFetch:
    def __init__(self, ages: dict[str, str] | None = None) -> None:
        self.ages = ages or {}
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.calls.append(url)
        if url == FEEDS["Phishing.Database"][0]:
            return HOSTS.encode()
        if url == FEEDS["OpenPhish"][0]:
            return URLS.encode()
        for domain, reg in self.ages.items():
            if url == RDAP_URL.format(domain):
                return _rdap(reg)
        raise FileNotFoundError(url)


@pytest.fixture()
def intel(tmp_path: Path) -> tuple[Intel, FakeFetch]:
    fetch = FakeFetch({"fresh-invoice.com": "2026-09-25", "oldshop.com": "2004-03-01"})
    i = Intel(tmp_path, fetch=fetch)
    i.refresh_feeds()
    return i, fetch


def test_listed_hosts_skip_shared_platforms(intel: tuple[Intel, FakeFetch]) -> None:
    i, _ = intel
    assert i.host_listed("evil-login.com") == ["Phishing.Database"]
    assert i.host_listed("mail.evil-login.com") == ["Phishing.Database"]  # its registered domain
    assert i.host_listed("sites.google.com") == []  # listed, but a shared platform
    assert i.host_listed("himorhis.square.site") == ["OpenPhish"]
    assert i.host_listed("square.site") == []
    assert i.url_listed("HTTPS://BAD.example.net/pay/now/") == ["OpenPhish"]
    assert norm_url("bad.example.net/x/") == "http://bad.example.net/x"


def test_check_reports_sender_link_and_new_domain(intel: tuple[Intel, FakeFetch]) -> None:
    i, _ = intel
    who = parse_sender("Billing <ap@fresh-invoice.com>", "refunds@evil-login.com")
    rep = i.check(who, ["https://bad.example.net/pay/now", "https://oldshop.com/a"], SENT)
    checks = sorted(f.check for f in rep.findings)
    assert checks == ["listed_domain", "listed_domain", "listed_url", "new_domain"]
    new = next(f for f in rep.findings if f.check == "new_domain")
    assert "fresh-invoice.com was registered 6 days before the email was sent" in new.text
    by = {d.domain: d for d in rep.domains}
    assert by["evil-login.com"].seen_as == ["reply-to"]
    assert by["oldshop.com"].age_days is not None and by["oldshop.com"].age_days > 8000


def test_well_known_domains_are_never_sent_to_rdap(intel: tuple[Intel, FakeFetch]) -> None:
    i, fetch = intel
    i.check(parse_sender("x <a@gmail.com>"), ["https://www.paypal.com/x", "http://10.0.0.1/"])
    assert not any("rdap" in c for c in fetch.calls)


def test_rdap_answers_are_cached(tmp_path: Path) -> None:
    fetch = FakeFetch({"fresh-invoice.com": "2026-09-25"})
    i = Intel(tmp_path, fetch=fetch)
    assert i.registered_on("fresh-invoice.com") == "2026-09-25"
    assert i.registered_on("unknown-tld.zz") is None  # lookup failed: cached as a miss
    n = len(fetch.calls)
    again = Intel(tmp_path, fetch=fetch)  # a new process reads the same cache file
    assert again.registered_on("fresh-invoice.com") == "2026-09-25"
    assert again.registered_on("unknown-tld.zz") is None
    assert len(fetch.calls) == n


def test_domain_registered_after_the_email_is_not_new(tmp_path: Path) -> None:
    i = Intel(tmp_path, fetch=FakeFetch({"reused.com": "2026-10-20"}))
    rep = i.check(parse_sender("a <x@reused.com>"), [], SENT)
    assert rep.findings == [] and rep.domains[0].age_days is None


def test_failed_download_keeps_the_old_feed(intel: tuple[Intel, FakeFetch], tmp_path: Path) -> None:
    def offline(url: str, timeout: float) -> bytes:
        raise OSError("offline")

    i = Intel(tmp_path, fetch=offline)
    i.refresh_feeds(force=True)
    assert i.feeds_loaded and i.host_listed("evil-login.com")


def test_sent_at_parses_rfc2822_dates() -> None:
    assert sent_at("Thu, 1 Oct 2026 12:00:00 +0000") == SENT
    assert sent_at("not a date") is None and sent_at(None) is None
    assert sent_at("Thu, 1 Oct 2026 14:00:00 +0200") == SENT  # same instant
    assert sent_at("1 Oct 2026 12:00:00") is not None
