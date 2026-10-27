"""Threat intelligence for the domains in an email: phishing blocklists and domain age.

Blocklists are downloaded and looked up locally, so nothing about an email leaves the machine:
- Phishing.Database ACTIVE domains (github.com/Phishing-Database/Phishing.Database, MIT):
  hostnames currently serving phishing.
- OpenPhish community feed (openphish.com, free): the newest phishing URLs.
A host on a shared platform (sites.google.com, a link shortener) gets listed because of one
page among millions, so a host listing there is ignored; an exact URL listing still counts.

Domain age comes from RDAP (WHOIS's successor): the registered domain, and only the domain,
is sent to rdap.org, which redirects to the registry. Answers are cached for weeks, and
free-mail, brand and shared-platform domains are never looked up.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx

from sentinel.common.logging import get_logger
from sentinel.phishing.sender import BRANDS, FREEMAIL, Finding, Sender, registered_of

log = get_logger(__name__)

Fetch = Callable[[str, float], bytes]  # (url, timeout seconds) -> body; raises on failure

FeedKind = Literal["hosts", "urls"]
FEEDS: dict[str, tuple[str, FeedKind]] = {
    "Phishing.Database": (
        "https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/"
        "phishing-domains-ACTIVE.txt",
        "hosts",
    ),
    "OpenPhish": ("https://openphish.com/feed.txt", "urls"),
}
RDAP_URL = "https://rdap.org/domain/{}"
RDAP_FOUND_TTL = timedelta(days=30)
RDAP_MISS_TTL = timedelta(days=1)  # no answer: an outage, a rate limit, a ccTLD without RDAP

INTEL_CHECKS = ("listed_url", "listed_domain", "new_domain")

# Hosts serving many people's content under paths: a listing of the whole host is noise.
SHARED_HOSTS = frozenset(
    {
        "sites.google.com",
        "docs.google.com",
        "drive.google.com",
        "script.google.com",
        "forms.gle",
        "storage.googleapis.com",
        "firebasestorage.googleapis.com",
        "s3.amazonaws.com",
        "dropbox.com",
        "www.dropbox.com",
        "dl.dropboxusercontent.com",
        "onedrive.live.com",
        "1drv.ms",
        "forms.office.com",
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "t.me",
        "ipfs.io",
        "cloudflare-ipfs.com",
        "github.com",
        "raw.githubusercontent.com",
        "gitlab.com",
        "linkedin.com",
        "www.linkedin.com",
        "facebook.com",
        "www.facebook.com",
        "twitter.com",
        "x.com",
        "discord.com",
        "cdn.discordapp.com",
        "mediafire.com",
        "www.mediafire.com",
        "wetransfer.com",
        "we.tl",
        "notion.so",
        "www.notion.so",
        "canva.com",
        "www.canva.com",
        "linktr.ee",
        "jotform.com",
        "form.jotform.com",
        "typeform.com",
        "feeds.feedburner.com",
        "angelfire.com",
        "www.angelfire.com",
    }
)
# Registered domains never sent to RDAP: their age says nothing about a sender.
_WELL_KNOWN = (
    FREEMAIL
    | {d for domains in BRANDS.values() for d in domains}
    | {registered_of(h) for h in SHARED_HOSTS}
)


@dataclass
class DomainInfo:
    domain: str  # registered domain
    seen_as: list[str]  # "sender", "reply-to", "link"
    registered: str | None = None  # ISO date from RDAP
    age_days: int | None = None  # at the email's Date (or now)
    listed_on: list[str] = field(default_factory=list)


@dataclass
class IntelReport:
    findings: list[Finding]
    domains: list[DomainInfo]
    feeds_loaded: bool


def _http_fetch(url: str, timeout: float) -> bytes:
    r = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"Accept": "application/rdap+json, text/plain, */*"},
    )
    if r.status_code == 404:
        raise FileNotFoundError(url)
    r.raise_for_status()
    return r.content


def host_of(url: str) -> str:
    u = url if "://" in url else f"http://{url}"
    try:
        return (urlsplit(u).hostname or "").lower().strip(".")
    except ValueError:  # malformed (e.g. a bad IPv6 literal)
        return ""


def norm_url(url: str) -> str:
    """Scheme and host lowercased, trailing slash dropped: how feed and email URLs meet."""
    u = url.strip() if "://" in url else f"http://{url.strip()}"
    try:
        p = urlsplit(u)
    except ValueError:
        return u
    rest = p.path.rstrip("/") + (f"?{p.query}" if p.query else "")
    return f"{p.scheme.lower()}://{(p.netloc or '').lower()}{rest}"


def _is_ip(host: str) -> bool:
    try:
        ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


def sent_at(date_header: str | None) -> datetime | None:
    if not date_header:
        return None
    try:
        d = parsedate_to_datetime(date_header)
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=UTC)


class Intel:
    """Blocklist lookups (in memory) and cached RDAP domain ages; safe across threads."""

    def __init__(
        self,
        cache_dir: Path,
        new_domain_days: int = 30,
        rdap: bool = True,
        rdap_timeout: float = 4.0,
        max_lookups: int = 6,
        refresh_hours: float = 12.0,
        fetch: Fetch | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.new_domain_days = new_domain_days
        self.rdap = rdap
        self.rdap_timeout = rdap_timeout
        self.max_lookups = max_lookups
        self.refresh_hours = refresh_hours
        self.fetch = fetch or _http_fetch
        self.hosts: dict[str, frozenset[str]] = {}
        self.urls: dict[str, frozenset[str]] = {}
        self.updated: dict[str, str] = {}
        self._db_lock = threading.Lock()
        self._db: sqlite3.Connection | None = None

    # -- blocklists ----------------------------------------------------------------------

    def _feed_path(self, name: str) -> Path:
        return self.cache_dir / f"{name.lower().replace('.', '_')}.txt"

    def refresh_feeds(self, force: bool = False) -> None:
        """Download feeds older than `refresh_hours` (a failed download keeps the old copy),
        then load them all."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for name, (url, _) in FEEDS.items():
            path = self._feed_path(name)
            age_h = (
                (datetime.now().timestamp() - path.stat().st_mtime) / 3600
                if path.exists()
                else float("inf")
            )
            if not force and age_h < self.refresh_hours:
                continue
            try:
                body = self.fetch(url, 60.0)
            except Exception as exc:  # offline: keep whatever copy there is
                log.warning("could not download the %s feed: %s", name, exc)
                continue
            tmp = path.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(path)
        self.load_feeds()

    def load_feeds(self) -> None:
        for name, (_, kind) in FEEDS.items():
            path = self._feed_path(name)
            if not path.exists():
                continue
            lines = [
                ln.strip()
                for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            if kind == "hosts":
                self.hosts[name] = frozenset(ln.lower() for ln in lines)
            else:
                self.urls[name] = frozenset(norm_url(ln) for ln in lines)
                self.hosts[name] = frozenset(h for h in map(host_of, lines) if h)
            self.updated[name] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
                timespec="seconds"
            )
        log.info(
            "blocklists loaded: %s",
            ", ".join(f"{n} {len(self.hosts.get(n, ()))} hosts" for n in FEEDS),
        )

    @property
    def feeds_loaded(self) -> bool:
        return bool(self.hosts)

    def status(self) -> dict[str, object]:
        return {
            "feeds": {
                n: {"hosts": len(self.hosts.get(n, ())), "updated": self.updated.get(n)}
                for n in FEEDS
            },
            "rdap": self.rdap,
            "new_domain_days": self.new_domain_days,
        }

    def host_listed(self, host: str) -> list[str]:
        """Feeds listing this host, or the registered domain it belongs to."""
        if not host or host in SHARED_HOSTS:
            return []
        reg = registered_of(host)
        return [
            name
            for name, hosts in self.hosts.items()
            if host in hosts or (reg not in SHARED_HOSTS and reg != host and reg in hosts)
        ]

    def url_listed(self, url: str) -> list[str]:
        u = norm_url(url)
        return [name for name, urls in self.urls.items() if u in urls]

    # -- domain age ----------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.cache_dir / "rdap.sqlite", check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS rdap "
                "(domain TEXT PRIMARY KEY, registered TEXT, checked TEXT NOT NULL)"
            )
        return self._db

    def _cached(self, domain: str) -> tuple[bool, str | None]:
        with self._db_lock:
            row = (
                self._conn()
                .execute("SELECT registered, checked FROM rdap WHERE domain = ?", (domain,))
                .fetchone()
            )
        if row is None:
            return False, None
        ttl = RDAP_FOUND_TTL if row[0] else RDAP_MISS_TTL
        fresh = datetime.now(UTC) - datetime.fromisoformat(row[1]) < ttl
        return fresh, row[0]

    def registered_on(self, domain: str) -> str | None:
        """ISO registration date of a registered domain, or None if unknown."""
        fresh, value = self._cached(domain)
        if fresh:
            return value
        try:
            data = json.loads(self.fetch(RDAP_URL.format(domain), self.rdap_timeout))
            value = next(
                (
                    e["eventDate"][:10]
                    for e in data.get("events", [])
                    if e.get("eventAction") == "registration" and e.get("eventDate")
                ),
                None,
            )
        except Exception as exc:  # unknown TLD, rate limit, offline
            log.info("RDAP lookup of %s failed: %s", domain, exc)
            value = None
        with self._db_lock:
            self._conn().execute(
                "INSERT OR REPLACE INTO rdap VALUES (?, ?, ?)",
                (domain, value, datetime.now(UTC).isoformat()),
            )
            self._conn().commit()
        return value

    # -- an email --------------------------------------------------------------------------

    def check(
        self, sender: Sender | None, urls: list[str], sent: datetime | None = None
    ) -> IntelReport:
        """Findings for the sender, Reply-To and link domains of one email."""
        findings: list[Finding] = []
        domains: dict[str, DomainInfo] = {}

        def note(host: str, seen_as: str) -> None:
            if not host or _is_ip(host) or "." not in host:
                return
            reg = registered_of(host)
            info = domains.setdefault(reg, DomainInfo(reg, []))
            if seen_as not in info.seen_as:
                info.seen_as.append(seen_as)
            for name in self.host_listed(host):
                if name not in info.listed_on:
                    info.listed_on.append(name)
                    findings.append(
                        Finding(
                            "listed_domain",
                            "high",
                            f"{seen_as} domain {host} is on the {name} phishing blocklist",
                        )
                    )

        if sender is not None:
            note(sender.domain, "sender")
            if sender.reply_to and "@" in sender.reply_to:
                note(sender.reply_to.rsplit("@", 1)[1], "reply-to")
        for u in urls:
            note(host_of(u), "link")
            for name in self.url_listed(u):
                findings.append(
                    Finding("listed_url", "high", f"link {u} is on the {name} phishing feed")
                )

        if self.rdap:
            todo = [d for d in domains if d not in _WELL_KNOWN][: self.max_lookups]
            with ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
                ages = dict(zip(todo, pool.map(self.registered_on, todo), strict=True))
            when = (sent or datetime.now(UTC)).date()
            for d, reg in ages.items():
                info = domains[d]
                info.registered = reg
                if reg is None:
                    continue
                age = (when - date.fromisoformat(reg)).days
                if age < 0:  # registered after the email was sent: re-registered since
                    continue
                info.age_days = age
                if age < self.new_domain_days:
                    findings.append(
                        Finding(
                            "new_domain",
                            "high",
                            f"{' and '.join(info.seen_as)} domain {d} was registered {age} "
                            f"day{'s' if age != 1 else ''} before the email"
                            + (" was sent" if sent else ""),
                        )
                    )
        return IntelReport(findings, list(domains.values()), self.feeds_loaded)
