"""Sender analysis: who an email claims to be from versus where it actually came from.

The text model reads only the subject and body. Phishing often gives itself away in the
sender instead: a display name that says "PayPal" on an address at some unrelated domain, a
look-alike domain (paypa1.com), a reply address elsewhere, or a sending server that fails the
domain's SPF/DKIM/DMARC checks.

Every check is a named finding with a plain-language reason. How much each one is trusted is
measured, not assumed: `sentinel phishing sender-eval` counts how often it fires on
legitimate mail and decides whether it may raise an alert, only mark an email for review,
or not count at all (`models/phishing/sender_policy.json`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import Literal

from sentinel.phishing.urls import SUSPICIOUS_TLDS

# Brands phishing most often impersonates, with the registered domains they send from.
# Written from public knowledge before any evaluation data was read; any registered domain
# whose first label is the brand's own (paypal.de, amazon.co.jp) also counts as official.
BRANDS: dict[str, tuple[str, ...]] = {
    "paypal": ("paypal.com",),
    "microsoft": ("microsoft.com", "microsoftonline.com", "office.com", "live.com", "outlook.com"),
    "office 365": ("microsoft.com", "office.com", "office365.com", "microsoftonline.com"),
    "outlook": ("microsoft.com", "outlook.com", "live.com", "office.com"),
    "onedrive": ("microsoft.com", "onedrive.com", "live.com"),
    "sharepoint": ("microsoft.com", "sharepoint.com"),
    "apple": ("apple.com", "icloud.com", "itunes.com", "mac.com", "me.com"),
    "icloud": ("apple.com", "icloud.com"),
    "itunes": ("apple.com", "itunes.com"),
    "amazon": ("amazon.com", "amazonses.com", "amazon.co.uk"),
    "netflix": ("netflix.com",),
    "google": ("google.com", "gmail.com", "youtube.com"),
    "gmail": ("google.com", "gmail.com"),
    "facebook": ("facebook.com", "facebookmail.com", "meta.com"),
    "instagram": ("instagram.com", "facebookmail.com", "mail.instagram.com"),
    "whatsapp": ("whatsapp.com", "facebookmail.com"),
    "linkedin": ("linkedin.com",),
    "twitter": ("twitter.com", "x.com"),
    "dropbox": ("dropbox.com", "dropboxmail.com"),
    "docusign": ("docusign.com", "docusign.net"),
    "adobe": ("adobe.com",),
    "dhl": ("dhl.com", "dhl.de"),
    "fedex": ("fedex.com",),
    "ups": ("ups.com",),
    "usps": ("usps.com", "usps.gov"),
    "royal mail": ("royalmail.com",),
    "ebay": ("ebay.com",),
    "wells fargo": ("wellsfargo.com",),
    "bank of america": ("bankofamerica.com", "bofa.com"),
    "chase": ("chase.com", "jpmorgan.com", "jpmchase.com"),
    "citibank": ("citi.com", "citibank.com"),
    "hsbc": ("hsbc.com", "hsbc.co.uk"),
    "barclays": ("barclays.com", "barclays.co.uk"),
    "american express": ("americanexpress.com", "aexp.com"),
    "amex": ("americanexpress.com", "aexp.com"),
    "mastercard": ("mastercard.com",),
    "coinbase": ("coinbase.com",),
    "binance": ("binance.com",),
    "metamask": ("metamask.io",),
    "steam": ("steampowered.com", "steamcommunity.com"),
    "spotify": ("spotify.com",),
    "irs": ("irs.gov",),
    "hmrc": ("hmrc.gov.uk", "gov.uk"),
    "norton": ("norton.com", "nortonlifelock.com", "gen.digital"),
    "mcafee": ("mcafee.com",),
    "geek squad": ("geeksquad.com", "bestbuy.com"),
    "walmart": ("walmart.com",),
    "costco": ("costco.com",),
    "att": ("att.com", "att.net"),
    "verizon": ("verizon.com", "verizonwireless.com"),
    "zoom": ("zoom.us",),
    "wetransfer": ("wetransfer.com",),
}

FREEMAIL = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "yahoo.com",
        "ymail.com",
        "hotmail.com",
        "outlook.com",
        "live.com",
        "msn.com",
        "aol.com",
        "icloud.com",
        "me.com",
        "mail.com",
        "gmx.com",
        "gmx.net",
        "gmx.de",
        "web.de",
        "yandex.ru",
        "yandex.com",
        "mail.ru",
        "protonmail.com",
        "proton.me",
        "zoho.com",
        "qq.com",
        "163.com",
        "126.com",
        "rediffmail.com",
        "yahoo.co.uk",
        "hotmail.co.uk",
        "btinternet.com",
        "comcast.net",
    }
)

# Display-name words that claim an organisation rather than a person.
ROLE_WORDS = frozenset(
    {
        "support",
        "security",
        "billing",
        "account",
        "accounts",
        "service",
        "services",
        "helpdesk",
        "admin",
        "administrator",
        "verification",
        "team",
        "notification",
        "notifications",
        "alert",
        "alerts",
        "department",
        "customer",
        "payroll",
        "invoice",
        "invoices",
        "payment",
        "payments",
        "bank",
        "it",
        "webmaster",
        "postmaster",
        "mailbox",
    }
)

# Glyphs swapped in look-alike domains (paypa1, rnicrosoft, g00gle).
_HOMOGLYPHS = (("rn", "m"), ("vv", "w"), ("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("l", "i"))
_SECOND_LEVEL = {"co", "com", "net", "org", "gov", "ac", "edu", "ne", "or"}
_ADDR_IN_NAME = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
_WORD = re.compile(r"[a-z0-9]+")
_BRAND_CONTEXT = frozenset(
    {
        "com",
        "inc",
        "the",
        "id",
        "store",
        "online",
        "web",
        "mail",
        "no",
        "reply",
        "noreply",
        "do",
        "not",
        "your",
        "from",
        "official",
        "center",
        "centre",
    }
)

Severity = Literal["high", "medium", "low"]

# Checks that need the raw .eml (headers the CSV corpora do not keep); their false-alarm
# rate on legitimate mail cannot be measured here, so they never raise an alert alone.
HEADER_CHECKS = frozenset({"reply_to_elsewhere", "dmarc_fail", "spf_fail", "dkim_fail"})


@dataclass
class Finding:
    check: str
    severity: Severity
    text: str


@dataclass
class Sender:
    """What the headers say about the sender (only `from_` comes from the CSV corpora)."""

    from_: str = ""
    display_name: str = ""
    address: str = ""
    domain: str = ""  # full host after @
    registered: str = ""  # registered domain (mail.paypal.com -> paypal.com)
    reply_to: str = ""
    auth: dict[str, str] = field(default_factory=dict)  # spf/dkim/dmarc -> result


def registered_of(host: str) -> str:
    parts = [p for p in host.lower().strip(".").split(".") if p]
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _first_label(registered: str) -> str:
    return registered.split(".", 1)[0]


def parse_sender(from_: str | None, reply_to: str | None = None, auth: str | None = None) -> Sender:
    name, addr = parseaddr(from_ or "")
    addr = addr.strip().lower()
    domain = addr.rsplit("@", 1)[1].strip(".") if "@" in addr else ""
    _, reply = parseaddr(reply_to or "")
    return Sender(
        from_=from_ or "",
        display_name=name.strip().strip("\"'"),
        address=addr,
        domain=domain,
        registered=registered_of(domain) if domain else "",
        reply_to=reply.strip().lower(),
        auth=parse_auth(auth or ""),
    )


def parse_auth(header: str) -> dict[str, str]:
    """`spf=pass ... dkim=fail ... dmarc=fail` -> {"spf": "pass", "dkim": "fail", ...}."""
    out: dict[str, str] = {}
    for method in ("spf", "dkim", "dmarc"):
        m = re.search(rf"\b{method}=(\w+)", header, re.IGNORECASE)
        if m:
            out[method] = m.group(1).lower()
    return out


def _skeleton(label: str) -> str:
    s = label.replace("-", "")
    for a, b in _HOMOGLYPHS:
        s = s.replace(a, b)
    return s


def _one_edit(a: str, b: str) -> bool:
    """True when a and b differ by exactly one insertion, deletion, substitution or swap."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        i, j = diff[0], diff[-1]
        return len(diff) == 2 and j == i + 1 and a[i] == b[j] and a[j] == b[i]
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def _official(brand: str, registered: str) -> bool:
    label = brand.replace(" ", "")
    return registered in BRANDS[brand] or _first_label(registered) == label


# Brands that are also surnames or ordinary words ("Scott Norton", "Chase Miller") count
# only when the display name is the brand alone or the brand with organisation words.
AMBIGUOUS_BRANDS = frozenset(
    {"apple", "chase", "norton", "steam", "zoom", "ups", "outlook", "amazon", "adobe"}
)


def _brand_in(text: str) -> str | None:
    """The first brand named as whole words in `text` (display names, local parts)."""
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return None
    padded = f" {' '.join(tokens)} "
    for brand in BRANDS:
        if f" {brand} " in padded or (" " in brand and f" {brand.replace(' ', '')} " in padded):
            if brand in AMBIGUOUS_BRANDS:
                rest = set(tokens) - set(brand.split())
                if rest - ROLE_WORDS - _BRAND_CONTEXT:
                    continue
            return brand
    return None


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def check_sender(s: Sender) -> list[Finding]:
    if not s.address or "@" not in s.address or not s.registered:
        out = [Finding("no_address", "low", "the From line has no usable address")]
        return out + _header_findings(s) if s.from_ else _header_findings(s)
    return _address_findings(s) + _header_findings(s)


def _address_findings(s: Sender) -> list[Finding]:
    out: list[Finding] = []
    reg, label = s.registered, _first_label(s.registered)

    brand = _brand_in(s.display_name)
    if brand and not _official(brand, reg):
        out.append(
            Finding(
                "brand_impersonation",
                "high",
                f'display name says "{s.display_name}" but the address is at {reg}, '
                f"not a {brand.title()} domain",
            )
        )

    # "someone@x.org (via RT)": ticket systems and mailing lists relay for the named sender.
    relayed = re.search(r"\bvia\b", s.display_name, re.IGNORECASE) is not None
    for other in [] if relayed else _ADDR_IN_NAME.findall(s.display_name):
        if registered_of(other) != reg:
            out.append(
                Finding(
                    "address_in_name",
                    "high",
                    f"display name shows an address at {registered_of(other)}, "
                    f"but the mail is from {reg}",
                )
            )
            break

    parts = label.split("-")
    for b, domains in BRANDS.items():
        for official in domains:
            o = _first_label(official)
            if len(o) < 5 or label == o or _official(b, reg):
                continue
            look = _skeleton(label) == _skeleton(o) or (len(label) >= 5 and _one_edit(label, o))
            # n0rton-secure.com: one hyphenated part is the brand with swapped glyphs
            look |= len(parts) > 1 and any(p != o and _skeleton(p) == _skeleton(o) for p in parts)
            if look:
                out.append(
                    Finding(
                        "lookalike_domain",
                        "high",
                        f"{reg} looks like {official} but is a different domain",
                    )
                )
                break
            if o in parts or any(o == p for p in s.domain.split(".")[:-2]):
                out.append(
                    Finding(
                        "brand_in_domain",
                        "high",
                        f'{s.domain} uses the name "{o}" but is not registered to {official}',
                    )
                )
                break
        else:
            continue
        break

    if "xn--" in s.domain:
        out.append(
            Finding("punycode_domain", "medium", f"{s.domain} is an internationalised domain")
        )

    words = set(_WORD.findall(s.display_name.lower()))
    if reg in FREEMAIL and words & ROLE_WORDS:
        out.append(
            Finding(
                "freemail_role",
                "medium",
                f'"{s.display_name}" claims to be an organisation but writes from a free '
                f"{reg} account",
            )
        )

    if reg.rsplit(".", 1)[-1] in SUSPICIOUS_TLDS:
        out.append(
            Finding("suspicious_tld", "low", f".{reg.rsplit('.', 1)[-1]} is a high-abuse domain")
        )

    local = s.address.split("@", 1)[0].split("+", 1)[0]  # plus-tags are per-recipient
    letters = re.sub(r"[^a-z]", "", local)
    digits = sum(c.isdigit() for c in local)
    if (
        len(local) >= 12
        and (_entropy(local) > 3.8 or digits >= 6)
        and len(letters) < 0.7 * len(local)
    ):
        out.append(Finding("random_address", "low", f"{local}@ looks machine-generated"))
    return out


Role = Literal["alert", "review", "ignore"]


def load_policy(path: Path) -> dict[str, Role]:
    """check -> role, as decided by `sentinel phishing sender-eval`.

    Without the file every check is "review": a finding can mark an email suspicious but
    never raise an alert on a rule nobody has measured.
    """
    if not path.exists():
        return {}
    policy: dict[str, Role] = json.loads(path.read_text(encoding="utf-8"))["policy"]
    return policy


def role_of(check: str, policy: dict[str, Role]) -> Role:
    if check in HEADER_CHECKS:  # unmeasured on legitimate mail: never alert alone
        return "review" if policy.get(check) != "ignore" else "ignore"
    return policy.get(check, "review")


def _header_findings(s: Sender) -> list[Finding]:
    """Reply-To and SPF/DKIM/DMARC: only present when the raw .eml is supplied."""
    out: list[Finding] = []
    reg = s.registered or "the sender's domain"
    if s.reply_to and "@" in s.reply_to:
        rreg = registered_of(s.reply_to.rsplit("@", 1)[1])
        if rreg and rreg != s.registered:
            free = " (a free mail account)" if rreg in FREEMAIL else ""
            out.append(
                Finding(
                    "reply_to_elsewhere",
                    "medium",
                    f"replies go to {rreg}{free}, not to {reg}",
                )
            )

    meaning = {
        "dmarc": f"the message fails {reg}'s own rule for proving mail really comes from it",
        "spf": f"the sending server is not one {reg} lists as allowed to send its mail",
        "dkim": "the message's signature does not verify (altered or forged)",
    }
    for method in ("dmarc", "spf", "dkim"):
        result = s.auth.get(method)
        if result in ("fail", "softfail", "permerror"):
            out.append(
                Finding(
                    f"{method}_fail",
                    "high" if method == "dmarc" else "medium",
                    f"{method.upper()} {result}: {meaning[method]}",
                )
            )
    return out


@dataclass
class ParsedEmail:
    subject: str
    body: str
    sender: Sender
    date: str | None


def _body(msg: EmailMessage) -> str:
    """The text/plain part, else the HTML part (HTML is stripped later by model_input)."""
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
    except (KeyError, LookupError):
        part = None
    if part is None:
        return ""
    try:
        content = part.get_content()
    except (LookupError, ValueError, AssertionError):
        payload = part.get_payload(decode=True)
        content = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else ""
    return content if isinstance(content, str) else ""


def _auth_header(msg: EmailMessage) -> str:
    """Authentication-Results added by the receiving server (the topmost one).

    Lower ones can be forged by the sender; only the recipient's own server is trusted.
    """
    value = msg.get_all("Authentication-Results") or msg.get_all("ARC-Authentication-Results")
    return str(value[0]) if value else ""


def parse_eml(raw: bytes) -> ParsedEmail:
    """A raw RFC 5322 message (.eml). Attachments are never decoded or opened."""
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    assert isinstance(msg, EmailMessage)

    def header(name: str) -> str:
        try:
            v = msg.get(name)
        except (ValueError, IndexError, AttributeError):  # malformed header
            return ""
        return str(v) if v is not None else ""

    from_ = header("From")
    if from_ and "@" not in from_:  # some senders put only the address in Sender
        from_ = header("Sender") or from_
    reply = ", ".join(a for _, a in getaddresses([header("Reply-To")]) if a)
    return ParsedEmail(
        subject=header("Subject"),
        body=_body(msg),
        sender=parse_sender(from_, reply.split(",")[0] if reply else "", _auth_header(msg)),
        date=header("Date") or None,
    )
