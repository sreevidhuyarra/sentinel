# Blocklists and domain age

Each email's sender, Reply-To and link domains are looked up in phishing blocklists (downloaded; nothing about the email is sent out). Roles follow the sender-check rules: **alert** at most 0.1% of legitimate validation emails, **review** at most 2%. Domain age (RDAP) is review-only: it cannot be measured on these corpora.

Feed snapshot used:

- Phishing.Database: 392,164 hosts, downloaded 2026-10-27T18:43:37+00:00
- OpenPhish: 276 hosts, downloaded 2026-10-27T18:43:38+00:00

## How often each check fires

| check | role | legitimate val | legitimate test | phishing (Nazario) test | phishing pot dev (hindsight) | phishing pot test (hindsight) |
|---|---|---|---|---|---|---|
| listed_url | alert | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| listed_domain | alert | 0.07% | 0.05% | 0.32% | 1.07% | 0.95% |
| new_domain | review | not measured | | | | |

Phishing Pot columns are hindsight: today's feeds against emails sent months or years earlier. They show overlap, not what the check would have caught at the time.

## Test results

Share flagged **malicious** and **malicious or suspicious**, before and after adding the blocklist checks to the text, link and sender checks.

| emails | n | malicious | + blocklists | malicious or suspicious | + blocklists |
|---|---|---|---|---|---|
| legitimate | 17584 | 0.44% | 0.49% | 5.70% | 5.74% |
| phishing | 309 | 81.55% | 81.55% | 95.47% | 95.47% |
| fraud | 650 | 90.77% | 90.77% | 99.38% | 99.38% |
| spam | 17544 | 89.92% | 89.93% | 97.75% | 97.75% |
