# Sender checks

Who an email claims to be from versus where it came from. Checks are measured on legitimate validation mail and given a role: **alert** (may flag an email alone), **review** (marks it suspicious) or **ignore**. Header checks (Reply-To, SPF, DKIM, DMARC) need the raw .eml; the corpora keep only the From line, so their false-alarm rate on legitimate mail is unmeasured and they are limited to review.

Modern phishing: [Phishing Pot](https://github.com/rf-peixoto/phishing_pot) at commit `2933c03` (CC BY-NC 4.0), split by arrival order; the later half is the held-out test.

## Policy and how often each check fires

| check | role | legitimate val | phishing (Nazario) val | Phishing Pot dev | legitimate test | Phishing Pot test |
|---|---|---|---|---|---|---|
| brand_impersonation | alert | 0.01% | 15.86% | 5.37% | 0.04% | 2.93% |
| address_in_name | alert | 0.01% | 1.29% | 0.26% | 0.02% | 0.08% |
| lookalike_domain | alert | 0.00% | 0.97% | 0.06% | 0.11% | 0.04% |
| brand_in_domain | alert | 0.01% | 0.32% | 0.04% | 0.00% | 0.06% |
| punycode_domain | alert | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| freemail_role | alert | 0.00% | 0.32% | 0.50% | 0.00% | 0.14% |
| suspicious_tld | alert | 0.00% | 1.62% | 1.19% | 0.00% | 0.12% |
| random_address | review | 0.26% | 2.91% | 5.61% | 0.09% | 1.19% |
| no_address | review | 0.16% | 0.97% | 19.90% | 0.20% | 0.26% |
| reply_to_elsewhere | review | 0.00% | 0.00% | 13.52% | 0.00% | 14.76% |
| dmarc_fail | review | 0.00% | 0.00% | 16.65% | 0.00% | 7.15% |
| spf_fail | review | 0.00% | 0.00% | 15.22% | 0.00% | 49.41% |
| dkim_fail | review | 0.00% | 0.00% | 10.84% | 0.00% | 13.56% |

## Test results

Share flagged **malicious** (an alert) and **malicious or suspicious** (an alert or a review). Corpus emails count only when they have a From line.

| emails | n | text + link: malicious | + sender: malicious | text + link: malicious or suspicious | + sender: malicious or suspicious |
|---|---|---|---|---|---|
| legitimate | 13964 | 0.39% | 0.55% | 6.66% | 7.08% |
| phishing (Nazario) | 309 | 77.67% | 81.55% | 94.17% | 95.47% |
| fraud | 592 | 90.20% | 90.71% | 99.32% | 99.32% |
| spam | 14647 | 89.95% | 90.05% | 97.71% | 97.79% |

Phishing Pot (modern phishing, raw .eml):

| split | configuration | n | malicious | malicious or suspicious |
|---|---|---|---|---|
| dev | text + link | 5046 | 60.17% | 92.55% |
| dev | text + link + sender (From line only) | 5046 | 62.41% | 93.78% |
| dev | text + link + sender (with headers) | 5046 | 62.41% | 95.20% |
| test | text + link | 5046 | 74.85% | 94.95% |
| test | text + link + sender (From line only) | 5046 | 76.36% | 95.32% |
| test | text + link + sender (with headers) | 5046 | 76.36% | 96.25% |
