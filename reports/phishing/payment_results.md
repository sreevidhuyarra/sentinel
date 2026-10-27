# Payment and invoice-fraud checks

Wording checks (new bank details, gift cards, wire transfers, payment urgency) only mark an email for review: the corpora hold few genuine invoices, so their legitimate false-alarm rate would not carry over to a finance inbox. A payment request together with a From-line warning sign is measured: alert at most 0.1% of legitimate validation emails, review at most 2%.

None of these corpora is invoice fraud (the fraud kind is advance-fee scams), so the catch rates below are not an invoice-fraud detection rate.

## How often each check fires

| check | role | legitimate val | legitimate test | phishing (Nazario) test | fraud (advance-fee) test | phishing pot dev | phishing pot test |
|---|---|---|---|---|---|---|---|
| payment_change | review | 0.00% | 0.00% | 0.00% | 0.92% | 0.30% | 0.04% |
| gift_card_request | review | 0.00% | 0.00% | 0.00% | 0.00% | 0.02% | 0.00% |
| wire_request | review | 0.00% | 0.00% | 0.00% | 0.00% | 0.02% | 0.00% |
| payment_urgency | review | 0.02% | 0.03% | 1.29% | 3.69% | 0.30% | 0.24% |
| payment_from_risky_sender | alert | 0.00% | 0.00% | 0.00% | 0.15% | 0.46% | 0.12% |

## Test results

| emails | n | malicious | + payment | malicious or suspicious | + payment |
|---|---|---|---|---|---|
| legitimate | 17584 | 0.44% | 0.44% | 5.70% | 5.72% |
| phishing | 309 | 81.55% | 81.55% | 95.47% | 96.12% |
| fraud | 650 | 90.77% | 90.77% | 99.38% | 99.38% |
| spam | 17544 | 89.92% | 89.92% | 97.75% | 97.75% |
| Phishing Pot dev | 5046 | 62.41% | 62.45% | 95.20% | 95.26% |
| Phishing Pot test | 5046 | 76.36% | 76.36% | 96.25% | 96.27% |
