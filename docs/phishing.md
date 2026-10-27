[← Back to the README](../README.md)

# Module 3: phishing

`uv run sentinel phishing download` fetches the Phishing Email Curated Datasets (11 public
corpora, [Zenodo 8339691](https://doi.org/10.5281/zenodo.8339691), CC BY 4.0) and
`uv run dvc repro prepare_phishing` turns them into 200,517 deduplicated emails, split so no
sender domain or subject campaign appears in two splits. Label 1 means phishing only in the
Nazario corpora, fraud in the Nigerian ones and spam elsewhere, so results report each kind.

Fine-tuning DeBERTa-v3-small with LoRA needs a GPU and runs on Kaggle:

1. Kaggle → Create → New Dataset → upload the four files in `kaggle/phishing-emails/`
   (train/val/test parquet + README), name it `sentinel-phishing-emails`, keep it private.
2. Kaggle → Create → New Notebook → File → Import Notebook →
   `notebooks/phishing_lora_kaggle.ipynb`.
3. Session options: Accelerator **GPU T4 x2**, Internet **On**; Add Input → your dataset.
4. Save Version → Save & Run All (~75 min on 2× T4), then download
   `phishing_lora_output.zip` from Output and unzip it into `models/phishing/`.

Then `uv run dvc repro build_phishing` (~11 min, CPU) merges the adapter, exports ONNX,
quantizes it, benchmarks every variant and registers `phishing-classifier` in MLflow.
Full tables: [reports/phishing/results.md](../reports/phishing/results.md),
[model card](../reports/phishing/model_card.md).

**Model.** DeBERTa-v3-small (141M parameters) with LoRA rank 16 on the attention
projections (0.72% of weights trainable), 256 tokens of subject + body, 2 epochs.

**Results (test split, 36,087 emails):**

| Threshold | F1 | False alarms on legitimate mail | Malicious caught | Phishing caught |
|---|---|---|---|---|
| 0.5 | 0.970 | 3.99% | 97.6% | 93.2% |
| **0.989 (deployed: 2% validation budget)** | **0.944** | **0.31%** | **89.7%** | **77.7%** |

PR-AUC 0.997, ROC-AUC 0.997. A TF-IDF + logistic regression baseline on the same split
reaches ROC-AUC 0.979 and cannot get its false alarms below ~2.7%.

**Optimisation** (1,000 test emails, laptop CPU):

| Model | Size | F1 | Verdicts changed | Latency p50 / p95 |
|---|---|---|---|---|
| PyTorch, float32 | — | 0.957 | — | 204 / 252 ms |
| ONNX, float32 | 568 MB | 0.957 | 0 | 130 / 161 ms |
| **ONNX, int8 embeddings (deployed)** | **273 MB** | **0.957** | **2** | **129 / 158 ms** |
| ONNX, int8 everything | 165 MB | 0.872 | 85 | 87 / 113 ms |

What we learned:

- **ONNX alone gives the speed-up** (1.6× lower latency); quantizing the embedding table
  halves the size at no accuracy cost. Full int8 is fastest, but DeBERTa's outlier
  activations shift high probabilities enough to flip 85 of 1,000 verdicts at the deployed
  threshold, so it is not used. The report's target of ≤1 F1 point with ≥2× speed and ~4×
  size is not reachable on this model with dynamic int8.
- **Export bugs a smoke test catches.** Transformers 5 loads the fp16 checkpoint as fp16
  (breaks mixed-precision training, ~100× slower on CPU): load as float32. The legacy ONNX
  exporter mistranslates DeBERTa's masked softmax: exact on unpadded input, wrong on padded
  batches (logits off by up to 2). The `torch.export` exporter is exact; a regression test
  guards it.
- **Mostly spam, little phishing.** Only ~1,500 emails are true phishing, and phishing
  recall is the weakest per kind, so it is reported separately rather than hidden in F1.

`POST /score/phishing` takes `{subject, body, urls[]}` and returns the verdict,
probability, severity, the URLs found, and the sentences whose removal most lowers the
score, e.g. *"Dear customer,"* and *"Urgent: your account has been suspended"* for a
credential-phishing email (~200 ms with explanation).

**Three-way verdict.** The alert threshold is strict, so `verdict` is one of three values:
- `malicious`: an alert, as before (`is_malicious`).
- `suspicious` (severity Low, for review): the text score falls between
  `phishing.suspicious_threshold` (0.5) and the alert threshold, or a link in the email is
  one the URL checker judges malicious. `suspicious_because` says which.
- `clean`: neither.

On the test split, the review band adds 3.7% of legitimate mail and lifts phishing caught
from 77.7% to 93.2%. A hand-written test email shows the gap it closes: *"Urgent: Your
account has been suspended… verify your identity… [link]"* scores 0.964, under the 0.989
alert threshold, and its link scores 1.000. It used to pass as legitimate and is now
suspicious. A blind spot remains: an invoice-fraud email ("Please find attached the overdue
invoice… complete the payment today") scores 0.014. Everyday business wording is what
legitimate training mail looks like, and only about 1,500 of the 200k training emails are
true phishing. Fixing that needs modern phishing and invoice-fraud data, not a different
threshold.

### Sender checks and a modern phishing test set

The text model reads only the subject and body, so it misses phishing whose giveaway is the
sender. `POST /score/phishing` now also takes the From line (`sender`), `reply_to`, or the
whole message (`eml`). The dashboard has an .eml upload. `phishing/sender.py` checks:

- **Address** (From line):
  - a display name naming a brand on an unrelated domain (`PayPal <service@paypa1.com>`);
  - an address in the display name that differs from the real one;
  - look-alike domains (`paypa1`, `n0rton-secure`);
  - a brand inside another domain (`paypal.com.secure-check.ru`);
  - punycode;
  - an organisation name on a free mail account;
  - high-abuse TLDs;
  - machine-generated addresses;
  - no usable address.
- **Headers** (.eml only): Reply-To at another domain, and SPF, DKIM or DMARC failures. These
  are taken from the topmost Authentication-Results, the one added by the recipient's own
  server; lower ones can be forged.

How far each check is trusted is measured, not assumed (`sentinel phishing sender-eval`,
DVC stage `sender_eval`). The rules were fixed before any modern phishing was scored:

- **May raise an alert on its own:** a check that fires on ≤ 0.1% of legitimate validation
  emails, as long as all such checks together add ≤ 0.5% false alarms.
- **Can only mark an email suspicious:**
  - a check that fires on up to 2% of legitimate validation emails;
  - every header check, because the email corpora keep only the From line, so their
    false-alarm rate on legitimate mail can't be measured here.
- **Ignored:** anything more frequent.

The policy is saved to `models/phishing/sender_policy.json`.

Modern phishing comes from [Phishing Pot](https://github.com/rf-peixoto/phishing_pot), which
is licensed CC BY-NC 4.0 and pinned at commit `2933c03`. It holds real phishing received from
about 2022 on, as raw .eml with headers:

- 12,271 messages, 10,092 after removing duplicates; none share text with the training corpora;
- split by arrival order: the earlier half for development, the later half held out for test.

The emails are not redistributed here.

Test results (`reports/phishing/sender_results.md`):

| emails (test) | n | text + link: alert | + sender checks: alert | + sender: alert or review |
|---|---|---|---|---|
| legitimate (with a From line) | 13,964 | 0.39% | 0.55% | 7.08% |
| phishing (Nazario) | 309 | 77.7% | 81.6% | 95.5% |
| Phishing Pot (modern, held out) | 5,046 | 74.9% | 76.4% (From line) | 96.3% (with headers) |

What this shows:

- **Modern phishing is harder.** The deployed text and link models alert on 74.9% of modern
  phishing (60.2% on the development half), against 77.7% of the older Nazario set.
  Counting reviews, they flag 95%.
- **The From line catches a few more, at a cost.** The checks catch 1.5 points more modern
  phishing and 3.9 points more Nazario phishing as alerts, and add 0.16 points of false
  alarms on legitimate mail.
- **The headers help with review, not alerts.** They raise "alert or review" on modern
  phishing from 95.3% to 96.3%. SPF failed on 49% of held-out modern phishing, Reply-To
  pointed elsewhere on 15% and DMARC failed on 7%. Because those checks are unmeasured on
  legitimate mail, they only mark email for review.
- **Most of the legitimate test false alarms are a known flaw found on test.** The
  one-letter-difference rule makes `email.si`, `email.com` and `email.cz` "look like" gmail
  and `tunes.org` like itunes. Fixing it now would tune on the test set, so it is reported
  rather than patched; a fix (skip ordinary words) needs fresh test data to measure.

The invoice-fraud blind spot is unchanged: such emails need a sender giveaway, or invoice-fraud
training data.

### Domain reputation: blocklists and domain age

`phishing/intel.py` looks up the sender, Reply-To and link domains of every checked email:

- **Blocklists**, downloaded every 12 hours and looked up locally, so nothing about the email
  leaves the machine:
  - [Phishing.Database](https://github.com/Phishing-Database/Phishing.Database) (MIT): about
    392,000 hostnames currently serving phishing;
  - the [OpenPhish](https://openphish.com) community feed: the newest ~300 phishing URLs.

  Shared platforms (sites.google.com, link shorteners, feedburner, …) are listed because of
  one page among millions, so a host listing there is ignored; an exact URL listing still
  counts. (URLhaus was the first choice, but its free download now needs an API key.)
- **Domain age** from RDAP, WHOIS's successor. A domain registered less than 30 days before
  the email was sent is a classic phishing sign. Only the registered domain is sent to
  rdap.org, never the email; answers are cached, and free-mail, brand and shared-platform
  domains are never looked up. `phishing.intel.rdap: false` in `params.yaml` turns it off.

The same rules as the sender checks decide each check's role (`sentinel phishing intel-eval`,
`reports/phishing/intel_results.md`):

| check | legitimate val | legitimate test | role |
|---|---|---|---|
| link URL on a phishing feed | 0.00% | 0.00% | alert |
| sender or link domain on a blocklist | 0.07% | 0.05% | alert |
| domain registered < 30 days before the email | not measurable | | review |

- **The measurement is a snapshot.** The feeds change daily, so the report records the
  download dates. The corpora are years older than the feeds, so a legitimate domain that
  has since expired and been re-registered by a phisher (an old Irish ISP domain, 12
  validation emails) counts as a false alarm. The rate is an upper bound.
- **Two shared platforms were found on validation.** On the first run, domain listings
  fired on 0.11% of legitimate validation mail, just over the 0.1% alert limit. Two of the
  three hosts were shared platforms (feeds.feedburner.com, angelfire.com) missing from the
  exclusion list. With them excluded it is 0.07%; the test split (0.05%) was not looked at.
- **On these corpora the blocklists catch nothing extra.** The corpora are old, and the feeds
  list today's phishing. The Phishing Pot overlap (about 1% of modern phishing) is hindsight:
  today's feed against emails sent earlier. The value is on live mail. An email whose link
  is on today's OpenPhish feed is now malicious even when its text scores 0.009.
- **Domain age is review-only.** RDAP gives today's registration date, and the corpora's
  domains are from 2000–2020, so "new at the time of the email" can't be rebuilt for them.

### Payment requests and invoice fraud

Invoice scams read like ordinary business mail, so the text model passes them (the invoice
example above scores 0.000). What gives them away is the request and who makes it.
`phishing/payment.py` finds the request:

- pay to **new or changed bank details**;
- **buy gift cards**;
- make a **wire or bank transfer**;
- **payment urgency**: an overdue or outstanding invoice, or payment due today or immediately.

A payment request together with any sender or domain warning sign (a company name on
free mail, a look-alike domain, a domain registered last week, Reply-To elsewhere) becomes
`payment_from_risky_sender`. The dashboard shows a "Payment request" card advising
confirmation by phone, using a number the analyst already has.

How far each is trusted (`sentinel phishing payment-eval`, `reports/phishing/payment_results.md`):

| check | legitimate val | legitimate test | role |
|---|---|---|---|
| any payment wording | 0.00–0.03% | 0.00–0.03% | review |
| payment request + sender warning sign | 0.00% | 0.00% | alert |

- **Payment wording is review-only by design,** although its measured rate would allow
  alerts. The patterns were designed on the training and validation splits only. The corpora
  hold few genuine invoices, while a real accounts-payable inbox receives invoice reminders and
  bank-detail notices every day. The training split alone has 44 legitimate "Past Due
  Invoice" emails, so the near-zero rate would not carry over.
- **No dataset here is invoice fraud.** The "fraud" category is advance-fee scams. The
  measured effect is small: Nazario phishing flagged malicious or suspicious goes from 95.5%
  to 96.1%, and legitimate mail from 5.70% to 5.72%, with no new alerts on either. None of
  this is an invoice-fraud detection rate.
- **On the live API**, the invoice example from a plain business sender is now
  *suspicious* (it was clean). From `Accounts Payable Team <…@gmail.com>` it is *malicious*. A
  "please note our new bank details" email is *suspicious*, and an ordinary lunch email stays
  clean.

Measuring invoice-fraud detection, and teaching the text model, needs business-email-compromise
training data and a new fine-tuning run.

### URL classifier

`uv run dvc repro train_url` trains character 3–5-gram TF-IDF (on normalised URLs) plus 23
lexical features (length, digit ratio, subdomains, entropy, suspicious TLD, IP host,
shortener, sensitive words, ...) with LightGBM. Data: PhiUSIIL (UCI 967) + Hannousse &
Yahiouche 2021, 242k raw URLs, split by registered domain. ISCX-URL2016, named in the
design, sits behind a registration form, and its open mirrors carry only pre-computed
features that serving could not reproduce. Report:
[reports/phishing/url_results.md](../reports/phishing/url_results.md).

| Test URLs | ROC-AUC | False alarms on legitimate | Malicious caught |
|---|---|---|---|
| All (deployed, source-balanced) | 0.999 | 0.00% | 34.4% |
| Hannousse only (realistic legitimate URLs) | 0.979 | 0.09% | 38.6% |
| Trained on PhiUSIIL only, scored on Hannousse | 0.661 | 99.5% | 99.8% |

- **Dataset shortcut.** Every legitimate PhiUSIIL URL is a bare homepage, so a model trained
  on it learns "has a path → malicious": trained on PhiUSIIL alone it flags 99.5% of
  Hannousse's legitimate URLs. Balancing the two sources and setting the threshold on
  Hannousse's realistic legitimate URLs keeps false alarms at 0.09%, but the price is a
  strict threshold that catches ~39% of malicious URLs. It works as a high-confidence link
  checker (`POST /score/url`, with TreeSHAP reasons such as *"suspicious tld = 1"*).
- **Links do not improve email verdicts.** Legitimate email links (newsletters, tracking,
  documents) look unlike the training data's legitimate URLs. On validation, no URL
  threshold let links flag emails while adding under 0.5 points of false alarms, so links
  are shown but do not change the email verdict. A learned text + link combination adds only
  ~1–2 points of malicious mail caught at matched false-alarm rates.
