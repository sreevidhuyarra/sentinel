# SOC copilot results

## Injection guard

Trained on 8,465 segments; chosen on validation: **deberta** (threshold 0.958, set for <= 1% false alarms on clean validation documents).

| Candidate | Val recall | Val false alarms | Train time |
|---|---|---|---|
| embed_lr | 93.9% | 0.9% | 980 s |
| deberta | 98.3% | 0.9% | 9362 s |

Held-out test (injection phrasings, email texts and hard negatives never seen in training):

| Detector | Source | Recall | False alarms | Injections / clean |
|---|---|---|---|---|
| rules | all | 71.9% | 0.1% | 1026 / 1090 |
| rules | deepset | 16.7% | 0.0% | 60 / 56 |
| rules | synthetic_email | 88.3% | 0.2% | 469 / 531 |
| rules | synthetic_log | 63.2% | 0.0% | 497 / 503 |
| classifier | all | 69.4% | 0.6% | 1026 / 1090 |
| classifier | deepset | 78.3% | 0.0% | 60 / 56 |
| classifier | synthetic_email | 49.5% | 1.3% | 469 / 531 |
| classifier | synthetic_log | 87.1% | 0.0% | 497 / 503 |
| rules+classifier | all | 94.8% | 0.7% | 1026 / 1090 |
| rules+classifier | deepset | 81.7% | 0.0% | 60 / 56 |
| rules+classifier | synthetic_email | 91.5% | 1.5% | 469 / 531 |
| rules+classifier | synthetic_log | 99.6% | 0.0% | 497 / 503 |

Scanning cost: 867 ms per document on CPU.

## Retrieval (no LLM)

108 gold alerts. Recall@k = an acceptable technique is among the top k candidates; MRR = mean reciprocal rank of the first acceptable one.

| Retriever | R@1 | R@3 | R@5 | R@8 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 50.9% | 78.7% | 79.6% | 80.6% | 0.644 | 26 |
| dense | 60.2% | 80.6% | 84.3% | 84.3% | 0.683 | 67 |
| hybrid | 71.3% | 85.2% | 85.2% | 85.2% | 0.781 | 101 |
| hybrid+rerank | 81.5% | 81.5% | 81.5% | 90.7% | 0.827 | 1272 |
| hybrid+rerank, no family hints | 17.6% | 38.0% | 64.8% | 67.6% | 0.319 | 1130 |
| hybrid, no family hints | 20.4% | 33.3% | 58.3% | 65.7% | 0.311 | 81 |

Recall@3 by label (hybrid + re-rank):

| Label | R@3 |
|---|---|
| Botnet | 0.0% |
| Botnet - Attempted | 0.0% |
| DDoS | 100.0% |
| DoS GoldenEye | 100.0% |
| DoS Hulk | 100.0% |
| DoS Slowhttptest | 100.0% |
| DoS Slowloris | 100.0% |
| FTP-Patator | 100.0% |
| Heartbleed | 0.0% |
| Infiltration | 80.0% |
| Infiltration - Portscan | 100.0% |
| Portscan | 100.0% |
| SSH-Patator | 100.0% |
| Web Attack - Brute Force | 0.0% |
| Web Attack - SQL Injection | 100.0% |
| Web Attack - XSS | 100.0% |
| fraud | 100.0% |
| phishing | 100.0% |

## Incident reports (LLM)

108 reports; generator ollama, providers used {'ollama': 108}; judge ollama:llama3.2:3b.

| Metric | Value | Target (design) |
|---|---|---|
| Technique precision@1 | 81.5% | >= 70% |
| Technique hit@3 | 82.4% | |
| Acceptable technique among candidates | 90.7% | |
| CVE recall (alerts with a known CVE) | 0.0% | |
| Key facts mentioned (IPs, port) | 50.7% | |
| Passed verification on the first draft | 100.0% | |
| Citation validity (after verify) | 100.0% | 100% |
| Invalid citations dropped per report | 0.00 | |
| Faithfulness (judge: supported claims) | 82.6% | |
| Tokens in / out per report | 1,838 / 451 | |
| LLM seconds per report | 64.3 | |

| Label | P@1 |
|---|---|
| Botnet | 0.0% |
| Botnet - Attempted | 0.0% |
| DDoS | 100.0% |
| DoS GoldenEye | 100.0% |
| DoS Hulk | 100.0% |
| DoS Slowhttptest | 100.0% |
| DoS Slowloris | 100.0% |
| FTP-Patator | 100.0% |
| Heartbleed | 0.0% |
| Infiltration | 80.0% |
| Infiltration - Portscan | 100.0% |
| Portscan | 100.0% |
| SSH-Patator | 100.0% |
| Web Attack - Brute Force | 0.0% |
| Web Attack - SQL Injection | 100.0% |
| Web Attack - XSS | 100.0% |
| fraud | 100.0% |
| phishing | 100.0% |

## Red team: prompt injection

24 attacked alerts (email bodies and correlated log lines carrying one injection in a held-out phrasing); generator ollama.

| Defense | Cases | Attack success | canary | downgrade | leak | Guard detected |
|---|---|---|---|---|---|---|
| naive | 24 | 12.5% | 0.0% | 37.5% | 0.0% | n/a |
| structural | 24 | 12.5% | 0.0% | 37.5% | 0.0% | n/a |
| guard+structural | 24 | 0.0% | 0.0% | 0.0% | 0.0% | 95.8% |

Severity is rule-based in every configuration, so no injection can lower it; a detected injection raises it by one level.
