# SOC copilot results

## Retrieval (no LLM)

105 gold alerts. Recall@k = an acceptable technique is among the top k candidates; MRR = mean reciprocal rank of the first acceptable one.

| Retriever | R@1 | R@3 | R@5 | R@8 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 45.7% | 79.0% | 80.0% | 81.0% | 0.619 | 37 |
| dense | 60.0% | 83.8% | 84.8% | 84.8% | 0.688 | 86 |
| hybrid | 75.2% | 87.6% | 87.6% | 87.6% | 0.811 | 139 |
| hybrid+rerank | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 1012 |
| hybrid+rerank, no family hints | 16.2% | 40.0% | 65.7% | 67.6% | 0.308 | 1010 |
| hybrid, no family hints | 21.9% | 32.4% | 60.0% | 67.6% | 0.325 | 89 |
| hybrid+rerank + behaviour | 35.2% | 71.4% | 83.8% | 84.8% | 0.528 | 727 |
| hybrid+rerank, behaviour, no family hints | 40.0% | 63.8% | 81.9% | 83.8% | 0.543 | 742 |
| deployed: hybrid+rerank, behaviour candidates merged | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 703 |

Recall@3 by label (hybrid + re-rank):

| Label | R@3 |
|---|---|
| Botnet | 0.0% |
| DDoS | 100.0% |
| DoS GoldenEye | 100.0% |
| DoS Hulk | 100.0% |
| DoS Slowhttptest | 100.0% |
| DoS Slowloris | 100.0% |
| FTP-Patator | 100.0% |
| Heartbleed | 0.0% |
| Infiltration | 100.0% |
| Infiltration - Portscan | 100.0% |
| Portscan | 100.0% |
| SSH-Patator | 100.0% |
| Web Attack - Brute Force | 0.0% |
| Web Attack - SQL Injection | 100.0% |
| Web Attack - XSS | 100.0% |
| fraud | 100.0% |
| phishing | 100.0% |

## Incident reports (LLM)

105 reports; generator gemini, providers used {'gemini': 105}; judge None.

| Metric | Value | Target (design) |
|---|---|---|
| Technique precision@1 | 80.0% | >= 70% |
| Technique hit@3 | 87.6% | |
| Acceptable technique among candidates | 92.4% | |
| CVE recall (alerts with a known CVE) | 0.0% | |
| Key facts in the prose (IPs, port) | 96.7% | |
| Facts block complete (attacker, target; built by code) | 100.0% | |
| Reports still inventing an IP after verify | 0.0% | 0% |
| Passed verification on the first draft | 99.0% | |
| Citation validity (after verify) | 100.0% | 100% |
| Invalid citations dropped per report | 0.00 | |
| Faithfulness (judge: supported claims) | n/a | |
| Tokens in / out per report | 2,152 / 756 | |
| LLM seconds per report | 3.8 | |

| Label | P@1 |
|---|---|
| Botnet | 100.0% |
| DDoS | 100.0% |
| DoS GoldenEye | 33.3% |
| DoS Hulk | 57.1% |
| DoS Slowhttptest | 83.3% |
| DoS Slowloris | 100.0% |
| FTP-Patator | 100.0% |
| Heartbleed | 0.0% |
| Infiltration | 0.0% |
| Infiltration - Portscan | 100.0% |
| Portscan | 100.0% |
| SSH-Patator | 100.0% |
| Web Attack - Brute Force | 0.0% |
| Web Attack - SQL Injection | 100.0% |
| Web Attack - XSS | 100.0% |
| fraud | 100.0% |
| phishing | 100.0% |
