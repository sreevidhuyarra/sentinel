# SOC copilot results

## Retrieval (no LLM)

105 gold alerts. Recall@k = an acceptable technique is among the top k candidates; MRR = mean reciprocal rank of the first acceptable one.

| Retriever | R@1 | R@3 | R@5 | R@8 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 45.7% | 79.0% | 80.0% | 81.0% | 0.619 | 29 |
| dense | 60.0% | 83.8% | 84.8% | 84.8% | 0.688 | 59 |
| hybrid | 75.2% | 87.6% | 87.6% | 87.6% | 0.811 | 142 |
| hybrid+rerank | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 1104 |
| hybrid+rerank, no family hints | 16.2% | 40.0% | 65.7% | 67.6% | 0.308 | 1282 |
| hybrid, no family hints | 21.9% | 32.4% | 60.0% | 67.6% | 0.325 | 123 |
| hybrid+rerank + behaviour | 35.2% | 71.4% | 83.8% | 84.8% | 0.528 | 1390 |
| hybrid+rerank, behaviour, no family hints | 40.0% | 63.8% | 81.9% | 83.8% | 0.543 | 1424 |
| deployed: hybrid+rerank, behaviour candidates merged | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 1337 |

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

18 reports; generator ollama, providers used {'ollama': 18}; judge ollama:llama3.2:3b.

| Metric | Value | Target (design) |
|---|---|---|
| Technique precision@1 | 77.8% | >= 70% |
| Technique hit@3 | 77.8% | |
| Acceptable technique among candidates | 88.9% | |
| CVE recall (alerts with a known CVE) | 0.0% | |
| Key facts in the prose (IPs, port) | 70.8% | |
| Facts block complete (attacker, target; built by code) | 100.0% | |
| Reports still inventing an IP after verify | 0.0% | 0% |
| Passed verification on the first draft | 22.2% | |
| Citation validity (after verify) | 100.0% | 100% |
| Invalid citations dropped per report | 0.00 | |
| Faithfulness (judge: supported claims) | 80.9% | |
| Tokens in / out per report | 3,346 / 776 | |
| LLM seconds per report | 152.1 | |

| Label | P@1 |
|---|---|
| Botnet | 0.0% |
| DDoS | 100.0% |
| DoS GoldenEye | 100.0% |
| DoS Hulk | 100.0% |
| DoS Slowhttptest | 100.0% |
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
