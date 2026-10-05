# SOC copilot results

## Retrieval (no LLM)

105 gold alerts. Recall@k = an acceptable technique is among the top k candidates; MRR = mean reciprocal rank of the first acceptable one.

| Retriever | R@1 | R@3 | R@5 | R@8 | MRR | ms/query |
|---|---|---|---|---|---|---|
| bm25 | 45.7% | 79.0% | 80.0% | 81.0% | 0.619 | 25 |
| dense | 60.0% | 83.8% | 84.8% | 84.8% | 0.688 | 78 |
| hybrid | 75.2% | 87.6% | 87.6% | 87.6% | 0.811 | 135 |
| hybrid+rerank | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 1167 |
| hybrid+rerank, no family hints | 16.2% | 40.0% | 65.7% | 67.6% | 0.308 | 1001 |
| hybrid, no family hints | 21.9% | 32.4% | 60.0% | 67.6% | 0.325 | 77 |
| hybrid+rerank + behaviour | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 889 |
| hybrid+rerank, behaviour, no family hints | 16.2% | 40.0% | 65.7% | 67.6% | 0.308 | 875 |
| deployed: hybrid+rerank, behaviour candidates merged | 84.8% | 84.8% | 84.8% | 92.4% | 0.858 | 856 |

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
| Technique precision@1 | 83.3% | >= 70% |
| Technique hit@3 | 83.3% | |
| Acceptable technique among candidates | 88.9% | |
| CVE recall (alerts with a known CVE) | 0.0% | |
| Key facts in the prose (IPs, port) | 43.8% | |
| Facts block complete (attacker, target; built by code) | 100.0% | |
| Reports still inventing an IP after verify | 0.0% | 0% |
| Passed verification on the first draft | 100.0% | |
| Citation validity (after verify) | 100.0% | 100% |
| Invalid citations dropped per report | 0.00 | |
| Faithfulness (judge: supported claims) | 75.2% | |
| Tokens in / out per report | 1,805 / 473 | |
| LLM seconds per report | 106.0 | |

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
| Infiltration | 100.0% |
| Infiltration - Portscan | 100.0% |
| Portscan | 100.0% |
| SSH-Patator | 100.0% |
| Web Attack - Brute Force | 0.0% |
| Web Attack - SQL Injection | 100.0% |
| Web Attack - XSS | 100.0% |
| fraud | 100.0% |
| phishing | 100.0% |
