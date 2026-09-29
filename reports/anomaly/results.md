# Anomaly detection results (CIC-IDS2017 test split)

Both detectors are trained on benign training flows only. Thresholds are set on benign validation flows. Autoencoder bottleneck: 8 (chosen on validation).

## Bottleneck sweep (validation)

| Bottleneck | Val ROC-AUC | Epochs | Val benign MSE | Seconds |
|---|---|---|---|---|
| 8 | 0.9436 | 30 | 0.0171 | 396 |
| 12 | 0.9327 | 30 | 0.0122 | 489 |
| 16 | 0.9400 | 30 | 0.0100 | 289 |

## Benign vs attack (test)

| Detector | ROC-AUC | PR-AUC | Benign FPR | Attack recall |
|---|---|---|---|---|
| autoencoder | 0.9430 | 0.7789 | 1.02% | 33.22% |
| iforest | 0.9184 | 0.6582 | 1.04% | 12.56% |

## Recall per family at the configured threshold (test)

| Family | Test flows | Autoencoder | Isolation Forest |
|---|---|---|---|
| DoS | 34,258 | 5.04% | 0.93% |
| DDoS | 19,009 | 79.80% | 0.00% |
| PortScan | 26,320 | 37.60% | 37.46% |
| BruteForce | 1,348 | 0.00% | 0.00% |
| WebAttack | 22 | 4.55% | 0.00% |
| Bot | 142 | 100.00% | 4.23% |
| Infiltration | 8 | 100.00% | 12.50% |

## False-alarm budget trade-off (thresholds from benign validation, scored on test)

| Target benign FPR | Autoencoder: benign FPR / attack recall | Isolation Forest: benign FPR / attack recall |
|---|---|---|
| 0.1% | 0.16% / 1.46% | 0.08% / 0.00% |
| 0.5% | 0.45% / 5.12% | 0.52% / 0.00% |
| 1.0% | 1.02% / 33.22% | 1.04% / 12.56% |
| 2.0% | 2.02% / 47.85% | 1.58% / 32.28% |

## Unknown-attack test: family removed from supervised training

LightGBM is retrained without the family; all of that family's flows are then scored. Recall = share flagged as any kind of alert.

| Held-out family | Flows | Supervised alone | Autoencoder alone | Isolation Forest alone | **Supervised + autoencoder** | Supervised + IF | Benign FPR sup. -> fused |
|---|---|---|---|---|---|---|---|
| Infiltration | 36 | 0.00% | 91.67% | 38.89% | **91.67%** | 38.89% | 0.00% -> 1.02% |
| Bot | 708 | 0.00% | 100.00% | 4.38% | **100.00%** | 4.38% | 0.00% -> 1.02% |
| WebAttack | 102 | 0.98% | 0.98% | 0.00% | **1.96%** | 0.98% | 0.00% -> 1.02% |

## Deployed supervised model + autoencoder (all families known, test)

| | Benign FPR | Benign false alarms | Attack recall |
|---|---|---|---|
| supervised | 0.00% | 3 | 99.59% |
| fused | 1.02% | 2,916 | 99.69% |

'Unknown anomaly' alerts added: 76 on attack flows, 2,913 on benign flows.

## New network: CIC-UNSW-NB15 (no UNSW attack labels used)

79 shared features, 521,219 UNSW test flows. For comparison, the supervised models transfer at ROC-AUC 0.55-0.68, and a supervised model trained with UNSW attack labels reaches 0.995 (reports/ids/cross_dataset.md).

| Detector | Needs from the new network | ROC-AUC | Benign FPR | Attack recall |
|---|---|---|---|---|
| CIC-trained autoencoder, as is | nothing | 0.3600 | 98.91% | 74.58% |
| CIC-trained autoencoder, re-thresholded | benign traffic | 0.3600 | 0.82% | 0.14% |
| Autoencoder retrained on UNSW benign | benign traffic | 0.9143 | 1.06% | 33.10% |
| Isolation Forest trained on UNSW benign | benign traffic | 0.9087 | 0.80% | 6.57% |

Recall per UNSW category (autoencoder retrained on UNSW benign):

| Category | Flows | Recall |
|---|---|---|
| Exploits | 6,165 | 53.03% |
| Fuzzers | 5,267 | 18.02% |
| Reconnaissance | 2,387 | 4.06% |
| Generic | 880 | 64.43% |
| DoS | 864 | 42.01% |
| Shellcode | 404 | 9.16% |
| Backdoor | 87 | 52.87% |
| Analysis | 75 | 26.67% |
| Worms | 47 | 12.77% |
