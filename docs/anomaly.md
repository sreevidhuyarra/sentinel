[← Back to the README](../README.md)

# Module 2: anomaly detection

A supervised model only recognises what it was trained on. Module 2 learns what *normal*
traffic looks like instead, so it can flag attacks nobody has labelled yet.

**Pipeline.** An autoencoder (84 → 64 → 32 → 8 → 32 → 64 → 84) is trained to reconstruct
benign training flows only; a flow it cannot reconstruct scores high. The code size (8) was
picked from {8, 12, 16} on validation, and the threshold is the score exceeded by 1% of benign
validation flows. An Isolation Forest is the baseline. In the deployed detector the supervised
verdict stands when it raises an alert; when it says Benign but the autoencoder fires, the flow
becomes an **"Unknown anomaly"** alert (severity Low, or Medium at twice the threshold) whose
reasons name the features that looked wrong and what a normal-looking flow would have, e.g.
*"forward header length = 18,868 is unusual for normal traffic (normal-looking value ~ 860)"*
(a Heartbleed flow the supervised model scored 0.3% likely to be an attack).
`uv run sentinel anomaly train` (or `dvc repro`) trains it and runs every study below;
full tables: [reports/anomaly/results.md](../reports/anomaly/results.md).

### Results (CIC-IDS2017 test split)

| Detector | ROC-AUC | Attacks caught at 1% benign FPR |
|---|---|---|
| Isolation Forest (baseline) | 0.918 | 12.6% |
| **Autoencoder** | **0.943** | **33.2%** |

**Unknown-attack test.** Each family was removed from supervised training in turn; all of its
flows were then scored:

| Held-out family | Flows | Supervised model | Autoencoder | Isolation Forest | Supervised + autoencoder |
|---|---|---|---|---|---|
| Infiltration | 36 | 0% | 92% | 39% | **92%** |
| Bot | 708 | 0% | 100% | 4% | **100%** |
| WebAttack | 102 | 1% | 1% | 0% | **2%** |

Benign false alarms go from 0.001% (supervised alone) to 1.0% with the autoencoder added.

**New network, no attack labels** (CIC-UNSW-NB15, 521k test flows):

| Detector | Needs from the new network | ROC-AUC |
|---|---|---|
| Supervised models trained on CIC-IDS2017 (Module 1) | nothing | 0.55–0.68 |
| Autoencoder trained on CIC-IDS2017, as is or re-thresholded | nothing / benign traffic | 0.36 |
| **Autoencoder retrained on UNSW benign traffic** | **benign traffic only** | **0.914** |
| Supervised model trained on UNSW (reference) | labelled attacks | 0.995 |

### What we learned

- **It catches unknown attacks that behave differently.** Bot and Infiltration, invisible to a
  supervised model that never saw them, are caught 92–100%. The autoencoder beats the
  Isolation Forest everywhere that matters.
- **It cannot see attacks that hide in content.** Web attacks (brute-force logins, XSS, SQL
  injection) are ordinary HTTP connections at the flow level; nothing here catches them
  unseen. Payload-level detectors (like Module 3's phishing model) cover that.
- **Its price is false alarms.** With all families known, it lifts attack recall only from
  99.59% to 99.69% while adding ~2,900 false alarms on 287k benign test flows (1%). That is why
  anomaly alerts stay Low/Medium and form their own queue. A stricter budget helps little:
  at 0.1% FPR it catches 1.5% of attacks (trade-off table in the report).
- **"Normal" is local.** Moved to UNSW-NB15 unchanged, the CIC-trained autoencoder scores
  below chance (0.36): UNSW's normal traffic looks stranger to it than UNSW's attacks. Retrained
  on UNSW's benign traffic alone, which needs no labelling, it reaches 0.914. On a new network,
  this is the detector you can have on day one.
