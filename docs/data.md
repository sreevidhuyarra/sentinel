[← Back to the README](../README.md)

# Data

Primary dataset: the **corrected CIC-IDS2017** release by Engelen et al. (DistriNet, KU Leuven),
which fixes labelling and feature-extraction errors in the original.

| | |
|---|---|
| Raw flows | 2,099,976 across 5 days |
| After removing exact duplicates | 1,841,347 |
| Model features | 84 (82 CICFlowMeter + 2 derived ratios) |
| Families | Benign, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration |

Decisions worth knowing:

- **Attempted attacks** (`X - Attempted`: the attack tool connected but sent no malicious
  payload) are labelled Benign by default, since their traffic is indistinguishable from benign.
  Configurable via `data.attempted_policy` in `params.yaml`.
- **"Infiltration - Portscan"** (~72k flows, a scan launched from the compromised host) is
  mapped to PortScan; true Infiltration is only 36 flows.
- **No random splits.** Each attack in CIC-IDS2017 runs in one contiguous time window, and
  consecutive flows of a session are near-identical. Flows are ordered in time within every
  (day, label) group; the earliest 60% go to train, the next 20% to validation, the latest 20%
  to test, with a small gap dropped at each boundary. A held-out-family split (for the anomaly
  detector) and a random split (to show the leakage it causes) are also available.
- Identifiers (IPs, source port, flow ID, timestamps) are never model inputs.

Modern phishing for the sender checks: [Phishing Pot](https://github.com/rf-peixoto/phishing_pot)
by rf-peixoto, CC BY-NC 4.0 (non-commercial use with attribution). It contains real phishing,
so it is downloaded locally and never committed:

```bash
curl -L -o data/raw/phishing_pot/phishing_pot-2933c03.zip \
  https://codeload.github.com/rf-peixoto/phishing_pot/zip/2933c0341b1248678c1e246b36d352add2edf0f1
```
