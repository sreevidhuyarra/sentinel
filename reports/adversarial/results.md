# Adversarial robustness results

Targets: 1,171 attack flows from the test split that the deployed ensemble catches (Bot 142, BruteForce 250, DDoS 250, DoS 250, Infiltration 8, PortScan 250, WebAttack 21). Evasion = share of the flows a detector caught that it scores Benign after the attack.

The attacker may change only its own padding, timing and TCP window (44 of 84 features; padding and delay only upwards), never destination port, protocol, flags, packet counts or the victim's replies.

## Clean test accuracy (full test split)

| Detector | Macro-F1 | Benign FPR | Attack recall |
|---|---|---|---|
| LightGBM (deployed member) | 0.9926 | 0.001% | 0.9958 |
| MLP (deployed member) | 0.9748 | 0.013% | 0.9712 |
| Ensemble (deployed) | 0.9957 | 0.001% | 0.9959 |
| MLP, adversarially trained | 0.9781 | 0.012% | 0.9864 |
| LightGBM without controllable features | 0.9654 | 0.002% | 0.9913 |
| LightGBM without timing features | 0.9804 | 0.001% | 0.9931 |

Review flag (disagreement > 0.001 or autoencoder anomaly) sends 1.4% of benign test flows to review (disagreement 0.7%, anomaly 1.0%).

## Feature-space attacks (upper bound)

Targeted towards Benign, L-inf budget in training standard deviations, ART FGSM / PGD (20 steps) with the controllable-feature mask and validity projection. White-box on the source MLP; every other column is a transfer attack.

| Attack | Source | Budget | MLP (deployed member) | LightGBM (deployed member) | Ensemble (deployed) | Ensemble + review flag | MLP, adversarially trained | LightGBM without timing features | LightGBM without controllable features |
|---|---|---|---|---|---|---|---|---|---|
| FGSM | MLP (deployed member) | 0.1 | 8.6% | 1.9% | 0.4% | 0.0% | 0.1% | 0.4% | 0.0% |
| FGSM | MLP (deployed member) | 0.25 | 32.1% | 12.6% | 8.7% | 0.1% | 5.5% | 2.1% | 0.0% |
| FGSM | MLP (deployed member) | 0.5 | 94.1% | 18.5% | 19.6% | 2.0% | 36.9% | 8.5% | 0.0% |
| FGSM | MLP (deployed member) | 1 | 92.2% | 46.8% | 57.6% | 0.0% | 93.7% | 43.3% | 0.0% |
| FGSM | MLP (deployed member) | 2 | 98.1% | 57.2% | 64.7% | 0.0% | 95.2% | 48.8% | 0.0% |
| FGSM | MLP, adversarially trained | 0.1 | 7.2% | 3.7% | 0.9% | 0.0% | 0.3% | 0.4% | 0.0% |
| FGSM | MLP, adversarially trained | 0.25 | 25.8% | 12.6% | 9.0% | 0.0% | 6.0% | 1.6% | 0.0% |
| FGSM | MLP, adversarially trained | 0.5 | 74.3% | 18.8% | 19.6% | 1.5% | 40.0% | 10.0% | 0.0% |
| FGSM | MLP, adversarially trained | 1 | 94.0% | 53.6% | 55.8% | 1.5% | 93.9% | 42.1% | 0.0% |
| FGSM | MLP, adversarially trained | 2 | 99.5% | 54.9% | 61.5% | 0.0% | 97.8% | 46.0% | 0.0% |
| PGD | MLP (deployed member) | 0.1 | 8.9% | 1.7% | 0.4% | 0.0% | 0.1% | 0.4% | 0.0% |
| PGD | MLP (deployed member) | 0.25 | 33.5% | 12.6% | 10.3% | 0.0% | 5.3% | 1.9% | 0.0% |
| PGD | MLP (deployed member) | 0.5 | 95.2% | 18.1% | 19.3% | 2.0% | 36.1% | 8.0% | 0.0% |
| PGD | MLP (deployed member) | 1 | 99.8% | 52.2% | 54.5% | 2.6% | 91.6% | 50.3% | 0.0% |
| PGD | MLP (deployed member) | 2 | 100.0% | 54.6% | 64.7% | 0.0% | 99.9% | 35.7% | 0.0% |
| PGD | MLP, adversarially trained | 0.1 | 7.7% | 1.4% | 0.9% | 0.0% | 0.3% | 0.3% | 0.0% |
| PGD | MLP, adversarially trained | 0.25 | 26.1% | 12.6% | 5.4% | 0.0% | 6.5% | 1.6% | 0.0% |
| PGD | MLP, adversarially trained | 0.5 | 77.5% | 18.3% | 18.5% | 2.0% | 44.7% | 10.5% | 0.0% |
| PGD | MLP, adversarially trained | 1 | 99.7% | 53.1% | 58.1% | 5.5% | 97.7% | 48.4% | 0.0% |
| PGD | MLP, adversarially trained | 2 | 100.0% | 53.6% | 56.0% | 0.0% | 99.8% | 53.3% | 0.0% |

## Black-box attack on LightGBM (HopSkipJump)

74 flows; an evading starting point was found for 67.6%. Success at any distance: 43.2%; median L-inf of successes 0.559.

| Budget | 0.1 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| Success | 14.9% | 14.9% | 20.3% | 41.9% | 43.2% |

## Problem-space attacks (realistic)

Every forward packet padded by up to `pad` of its payload and all timing stretched by up to `delay`, dependent features recomputed; the attacker tries every grid point inside the budget.

| Detector | low (pad <= 0.1, delay <= 1.5x) | medium (pad <= 0.5, delay <= 3x) | high (pad <= 1, delay <= 10x) |
|---|---|---|---|
| LightGBM (deployed member) | 9.7% | 12.9% | 20.9% |
| MLP (deployed member) | 4.4% | 11.2% | 23.2% |
| Ensemble (deployed) | 0.9% | 6.7% | 20.8% |
| Ensemble + review flag | 0.0% | 0.0% | 3.5% |
| MLP, adversarially trained | 0.1% | 4.8% | 15.0% |
| LightGBM without timing features | 0.2% | 0.4% | 9.4% |
| LightGBM without controllable features | 0.0% | 0.0% | 0.0% |

### By family (deployed ensemble, high budget)

| Family | Evaded | Padding only | Delay only | Evaded despite review flag |
|---|---|---|---|---|
| Bot | 94.4% | 0.0% | 0.0% | 0.0% |
| BruteForce | 0.8% | 0.8% | 0.0% | 0.0% |
| DDoS | 0.8% | 0.0% | 0.0% | 0.0% |
| DoS | 0.0% | 0.0% | 0.0% | 0.0% |
| Infiltration | 0.0% | 0.0% | 0.0% | 0.0% |
| PortScan | 41.2% | 0.0% | 41.2% | 16.4% |
| WebAttack | 14.3% | 14.3% | 14.3% | 0.0% |
