[← Back to the README](../README.md)

# Module 4: adversarial robustness

`uv run dvc repro -s robustness` (13 min on a laptop CPU) attacks the deployed network
detectors with the Adversarial Robustness Toolbox, trains three defenses and measures both.
Report: [reports/adversarial/results.md](../reports/adversarial/results.md).

**Threat model.** The attacker controls only their own side of the connection: timing,
their own packet sizes and TCP window, and the rates derived from them (44 of 84 features).
Padding and delay can only go up. The destination port, protocol, flags, packet counts and
the victim's replies stay fixed. Targets are 1,171 test attack flows that the deployed
ensemble catches (up to 250 per family).

Three attackers, strongest first:

1. **Feature space (upper bound).** ART FGSM and PGD (20 steps) with white-box access to the
   MLP, pushed towards Benign under an L-inf budget in training standard deviations, then
   transferred to every other detector.
2. **Black box.** HopSkipJump on the deployed LightGBM, starting from real benign flows.
3. **Problem space (realistic).** Pad every forward packet by up to `pad` and stretch all
   timing by up to `delay`, recompute every dependent feature, and try each grid point in
   the budget.

Defenses: an adversarially trained MLP (PGD on attack rows during training), LightGBM
without the controllable features, LightGBM without the timing features, and a **review
flag** that sends a flow to an analyst when LightGBM and the MLP disagree or the
autoencoder finds it anomalous.

| Detector | Clean macro-F1 | PGD evaded, budget 0.5 | Pad + delay evaded, high budget |
|---|---|---|---|
| Ensemble (deployed) | 0.9957 | 19% | 20.8% |
| Ensemble + review flag | 0.9957 | 2% | 3.5% |
| MLP (white-box target) | 0.9748 | 95% | 23.2% |
| MLP, adversarially trained (white-box target) | 0.9781 | 45% | 15.0% |
| LightGBM without timing features | 0.9804 | 8–11% | 9.4% |
| LightGBM without controllable features | 0.9654 | 0% | 0% |

### What we learned

- **The MLP is fragile, and attacks transfer.** White-box PGD evades 95% at a budget of 0.5.
  At 1.0 and above, the same perturbations also fool LightGBM and the ensemble about half
  the time, although neither was attacked directly.
- **Adversarial training helps only against small changes.** Evasion drops from 34% to 7%
  at a budget of 0.25 and from 95% to 45% at 0.5, but above 1.0 it is back to ~98%. It costs
  nothing on clean data: macro-F1 rises slightly (0.975 → 0.978).
- **The review flag is the most effective defense.** It keeps evasion at or below 5.5% in
  every feature-space attack and at 3.5% in the realistic high budget. The cost is 1.4% of
  benign test flows (~4,000) sent for review. Large perturbations make flows look unusual
  to the autoencoder and make the two models disagree.
- **Hardening trades accuracy for robustness.** Dropping every controllable feature makes the
  attack impossible by construction, but it costs 2.7 points of macro-F1. Dropping only
  timing halves realistic evasion (20.8% → 9.4%) for 1.2 points.
- **Realistic evasion is family-specific.** With the high budget, 94% of Bot flows evade the
  deployed ensemble, but only when padding and delay are combined (0% with either alone),
  and none evade with the review flag. 41% of PortScan evades with delay alone, and 16% gets
  past even the review flag, which makes it the weakest spot. DoS, DDoS, BruteForce and
  Infiltration stay caught (≤ 1%).
- **Black box works, but less well.** HopSkipJump finds an evasion for 43% of flows,
  20% within budget 0.5.
