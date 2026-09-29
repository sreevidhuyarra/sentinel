# Model card: ids-classifier

## Intended use
Classify network flows (CICFlowMeter features) as Benign or one of seven attack families
(DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration) to raise explained alerts
for a SOC analyst. It recommends; a human decides. Not for automated blocking.

## Model
Weighted average of a LightGBM classifier (weight 0.7) and a PyTorch MLP,
each calibrated with temperature + per-class bias on validation. A flow is an alert when
P(attack) >= 0.490, the threshold with the best validation macro-F1 at
benign FPR <= 1%. Explanations are TreeSHAP values from the LightGBM member.

## Data
Corrected CIC-IDS2017 (Engelen et al., DistriNet), 5 days of lab traffic, exact duplicates
removed. Split in time within every (day, label): train 1,103,765,
validation 367,926, test 367,940 flows. "Attempted" attacks
(no payload delivered) are labelled Benign; "Infiltration - Portscan" is labelled PortScan.

## Results (test split)
| Model | Macro-F1 | Benign FPR | Attack recall | PR-AUC (macro) | ECE | Train (s) | Latency (ms / 1k flows) |
|---|---|---|---|---|---|---|---|
| logreg | 0.9524 | 0.1764% | 0.9922 | 0.9480 | 0.0035 | 47 | 0.7 |
| lightgbm | 0.9926 | 0.0010% | 0.9958 | 0.9999 | 0.0008 | 152 | 18.6 |
| xgboost | 0.9987 | 0.0021% | 0.9963 | 0.9999 | 0.0008 | 585 | 8.7 |
| mlp | 0.9748 | 0.0126% | 0.9712 | 0.9870 | 0.0012 | 167 | 1.1 |
| ensemble | 0.9957 | 0.0010% | 0.9959 | 0.9999 | 0.0010 | 319 | 17.5 |

| Family | Precision | Recall | F1 | PR-AUC | Test flows |
|---|---|---|---|---|---|
| Benign | 0.9989 | 1.0000 | 0.9994 | 1.0000 | 286,833 |
| DoS | 0.9999 | 0.9942 | 0.9970 | 1.0000 | 34,258 |
| DDoS | 1.0000 | 0.9989 | 0.9994 | 0.9997 | 19,009 |
| PortScan | 1.0000 | 0.9965 | 0.9982 | 0.9998 | 26,320 |
| BruteForce | 1.0000 | 0.9889 | 0.9944 | 0.9999 | 1,348 |
| WebAttack | 1.0000 | 0.9545 | 0.9767 | 1.0000 | 22 |
| Bot | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 142 |
| Infiltration | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 8 |


## What the model relies on (top mean |SHAP| per family)
| Family | Top features |
|---|---|
| Benign | `rst_flag_count`, `fwd_bwd_bytes_ratio`, `bwd_packet_length_std` |
| DoS | `bwd_init_win_bytes`, `bwd_packet_length_std`, `fwd_packet_length_std` |
| DDoS | `bwd_packet_length_max`, `total_length_of_fwd_packet`, `fwd_packet_length_max` |
| PortScan | `flow_duration`, `dst_port`, `fwd_packet_length_max` |
| BruteForce | `dst_port`, `down_up_ratio`, `fwd_bwd_bytes_ratio` |
| WebAttack | `fwd_act_data_pkts`, `fwd_packet_length_std`, `fwd_seg_size_min` |
| Bot | `fwd_bwd_bytes_ratio`, `fwd_act_data_pkts`, `dst_port` |
| Infiltration | `dst_port`, `fwd_bwd_bytes_ratio`, `total_tcp_flow_time` |

## Limitations
- **One lab network; does not transfer.** All training traffic comes from one 2017
  testbed. Applied unchanged to UNSW-NB15 (`sentinel ids cross-dataset`), models trained
  here detect under 10% of attacks with ROC-AUC 0.55-0.68, while a model trained on
  UNSW-NB15 itself reaches 0.995 (reports/ids/cross_dataset.md). Retrain or fine-tune on
  the target network before use.
- **Possible testbed shortcuts.** TCP initial window sizes (`bwd_init_win_bytes`,
  `fwd_init_win_bytes`) and destination port carry much of the signal for some families;
  these can reflect the lab's specific hosts rather than attack behaviour. Module 4 tests
  models trained without them.
- **Tiny rare-class test sets.** WebAttack, Infiltration have fewer than 100 test flows,
  so their recall can move by several points from one or two flows.
- **Known attacks only.** Families absent from training are not recognised; the anomaly
  detector (Module 2) covers novel behaviour.
- **Not adversarially hardened yet** (Module 4).
