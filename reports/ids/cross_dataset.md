# Cross-dataset test: CIC-IDS2017 -> UNSW-NB15

Models trained on CIC-IDS2017 (79 shared CICFlowMeter features; dropped bwd_rst_flags, fwd_rst_flags, icmp_code, icmp_type, total_tcp_flow_time) and applied unchanged to CIC-UNSW-NB15. Threshold and calibration come from CIC-IDS2017 validation.

## Benign vs attack

| Model | CIC test macro-F1 | UNSW benign FPR | UNSW attack recall | UNSW precision | UNSW ROC-AUC | UNSW PR-AUC | Recall @1% FPR (UNSW-chosen threshold) |
|---|---|---|---|---|---|---|---|
| lightgbm | 0.9806 | 0.85% | 3.05% | 10.41% | 0.5699 | 0.0495 | 3.09% |
| xgboost | 0.9982 | 1.28% | 5.64% | 12.48% | 0.6768 | 0.0628 | 5.28% |
| mlp | 0.9473 | 5.77% | 8.63% | 4.60% | 0.5515 | 0.0442 | 2.63% |
| ensemble | 0.9694 | 3.33% | 4.60% | 4.27% | 0.5765 | 0.0453 | 2.63% |
| *UNSW-trained reference* | - | 1.13% | 87.36% | - | 0.9953 | 0.8479 | - |

## Ensemble detection rate per UNSW-NB15 category

| UNSW category | Flows | Flagged as attack (CIC-trained) | Flagged (UNSW-trained reference) | Most common predicted family |
|---|---|---|---|---|
| Benign | 2,525,314 | 3.33% | - | - |
| Exploits | 30,925 | 5.23% | 95.64% | Benign |
| Fuzzers | 26,433 | 6.17% | 71.14% | Benign |
| Reconnaissance | 12,035 | 0.02% (0% of detections as_PortScan) | 99.96% | Benign |
| Generic | 4,496 | 2.96% | 97.16% | Benign |
| DoS | 4,418 | 7.33% (98% of detections as_DoS) | 94.10% | Benign |
| Shellcode | 2,102 | 0.10% | 72.28% | Benign |
| Backdoor | 452 | 1.55% | 91.95% | Benign |
| Analysis | 385 | 0.00% | 21.33% | Benign |
| Worms | 243 | 11.93% | 100.00% | Benign |

## Where the shift comes from (benign traffic, top LightGBM features)

PSI above 0.25 is conventionally a major distribution shift.

| Feature | Share of gain | PSI (benign CIC vs UNSW) | Median CIC | Median UNSW |
|---|---|---|---|---|
| `rst_flag_count` | 20.5% | 0.64 | 0 | 0 |
| `flow_duration` | 18.5% | 2.04 | 9.68e+04 | 1.346e+04 |
| `bwd_packet_length_max` | 10.4% | 2.97 | 134 | 192 |
| `bwd_packet_length_std` | 9.3% | 0.32 | 0 | 65.52 |
| `total_length_of_fwd_packet` | 6.0% | 2.78 | 90 | 454 |
| `bwd_init_win_bytes` | 5.8% | 4.29 | 0 | 1.448e+04 |
| `dst_port` | 4.7% | 2.17 | 53 | 1.435e+04 |
| `bwd_psh_flags` | 3.6% | 2.66 | 0 | 0 |
| `fwd_bwd_bytes_ratio` | 3.2% | 2.35 | 0.3939 | 0.01496 |
| `fwd_packet_length_max` | 3.1% | 2.92 | 48 | 66 |
| `packet_length_std` | 1.9% | 2.16 | 55.43 | 101.3 |
| `bwd_bulk_rate_avg` | 1.6% | 0.75 | 0 | 2.24e+04 |
| `bwd_iat_mean` | 1.4% | 3.19 | 48 | 392.7 |
| `flow_iat_max` | 1.3% | 3.09 | 6.635e+04 | 1,995 |
| `fwd_act_data_pkts` | 1.0% | 1.23 | 1 | 3 |
