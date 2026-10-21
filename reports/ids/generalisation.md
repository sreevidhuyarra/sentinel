# Generalisation: training on one or two networks, testing on another

Benign vs attack, LightGBM. Thresholds come from the training networks' validation split (1% false alarms there); nothing is chosen on a target. Cells on the diagonal (a network's own test split) are in-domain; the rest are unseen networks.

| Network | Flows used | Attack share | Test flows |
|---|---|---|---|
| CIC-IDS2017 | 1,839,631 | 22.0% | 367,940 |
| UNSW-NB15 | 2,606,079 | 3.1% | 521,219 |
| CSE-CIC-IDS2018 | 2,018,324 | 38.3% | 403,674 |

## All shared features (79 features)

ROC-AUC (attack recall at the training threshold, false alarms) per test network.

| Trained on | CIC-IDS2017 | UNSW-NB15 | CSE-CIC-IDS2018 |
|---|---|---|---|
| CIC-IDS2017 | 1.000 (100% caught, 0.8% FA) | **0.637 (99% caught, 95.0% FA)** | **0.826 (64% caught, 12.4% FA)** |
| UNSW-NB15 | **0.310 (1% caught, 18.3% FA)** | 0.995 (87% caught, 1.2% FA) | **0.242 (1% caught, 9.4% FA)** |
| CSE-CIC-IDS2018 | **0.971 (100% caught, 22.6% FA)** | **0.202 (29% caught, 84.7% FA)** | 1.000 (100% caught, 1.2% FA) |
| UNSW-NB15 + CSE-CIC-IDS2018 | **0.942 (23% caught, 0.6% FA)** | 0.995 (94% caught, 2.1% FA) | 1.000 (100% caught, 0.0% FA) |
| CIC-IDS2017 + CSE-CIC-IDS2018 | 1.000 (100% caught, 0.9% FA) | **0.490 (81% caught, 83.9% FA)** | 1.000 (100% caught, 0.8% FA) |
| CIC-IDS2017 + UNSW-NB15 | 1.000 (99% caught, 0.0% FA) | 0.995 (94% caught, 2.1% FA) | **0.745 (2% caught, 0.6% FA)** |

## Without the shortcut features (75 features)

ROC-AUC (attack recall at the training threshold, false alarms) per test network.

| Trained on | CIC-IDS2017 | UNSW-NB15 | CSE-CIC-IDS2018 |
|---|---|---|---|
| CIC-IDS2017 | 1.000 (100% caught, 0.7% FA) | **0.712 (82% caught, 58.6% FA)** | **0.743 (42% caught, 5.3% FA)** |
| UNSW-NB15 | **0.424 (1% caught, 3.7% FA)** | 0.991 (69% caught, 1.1% FA) | **0.367 (0% caught, 0.7% FA)** |
| CSE-CIC-IDS2018 | **0.985 (99% caught, 9.2% FA)** | **0.452 (36% caught, 52.2% FA)** | 1.000 (100% caught, 0.9% FA) |
| UNSW-NB15 + CSE-CIC-IDS2018 | **0.977 (34% caught, 0.2% FA)** | 0.991 (86% caught, 2.1% FA) | 1.000 (99% caught, 0.0% FA) |
| CIC-IDS2017 + CSE-CIC-IDS2018 | 1.000 (100% caught, 0.8% FA) | **0.699 (60% caught, 10.9% FA)** | 1.000 (100% caught, 1.0% FA) |
| CIC-IDS2017 + UNSW-NB15 | 1.000 (99% caught, 0.1% FA) | 0.991 (80% caught, 1.7% FA) | **0.832 (2% caught, 0.1% FA)** |

Bold: a network the model never saw. FA: share of that network's benign test flows flagged.

## Effect of removing the shortcut features (ROC-AUC on the unseen network)

| Trained on | Unseen network | All shared | Without shortcuts | Change |
|---|---|---|---|---|
| CIC-IDS2017 | UNSW-NB15 | 0.637 | 0.712 | +0.074 |
| CIC-IDS2017 | CSE-CIC-IDS2018 | 0.826 | 0.743 | -0.083 |
| UNSW-NB15 | CIC-IDS2017 | 0.310 | 0.424 | +0.114 |
| UNSW-NB15 | CSE-CIC-IDS2018 | 0.242 | 0.367 | +0.125 |
| CSE-CIC-IDS2018 | CIC-IDS2017 | 0.971 | 0.985 | +0.013 |
| CSE-CIC-IDS2018 | UNSW-NB15 | 0.202 | 0.452 | +0.249 |
| UNSW-NB15 + CSE-CIC-IDS2018 | CIC-IDS2017 | 0.942 | 0.977 | +0.036 |
| CIC-IDS2017 + CSE-CIC-IDS2018 | UNSW-NB15 | 0.490 | 0.699 | +0.209 |
| CIC-IDS2017 + UNSW-NB15 | CSE-CIC-IDS2018 | 0.745 | 0.832 | +0.087 |

## What the models rely on (top features by gain share)

| Features | Trained on | Top features |
|---|---|---|
| shared | CIC-IDS2017 | `rst_flag_count` 58%, `fwd_bwd_bytes_ratio` 13%, `psh_flag_count` 8%, `dst_port` 5%, `bwd_packet_length_std` 3% |
| shared | UNSW-NB15 | `fwd_seg_size_min` 69%, `bwd_packet_length_min` 13%, `bwd_init_win_bytes` 10%, `fwd_packet_length_min` 1%, `bwd_packets_s` 1% |
| shared | CSE-CIC-IDS2018 | `fwd_init_win_bytes` 32%, `dst_port` 18%, `fwd_packet_length_mean` 13%, `fwd_seg_size_min` 6%, `ack_flag_count` 4% |
| shared | UNSW-NB15 + CSE-CIC-IDS2018 | `fwd_init_win_bytes` 33%, `total_fwd_packet` 9%, `fwd_packet_length_max` 7%, `bwd_packet_length_mean` 6%, `bwd_init_win_bytes` 6% |
| shared | CIC-IDS2017 + CSE-CIC-IDS2018 | `syn_flag_count` 27%, `fwd_act_data_pkts` 17%, `dst_port` 12%, `psh_flag_count` 8%, `flow_duration` 6% |
| shared | CIC-IDS2017 + UNSW-NB15 | `down_up_ratio` 23%, `rst_flag_count` 18%, `bwd_init_win_bytes` 12%, `psh_flag_count` 9%, `fwd_init_win_bytes` 5% |
| no_shortcuts | CIC-IDS2017 | `rst_flag_count` 62%, `fwd_bwd_bytes_ratio` 10%, `psh_flag_count` 7%, `bwd_packet_length_std` 3%, `flow_duration` 3% |
| no_shortcuts | UNSW-NB15 | `bwd_packets_s` 47%, `flow_iat_mean` 13%, `psh_flag_count` 11%, `fin_flag_count` 7%, `packet_length_max` 4% |
| no_shortcuts | CSE-CIC-IDS2018 | `fin_flag_count` 18%, `packet_length_max` 16%, `total_length_of_bwd_packet` 12%, `bwd_packet_length_max` 7%, `bwd_iat_max` 6% |
| no_shortcuts | UNSW-NB15 + CSE-CIC-IDS2018 | `psh_flag_count` 23%, `fwd_iat_std` 16%, `fin_flag_count` 6%, `rst_flag_count` 6%, `fwd_psh_flags` 4% |
| no_shortcuts | CIC-IDS2017 + CSE-CIC-IDS2018 | `syn_flag_count` 27%, `fwd_act_data_pkts` 16%, `fwd_psh_flags` 10%, `psh_flag_count` 9%, `bwd_packet_length_min` 5% |
| no_shortcuts | CIC-IDS2017 + UNSW-NB15 | `down_up_ratio` 22%, `rst_flag_count` 17%, `psh_flag_count` 12%, `fin_flag_count` 7%, `flow_duration` 4% |
