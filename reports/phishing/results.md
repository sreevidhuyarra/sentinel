# Phishing email classifier results

DeBERTa-v3-small fine-tuned with LoRA on Kaggle (68 min on Tesla T4, Tesla T4). Label 1 = phishing, fraud or spam; 0 = legitimate. Full-split numbers come from the Kaggle run's predictions.

Alert threshold 0.989: the score exceeded by 2% of legitimate validation emails.

## Test split (36,087 emails)

| Threshold | F1 | Precision | Recall | False alarms on legitimate | Phishing caught | Fraud | Spam |
|---|---|---|---|---|---|---|---|
| 0.5 (as trained) | 0.9695 | 0.9626 | 0.9764 | 3.99% | 93.20% | 99.38% | 97.66% |
| 0.989 (deployed) | 0.9440 | 0.9967 | 0.8966 | 0.31% | 77.67% | 90.31% | 89.84% |

Threshold-free: PR-AUC 0.9971, ROC-AUC 0.9968.

## Per source corpus (test, deployed threshold)

| Corpus | Emails | False alarms on legitimate | Malicious caught |
|---|---|---|---|
| CEAS_08 | 6,882 | 0.15% | 92.26% |
| Enron | 5,880 | 0.03% | 90.17% |
| Ling | 571 | 0.00% | 68.13% |
| Nazario | 309 | - | 77.67% |
| Nazario_5 | 240 | 1.25% | - |
| Nigerian_5 | 891 | 1.24% | 90.31% |
| SpamAssasin | 1,034 | 0.00% | 65.18% |
| TREC_05 | 6,767 | 0.31% | 89.14% |
| TREC_06 | 3,118 | 0.21% | 90.51% |
| TREC_07 | 10,395 | 0.65% | 90.50% |

## Before and after optimisation (1,000 test emails, laptop CPU)

| Model | Size | F1 | ROC-AUC | Verdicts differing from PyTorch | Latency p50 / p95 (1 email) | Emails/s (batched) |
|---|---|---|---|---|---|---|
| PyTorch, float32 (reference) | - | 0.9569 | 0.9981 | - | 204 / 252 ms | 6.1 |
| ONNX, float32 | 568 MB | 0.9569 | 0.9981 | 0 | 130 / 161 ms | 7.9 |
| ONNX, int8 embeddings only **(deployed)** | 273 MB | 0.9568 | 0.9981 | 2 | 129 / 158 ms | 7.8 |
| ONNX, int8 everything | 165 MB | 0.8724 | 0.9942 | 85 | 87 / 113 ms | 12.0 |
