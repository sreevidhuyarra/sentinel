# URL classifier results

Character 3-5-gram TF-IDF on normalised URLs + 23 lexical features -> LightGBM. Trained on PhiUSIIL + Hannousse (145,556 train / 48,518 validation / 48,518 test URLs), split by registered domain, each dataset given equal total weight. Per-URL threshold 1.000: at most 1% of legitimate Hannousse validation URLs flagged (they have paths like real links; PhiUSIIL's legitimate URLs are all bare homepages).

| Test set | ROC-AUC | PR-AUC | F1 | False alarms on legitimate | Malicious caught |
|---|---|---|---|---|---|
| **LightGBM, source-balanced (deployed)** | 0.9986 | 0.9985 | 0.5121 | 0.00% | 34.42% |
|   of which Hannousse | 0.9793 | 0.9793 | 0.5571 | 0.09% | 38.64% |
|   of which PhiUSIIL | 0.9991 | 0.9992 | 0.5096 | 0.00% | 34.19% |
| LightGBM, unweighted (first attempt) | 0.9987 | 0.9986 | 0.6261 | 0.00% | 45.57% |
|   of which Hannousse | 0.9671 | 0.9668 | 0.4664 | 0.09% | 30.44% |
|   of which PhiUSIIL | 0.9994 | 0.9994 | 0.6339 | 0.00% | 46.40% |
| Logistic regression on n-grams (baseline) | 0.9267 | 0.9262 | 0.7486 | 0.85% | 60.52% |
| Cross-dataset: trained on PhiUSIIL only, scored on all Hannousse | 0.6611 | 0.6224 | 0.6574 | 99.49% | 99.81% |

## Combined verdict on the email test split

36,087 test emails, 15,491 with at least one URL; 35,407 URLs scored (first 20 per email). A link flags its email when its score reaches the email threshold, chosen on the email validation split so links add at most 0.5% of false alarms: none qualified, links not used.

| Verdict | False alarms on legitimate | Malicious caught | F1 | Phishing | Fraud | Spam |
|---|---|---|---|---|---|---|
| Text model alone | 0.31% | 89.66% | 0.9440 | 77.67% | 90.31% | 89.84% |
| Links alone (email threshold) | 0.00% | 0.00% | 0.0000 | 0.00% | 0.00% | 0.00% |
| **Text OR link (deployed)** | 0.31% | 89.66% | 0.9440 | 77.67% | 90.31% | 89.84% |
| Text OR link at the per-URL threshold (untuned) | 1.75% | 89.90% | 0.9386 | 79.94% | 90.46% | 90.06% |
| Unweighted URL model: text OR link, untuned | 1.53% | 89.96% | 0.9400 | 80.26% | 91.38% | 90.08% |

## Learned combination (fairer than OR)

Logistic regression over the text logit, the best link's logit and has-link, fitted on half of the validation emails; thresholds for it and for text alone set on the other half at each false-alarm budget, then scored on test. Weights: text 0.54, link 0.14, has-link 0.42.

| Validation budget | Method | Test false alarms | Malicious caught | Phishing caught |
|---|---|---|---|---|
| 0.5% | text only | 0.09% | 84.11% | 65.70% |
| 0.5% | learned text + link | 0.15% | 86.50% | 68.93% |
| 1.0% | text only | 0.14% | 85.48% | 68.28% |
| 1.0% | learned text + link | 0.20% | 87.73% | 69.90% |
| 2.0% | text only | 0.29% | 89.43% | 77.67% |
| 2.0% | learned text + link | 0.59% | 90.70% | 77.99% |
