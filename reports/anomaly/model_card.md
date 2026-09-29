# Model card: anomaly-detector

## Intended use
Flag network flows that look unlike normal traffic, including attack types absent from the
supervised model's training data. Used behind the supervised classifier: a flow it calls
Benign but the autoencoder cannot reconstruct becomes an "Unknown anomaly" alert (severity
Low, or Medium at twice the threshold) for an analyst to triage.

## Model
Autoencoder (84 -> 64 -> 32 -> 8 -> 32 -> 64 -> 84, ReLU, linear code) on
standard-scaled FeatureSpec features, trained with MSE on benign flows only. Anomaly score =
mean squared reconstruction error. Threshold = the score exceeded by 1% of benign
validation flows. Explanations list the worst-reconstructed features with the observed value
and the value the model reconstructs for a normal-looking flow.

## Data
Benign flows of the corrected CIC-IDS2017 training split. No attack labels are used for
training; validation attacks are used only to choose the bottleneck size.

## Results (CIC-IDS2017 test split)
ROC-AUC 0.9430, benign FPR 1.02%, attack recall
33.22%. See results.md for per-family recall, the unknown-attack test and
the new-network test.

## Limitations
- **False alarms are the price.** At a 1% benign FPR the detector adds roughly one alert per
  hundred benign flows; the supervised model alone raises about one per hundred thousand.
  Anomaly alerts are therefore kept at Low / Medium severity.
- **Unusual is not malicious.** Rare but legitimate traffic (new services, backups, scans by
  IT) scores high; slow or low-volume attacks that resemble normal flows score low.
- **Learns one network's normal.** Trained on CIC-IDS2017 benign traffic; on another network
  it must at least be re-thresholded, and preferably retrained, on that network's benign
  traffic (see the new-network table).
