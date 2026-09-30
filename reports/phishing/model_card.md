# Model card: phishing-classifier

## Intended use
Score an email (subject + body) for phishing, fraud or spam so a SOC analyst can triage it.
It flags for review; it does not quarantine or delete mail on its own.

## Model
DeBERTa-v3-small (141M parameters) fine-tuned with LoRA (rank 16 on the attention
query/key/value projections; classification head and pooler trained in full; 0.72% of
weights trainable). Merged and exported to ONNX; onnx int8 embeddings deployed
(273 MB). Input: subject + body, HTML removed, URLs replaced by [URL],
first 256 tokens. Alert when P(malicious) >= 0.989.

## Data
Phishing Email Curated Datasets (Champa, Rabbi & Zibran 2024; Zenodo 8339691; CC BY 4.0):
11 public corpora, 200,517 emails after deduplication, split by sender domain / subject
campaign so no sender or campaign appears in two splits.

## Results (test split, deployed threshold)
F1 0.9440, precision 0.9967, recall 0.8966, false alarms on
legitimate mail 0.31%; PR-AUC 0.9971. Phishing caught
77.67%, fraud 90.31%, spam
89.84%.

## Limitations
- **Mostly spam, little phishing.** Of ~89k malicious training emails only ~900 are true
  phishing (Nazario corpus); phishing recall is the weakest per-kind number.
- **Old mail.** The corpora span 1995-2022, much of it 2002-2008. Modern phishing
  (QR codes, MFA-fatigue lures, LLM-written text) is under-represented.
- **Corpus effects.** False alarms concentrate in some corpora (TREC), whose "legitimate"
  mail includes newsletters and bulk mail; the model partly learns corpus style.
- **Text only.** URLs are replaced by [URL]; link reputation needs the separate URL model.
- **Threshold trade-off.** At 0.5 it catches more (97.64%) but
  flags 3.99% of legitimate mail.
