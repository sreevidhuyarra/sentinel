---
title: Sentinel
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Laptop SOC demo with ML alerts and a live phishing check
---

# Sentinel: public demo

A read-only demo of [Sentinel](https://github.com/sreevidhuyarra/sentinel), a Security
Operations Center that runs on a laptop: machine learning that detects network attacks and
phishing, an LLM copilot that writes cited incident reports, and the MLOps that keeps both
honest.

What runs here:

- **Dashboard** over a recorded replay of the CIC-IDS2017 test traffic: alerts with their
  explanations, related activity, model health and the robustness study. The live feed
  replays the recorded alerts, and the throughput chart shows a series measured during a
  real replay.
- **Copilot reports** that were generated earlier; live investigations call an LLM and are
  turned off here.
- **Phishing check, live:** paste an email (or upload an .eml) and see the verdict from the
  text model, the links, the sender, domain reputation and payment-request checks. Domains in
  the email are looked up in public blocklists and RDAP; nothing you paste is stored.

Turned off: analyst feedback, live investigations and live evasion runs. Scoring is limited
to 20 checks a minute per visitor.

Code, full results and how to run the whole system: https://github.com/sreevidhuyarra/sentinel
