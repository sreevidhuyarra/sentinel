"""Assemble the Hugging Face Space for the public demo into build/hf-space.

    uv run sentinel demo snapshot          # data/demo/sentinel_demo.sqlite from Postgres
    uv run python scripts/stage_hf_space.py
    docker build -t sentinel-demo build/hf-space && docker run -p 7860:7860 sentinel-demo

The folder is then pushed to a Docker Space (see deploy/huggingface/PUBLISH.md). It holds
only what the demo serves: the code, the deployed model bundles, the sender / intel /
payment policies, the robustness study and the database snapshot. No secrets, no raw data.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf-space"

FILES = [
    "pyproject.toml",
    "uv.lock",
    "params.yaml",
    "LICENSE",
    "reports/adversarial/results.json",
    "models/phishing/sender_policy.json",
    "models/phishing/intel_policy.json",
    "models/phishing/payment_policy.json",
    "data/demo/sentinel_demo.sqlite",
]
DIRS = [
    "src",
    "frontend",
    "models/ids",
    "models/anomaly",
    "models/phishing/bundle",
    "models/phishing/url_bundle",
]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist", ".vite", "*.egg-info")
# Stored with Git LFS on the Hub (files over 10 MB must be; binaries should be).
LFS = ["*.onnx", "*.sqlite", "*.joblib", "*.pt", "models/**/*.txt", "tokenizer.json"]
NEVER = (".env", "phishing_pot", ".eml")


def main() -> None:
    missing = [p for p in FILES + DIRS if not (ROOT / p).exists()]
    if missing:
        sys.exit(f"missing: {missing} (run `sentinel demo snapshot` for the database)")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=SKIP)
    for f in ("Dockerfile", "README.md"):
        shutil.copy2(ROOT / "deploy" / "huggingface" / f, OUT / f)
    (OUT / ".gitattributes").write_text(
        "".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in LFS), encoding="utf-8"
    )

    files = [p for p in OUT.rglob("*") if p.is_file()]
    leaked = [p for p in files if any(n in str(p) for n in NEVER)]
    if leaked:
        sys.exit(f"refusing: {leaked}")
    size = sum(p.stat().st_size for p in files)
    print(f"{OUT}: {len(files)} files, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
