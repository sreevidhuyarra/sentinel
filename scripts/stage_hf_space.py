"""Assemble the Hugging Face Space for the public demo into build/hf-space.

    uv run sentinel demo snapshot                   # data/demo/sentinel_demo.sqlite
    cd frontend && npm run build && cd ..           # the Gradio Space serves the built dashboard
    uv run python scripts/stage_hf_space.py         # Gradio SDK (default; free hardware)
    uv run python scripts/stage_hf_space.py docker  # Docker SDK (paid on some accounts)

The folder is then pushed to the Space (see deploy/huggingface/PUBLISH.md). It holds only
what the demo serves: the code, the deployed model bundles, the sender / intel / payment
policies, the robustness study and the database snapshot. No secrets, no raw data.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf-space"
DEPLOY = ROOT / "deploy" / "huggingface"

FILES = [
    "pyproject.toml",
    "params.yaml",
    "LICENSE",
    "reports/adversarial/results.json",
    "models/phishing/sender_policy.json",
    "models/phishing/intel_policy.json",
    "models/phishing/payment_policy.json",
    "data/demo/sentinel_demo.sqlite",
    "data/demo/throughput.json",  # Prometheus series recorded during a real replay
]
DIRS = [
    "src",
    "models/ids",
    "models/anomaly",
    "models/phishing/bundle",
    "models/phishing/url_bundle",
]
# Per SDK: extra repository files and directories, and the deploy/huggingface/<sdk> files.
SDK = {
    # The Space installs requirements.txt and runs app.py; there is no Node, so the
    # dashboard ships prebuilt.
    "gradio": (["frontend/dist"], ["app.py", "README.md", "requirements.txt", "packages.txt"]),
    # The image builds the dashboard and installs from uv.lock itself.
    "docker": (["frontend", "uv.lock"], ["Dockerfile", "README.md"]),
}
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist", ".vite", "*.egg-info")
# Stored with Git LFS on the Hub (files over 10 MB must be; binaries should be).
LFS = ["*.onnx", "*.sqlite", "*.joblib", "*.pt", "models/**/*.txt", "tokenizer.json"]
NEVER = (".env", "phishing_pot", ".eml")


def main(sdk: str = "gradio") -> None:
    extra, deploy_files = SDK[sdk]
    missing = [p for p in FILES + DIRS + extra if not (ROOT / p).exists()]
    if missing:
        sys.exit(
            f"missing: {missing} (`sentinel demo snapshot` makes the database; "
            "`npm run build` in frontend/ makes frontend/dist)"
        )
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in FILES + [p for p in extra if (ROOT / p).is_file()]:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    for d in DIRS + [p for p in extra if (ROOT / p).is_dir()]:
        # frontend/dist is copied as built; everywhere else build output is skipped.
        ignore = None if d.endswith("dist") else SKIP
        shutil.copytree(ROOT / d, OUT / d, ignore=ignore)
    for f in deploy_files:
        shutil.copy2(DEPLOY / sdk / f, OUT / f)
    (OUT / ".gitattributes").write_text(
        "".join(f"{p} filter=lfs diff=lfs merge=lfs -text\n" for p in LFS), encoding="utf-8"
    )

    files = [p for p in OUT.rglob("*") if p.is_file()]
    leaked = [p for p in files if any(n in str(p) for n in NEVER)]
    if leaked:
        sys.exit(f"refusing: {leaked}")
    size = sum(p.stat().st_size for p in files)
    print(f"{OUT} ({sdk} Space): {len(files)} files, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:2])
