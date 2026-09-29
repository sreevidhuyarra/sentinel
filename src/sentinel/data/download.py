"""Fetch the corrected CIC-IDS2017 release (Engelen et al., DistriNet, KU Leuven)."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from sentinel.common.logging import get_logger

log = get_logger(__name__)

CICIDS2017_IMPROVED_URL = (
    "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
)


# CIC-UNSW-NB15 (UNSW-NB15 re-extracted with CICFlowMeter by CIC, 2024). The official
# download (unb.ca/cic/datasets/cic-unsw-nb15.html) sits behind a registration form;
# this public Hugging Face mirror hosts the same files.
CIC_UNSW_NB15_URL = (
    "https://huggingface.co/datasets/bencorn/CIC-UNSW-NB15/resolve/main/CICFlowMeter_out.csv"
)


def download_cic_unsw_nb15(raw_dir: Path, url: str = CIC_UNSW_NB15_URL) -> Path:
    target = raw_dir / "CICFlowMeter_out.csv"
    if target.exists():
        log.info("%s already present; skipping download", target)
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    log.info("downloading %s (~1.8 GB)", url)
    with urllib.request.urlopen(url) as resp, partial.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    partial.rename(target)
    return target


def download_cicids2017(raw_dir: Path, url: str = CICIDS2017_IMPROVED_URL) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(raw_dir.glob("*.csv"))
    if existing:
        log.info("%d CSVs already in %s; skipping download", len(existing), raw_dir)
        return existing
    zip_path = raw_dir.parent / "CICIDS2017_improved.zip"
    log.info("downloading %s (~330 MB)", url)
    with urllib.request.urlopen(url) as resp, zip_path.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv"):
                target = raw_dir / Path(member).name
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
    zip_path.unlink()
    return sorted(raw_dir.glob("*.csv"))
