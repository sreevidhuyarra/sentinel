"""Convert a `# %%`-style Python script into a Jupyter notebook (.ipynb).

The Kaggle notebook is kept as notebooks/phishing_lora_kaggle.py so it can be linted and
smoke-tested locally; this builds the .ipynb that is uploaded to Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _cell(kind: str, lines: list[str]) -> dict[str, Any]:
    while lines and not lines[-1].strip():
        lines.pop()
    source = [ln + "\n" for ln in lines]
    if source:
        source[-1] = source[-1].rstrip("\n")
    cell: dict[str, Any] = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def script_to_notebook(src: Path, dst: Path) -> Path:
    cells: list[dict[str, Any]] = []
    kind: str | None = None
    buf: list[str] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append(_cell(kind, buf))
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
            continue
        if kind == "markdown":
            buf.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        elif kind == "code":
            buf.append(line)
    if kind is not None:
        cells.append(_cell(kind, buf))
    nb = {
        "cells": [c for c in cells if c["source"]],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return dst
