from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sentinel.common.config import Params, load_params
from sentinel.data.synthetic import generate


@pytest.fixture(scope="session")
def synth_params(tmp_path_factory: pytest.TempPathFactory) -> Params:
    """Params pointing every data directory at a temp dir holding the synthetic CSVs."""
    root = tmp_path_factory.mktemp("sentinel")
    generate(root / "raw", seed=0)
    cfg = {
        "seed": 42,
        "data": {
            k: str(root / k.removesuffix("_dir"))
            for k in ("raw_dir", "interim_dir", "processed_dir", "reference_dir", "reports_dir")
        },
    }
    path = root / "params.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_params(path)


@pytest.fixture(scope="session")
def built(synth_params: Params) -> Params:
    from sentinel.data.ingest import ingest
    from sentinel.data.pipeline import build

    ingest(Path(synth_params.data.raw_dir), Path(synth_params.data.interim_dir) / "flows.parquet")
    build(synth_params)
    return synth_params
