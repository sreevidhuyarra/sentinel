import json
from pathlib import Path

import polars as pl
import pytest
import yaml

from sentinel.common.config import load_params
from sentinel.phishing.data import build, kind_of
from sentinel.phishing.notebook import script_to_notebook
from sentinel.phishing.text import (
    URL_TOKEN,
    clean_text,
    dedup_key,
    extract_urls,
    model_input,
    sender_domain,
)


def test_clean_text_strips_html_and_replaces_urls() -> None:
    raw = (
        "<html><style>p{color:red}</style><p>Dear&nbsp;user,</p>"
        '<a href="http://evil.example/login">Click</a> or visit https://paypa1.com/verify.</html>'
    )
    out = clean_text(raw)
    assert "<" not in out and "color:red" not in out
    assert "Dear user," in out
    assert out.count(URL_TOKEN) == 1  # the href URL is inside a tag and removed with it
    assert "paypa1.com" not in out


def test_extract_urls_includes_hrefs_without_duplicates() -> None:
    raw = '<a href="http://evil.example/login">x</a> http://evil.example/login www.b.org/x).'
    assert extract_urls(raw) == ["http://evil.example/login", "www.b.org/x"]


def test_model_input_joins_subject_and_truncated_body() -> None:
    assert model_input("Hi", "a" * 50, max_chars=10) == "Hi\n\n" + "a" * 10
    assert model_input(None, None) == ""


@pytest.mark.parametrize(
    ("sender", "domain"),
    [
        ("Bob <bob@mail.PayPal.com>", "paypal.com"),
        ("alice@shop.example.co.uk", "example.co.uk"),
        ("no address here", None),
        (None, None),
    ],
)
def test_sender_domain(sender: str | None, domain: str | None) -> None:
    assert sender_domain(sender) == domain


def test_dedup_key_ignores_case_and_spacing() -> None:
    assert dedup_key("Hello  World\n") == dedup_key("hello world")


@pytest.mark.parametrize(
    ("source", "label", "kind"),
    [
        ("Nazario_5", 1, "phishing"),
        ("Nigerian_Fraud", 1, "fraud"),
        ("CEAS_08", 1, "spam"),
        ("Nazario", 0, "legitimate"),
    ],
)
def test_kind_of(source: str, label: int, kind: str) -> None:
    assert kind_of(source, label) == kind


def _write_corpora(raw: Path) -> None:
    raw.mkdir(parents=True)
    rows = []
    for i in range(60):
        dom = f"d{i % 12}.com"
        rows.append(
            {
                "sender": f"x@{dom}",
                "date": "Fri, 29 Jun 2001 08:36:09 -0500",
                "subject": f"Offer {i}",
                "body": f"Body number {i} with enough text http://s{i}.com",
                "label": str(i % 2),
                "urls": "1",
            }
        )
    rows.append(dict(rows[0]))  # exact duplicate
    rows.append({**rows[1], "label": "0"})  # same text, conflicting label
    rows.append({**rows[1], "sender": "y@d1.com"})  # duplicate of the conflicted text
    pl.DataFrame(rows).write_csv(raw / "CEAS_08.csv")
    enron = [
        {
            "subject": f"Meeting {i}",
            "body": f"Minutes of meeting {i}, see attached notes",
            "label": "0",
        }
        for i in range(20)
    ]
    pl.DataFrame(enron).write_csv(raw / "Enron.csv")


def test_build_dedups_and_never_splits_a_group(tmp_path: Path) -> None:
    _write_corpora(tmp_path / "raw")
    cfg = {
        "seed": 1,
        "data": {
            k: str(tmp_path / k)
            for k in ("raw_dir", "interim_dir", "processed_dir", "reference_dir", "reports_dir")
        },
        "phishing": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "proc"),
            "kaggle_dir": str(tmp_path / "kaggle"),
        },
    }
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    summary = build(load_params(tmp_path / "params.yaml"))
    assert summary["conflicting_label_texts"] == 1
    assert summary["groups_in_more_than_one_split"] == 0
    df = pl.read_parquet(tmp_path / "proc" / "emails.parquet")
    assert df["text"].map_elements(dedup_key, return_dtype=pl.String).n_unique() == df.height
    assert set(df["split"]) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        part = pl.read_parquet(tmp_path / "kaggle" / f"{split}.parquet")
        assert part.columns == ["id", "text", "label", "kind", "source"]
    assert json.loads((tmp_path / "kaggle" / "dataset-metadata.json").read_text())["title"]


def test_script_to_notebook(tmp_path: Path) -> None:
    src = tmp_path / "nb.py"
    src.write_text(
        "# %% [markdown]\n# # Title\n# text\n\n# %%\nx = 1\n\n# %%\nprint(x)\n", encoding="utf-8"
    )
    nb = json.loads(script_to_notebook(src, tmp_path / "nb.ipynb").read_text())
    assert [c["cell_type"] for c in nb["cells"]] == ["markdown", "code", "code"]
    assert nb["cells"][0]["source"] == ["# Title\n", "text"]
    assert nb["cells"][2]["source"] == ["print(x)"]
