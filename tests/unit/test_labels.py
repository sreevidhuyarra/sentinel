import polars as pl
import pytest

from sentinel.data.labels import Family, add_family_columns, is_attempted, to_family


@pytest.mark.parametrize(
    ("raw", "family"),
    [
        ("BENIGN", Family.BENIGN),
        ("DDoS", Family.DDOS),
        ("DoS Hulk", Family.DOS),
        ("DoS Slowhttptest - Attempted", Family.DOS),
        ("Heartbleed", Family.DOS),
        ("Portscan", Family.PORTSCAN),
        ("PortScan", Family.PORTSCAN),
        ("Infiltration - Portscan", Family.PORTSCAN),
        ("Infiltration", Family.INFILTRATION),
        ("FTP-Patator", Family.BRUTEFORCE),
        ("SSH-Patator - Attempted", Family.BRUTEFORCE),
        ("Web Attack - Brute Force", Family.WEBATTACK),
        ("Web Attack � Brute Force", Family.WEBATTACK),  # original release encoding
        ("Web Attack \x96 Sql Injection", Family.WEBATTACK),
        ("Botnet", Family.BOT),
        ("Bot", Family.BOT),
    ],
)
def test_to_family(raw: str, family: Family) -> None:
    assert to_family(raw) is family


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="Unmapped"):
        to_family("Cryptominer")


def test_is_attempted() -> None:
    assert is_attempted("Botnet - Attempted")
    assert not is_attempted("Botnet")


@pytest.mark.parametrize(
    ("policy", "rows", "attempted_family"),
    [
        ("benign", 3, "Benign"),
        ("drop", 2, None),
        ("keep", 3, "Bot"),
    ],
)
def test_attempted_policy(policy: str, rows: int, attempted_family: str | None) -> None:
    df = pl.DataFrame({"label": ["BENIGN", "Botnet", "Botnet - Attempted"]})
    out = add_family_columns(df, policy=policy)  # type: ignore[arg-type]
    assert out.height == rows
    fam = out.filter(pl.col("attempted"))["family"].to_list()
    assert fam == ([attempted_family] if attempted_family else [])
    assert out.filter(pl.col("label") == "Botnet")["family"].item() == "Bot"
