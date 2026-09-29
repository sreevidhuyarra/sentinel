import polars as pl

from sentinel.data.columns import canonical, feature_columns, normalize_columns


def test_canonical_names() -> None:
    assert canonical("Flow Bytes/s") == "flow_bytes_s"
    assert canonical(" Destination Port") == "dst_port"
    assert canonical("Down/Up Ratio") == "down_up_ratio"
    assert canonical("Total Length of Fwd Packets") == "total_length_of_fwd_packet"
    assert canonical("FWD Init Win Bytes") == "fwd_init_win_bytes"


def test_duplicate_header_dropped() -> None:
    df = pl.DataFrame({" Fwd Header Length": [1], "Fwd Header Length.1": [1], " Label": ["BENIGN"]})
    assert normalize_columns(df).columns == ["fwd_header_length", "label"]


def test_feature_columns_exclude_identifiers() -> None:
    df = pl.DataFrame(
        {"row_id": [1], "src_port": [5], "dst_port": [22], "label": ["x"], "flow_duration": [1.0]}
    )
    assert feature_columns(df) == ["dst_port", "flow_duration"]
