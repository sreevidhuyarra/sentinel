import numpy as np
import polars as pl
import pytest
import torch

from sentinel.adversarial.attacks import ZSpace, masked_pgd
from sentinel.adversarial.threat import CONTROLLABLE, INCREASE_ONLY, masks, pad_and_delay, project
from sentinel.data.synthetic import HEADER
from sentinel.ids.dataset import BENIGN, CLASSES


def _frame() -> pl.DataFrame:
    from sentinel.data.columns import normalize_columns

    rng = np.random.default_rng(0)
    n = 8
    data = {
        c: rng.integers(1, 1000, n).astype(float)
        for c in HEADER
        if c not in ("Label", "Timestamp", "Flow ID", "Src IP", "Dst IP")
    }
    df = normalize_columns(pl.DataFrame(data))
    # Make the byte / length stats mutually consistent enough for the checks below.
    return df.with_columns(
        pl.max_horizontal("fwd_packet_length_max", "bwd_packet_length_max").alias(
            "packet_length_max"
        )
    )


def test_pad_and_delay_identity() -> None:
    df = _frame()
    out = pad_and_delay(df, 0.0, 1.0)
    for c in df.columns:
        assert np.allclose(out[c].cast(pl.Float64).to_numpy(), df[c].cast(pl.Float64).to_numpy()), c


def test_padding_and_delay_update_dependent_features() -> None:
    df = _frame()
    out = pad_and_delay(df, 1.0, 2.0)
    assert np.allclose(out["total_length_of_fwd_packet"], 2 * df["total_length_of_fwd_packet"])
    assert np.allclose(out["flow_duration"], 2 * df["flow_duration"])
    assert np.allclose(out["flow_packets_s"], df["flow_packets_s"] / 2)
    fwd, bwd = (
        df["total_length_of_fwd_packet"].to_numpy(),
        df["total_length_of_bwd_packet"].to_numpy(),
    )
    ratio = (2 * fwd + bwd) / (fwd + bwd)
    assert np.allclose(out["flow_bytes_s"], df["flow_bytes_s"] * ratio / 2)
    assert (out["packet_length_max"] >= out["fwd_packet_length_max"]).all()
    assert np.allclose(out["dst_port"], df["dst_port"])  # never touched
    with pytest.raises(ValueError):
        pad_and_delay(df, -0.1, 1.0)
    with pytest.raises(ValueError):
        pad_and_delay(df, 0.0, 0.5)


def test_masks() -> None:
    cols = ["dst_port", "flow_duration", "flow_bytes_s", "syn_flag_count", "fwd_init_win_bytes"]
    ctrl, inc = masks(cols)
    assert ctrl.tolist() == [False, True, True, False, True]
    assert inc.tolist() == [False, True, False, False, False]
    assert set(INCREASE_ONLY) <= set(CONTROLLABLE)


def test_project_enforces_threat_model() -> None:
    z = np.array([[0.0, 0.0, 0.0]])
    z_adv = np.array([[5.0, -1.0, 9.0]])
    ctrl, inc = np.array([False, True, True]), np.array([False, True, False])
    out = project(z_adv, z, np.full(3, -2.0), np.full(3, 2.0), ctrl, inc)
    # fixed restored; increase-only cannot go below the original; everything clipped
    assert out.tolist() == [[0.0, 0.0, 2.0]]


def test_masked_pgd_respects_constraints_and_moves_towards_benign() -> None:
    torch.manual_seed(0)
    n_feat = 6
    net = torch.nn.Sequential(
        torch.nn.Linear(n_feat, 16), torch.nn.ReLU(), torch.nn.Linear(16, len(CLASSES))
    )
    zs = ZSpace(
        np.zeros(n_feat, np.float32),
        np.ones(n_feat, np.float32),
        np.full(n_feat, -3.0),
        np.full(n_feat, 3.0),
    )
    ctrl = np.array([True, True, True, False, False, True])
    inc = np.array([True, False, False, False, False, False])
    z = torch.randn(32, n_feat)
    target = torch.full((32,), BENIGN)
    before = torch.nn.functional.cross_entropy(net(z), target).item()
    adv = masked_pgd(net, z, eps=0.5, steps=10, zs=zs, ctrl=ctrl, inc=inc)
    after = torch.nn.functional.cross_entropy(net(adv), target).item()
    assert after < before
    d = (adv - z).numpy()
    assert np.abs(d).max() <= 0.5 + 1e-5
    assert np.allclose(d[:, ~ctrl], 0)
    assert (d[:, inc] >= -1e-6).all()
