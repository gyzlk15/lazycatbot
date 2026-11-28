"""Contract tests: HalfKP features, policy indices, `.nnue` header and payload."""
from __future__ import annotations

import math
import struct

import numpy as np
import pytest

import export_nnue
from reference_inference import (
    HEADER_SIZE,
    INPUTS,
    POLICY_SIZE,
    _expected_payload_size,
    decode_policy_index,
    encode_policy_index,
    fen_to_halfkp_indices,
    fnv1a_64,
    load_nnue,
    material_from_feature_indices,
    parse_header,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _random_model_arrays(hidden: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "hidden": hidden,
        "ft": (rng.standard_normal((INPUTS, hidden)) * 0.02).astype(np.float32),
        "ft_bias": (rng.standard_normal(hidden) * 0.01).astype(np.float32),
        "fc1_w": (rng.standard_normal((hidden, 32)) * 0.05).astype(np.float32),
        "fc1_b": np.zeros(32, dtype=np.float32),
        "fc2_w": (rng.standard_normal((32, 32)) * 0.05).astype(np.float32),
        "fc2_b": np.zeros(32, dtype=np.float32),
        "out_w": (rng.standard_normal(32) * 0.01).astype(np.float32),
        "out_b": np.array([0.25], dtype=np.float32),
    }


def _write(tmp_path, name, arrays, policy=False, **kwargs):
    policy_w = policy_b = None
    if policy:
        rng = np.random.default_rng(99)
        policy_w = (rng.standard_normal((arrays["hidden"], POLICY_SIZE)) * 0.01).astype(
            np.float32
        )
        policy_b = np.zeros(POLICY_SIZE, dtype=np.float32)
    return export_nnue.write_nnue(
        str(tmp_path / name),
        hidden=arrays["hidden"],
        ft=arrays["ft"],
        ft_bias=arrays["ft_bias"],
        fc1_w=arrays["fc1_w"],
        fc1_b=arrays["fc1_b"],
        fc2_w=arrays["fc2_w"],
        fc2_b=arrays["fc2_b"],
        out_w=arrays["out_w"],
        out_b=arrays["out_b"],
        policy_w=policy_w,
        policy_b=policy_b,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Feature function (contract 3.1)
# ---------------------------------------------------------------------------
def test_halfkp_feature_indices_for_known_position():
    fen = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    white = set(fen_to_halfkp_indices(fen, "w").tolist())
    # bucket e1=4; own rook h1 rel=0 type=3; enemy king e8 rel=1 type=5 osq=e8
    expected_rook = (4 * 12 + 0 * 6 + 3) * 64 + 7
    expected_black_king = (4 * 12 + 1 * 6 + 5) * 64 + 60
    assert white == {expected_rook, expected_black_king}

    black = set(fen_to_halfkp_indices(fen, "b").tolist())
    # black own king e8 -> bucket 60^56=4; black king excluded; white king and
    # rook are flipped vertically (e1->e8, h1->h8).
    expected_white_king = (4 * 12 + 1 * 6 + 5) * 64 + 60
    expected_white_rook = (4 * 12 + 1 * 6 + 3) * 64 + 63
    assert black == {expected_white_king, expected_white_rook}


def test_halfkp_omits_own_king_and_counts_features():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    white = fen_to_halfkp_indices(fen, "w")
    black = fen_to_halfkp_indices(fen, "b")
    assert len(white) == 31  # 32 pieces minus own king
    assert len(black) == 31
    assert np.all((white >= 0) & (white < INPUTS))
    # own king bucket must be e1 for White and (e8^56)==e1 for Black
    assert np.all(white // 768 == 4)
    assert np.all(black // 768 == 4)


def test_material_decode_from_features():
    fen = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
    idx = fen_to_halfkp_indices(fen, "w")
    own, opp, phase = material_from_feature_indices(idx)
    assert (own, opp) == (5, 0)  # rook vs nothing (king worth 0)
    assert phase == 2  # rook phase increment


# ---------------------------------------------------------------------------
# Policy encoding (contract 3.2)
# ---------------------------------------------------------------------------
def test_policy_index_roundtrip():
    assert encode_policy_index(12, 28) == 796  # e2e4
    assert decode_policy_index(796) == "e2e4"
    assert decode_policy_index(0xFFFF) is None
    promo = encode_policy_index(52, 60, promo=5)  # e7e8q
    assert promo == 4096 + (5 - 2) * 64 + 52
    assert decode_policy_index(promo, white=True) == "e7e8q"
    with pytest.raises(ValueError):
        decode_policy_index(POLICY_SIZE + 1)


# ---------------------------------------------------------------------------
# File format (contract 3.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("policy", [False, True])
def test_header_and_payload_sizes(tmp_path, policy):
    arrays = _random_model_arrays(128)
    info = _write(tmp_path, "m.nnue", arrays, policy=policy)
    path = tmp_path / "m.nnue"
    size = path.stat().st_size
    expected_payload = _expected_payload_size(128, INPUTS, POLICY_SIZE if policy else 0)
    assert info["payload_size"] == expected_payload
    assert size == HEADER_SIZE + expected_payload

    hdr = parse_header(path.read_bytes())
    assert hdr["magic"] == "LCNN"
    assert hdr["version"] == 1
    assert hdr["arch"] == 1
    assert hdr["inputs"] == INPUTS
    assert hdr["hidden"] == 128
    assert hdr["buckets"] == 64
    assert hdr["quant"] == 1
    assert hdr["policy_size"] == (POLICY_SIZE if policy else 0)
    assert hdr["ft_scale"] > 0

    blob = path.read_bytes()
    payload = blob[HEADER_SIZE:]
    assert fnv1a_64(payload) == hdr["checksum"]


def test_dense_arrays_are_exact_and_output_is_centipawns(tmp_path):
    arrays = _random_model_arrays(128, seed=3)
    _write(tmp_path, "m.nnue", arrays)
    m = load_nnue(str(tmp_path / "m.nnue"))
    np.testing.assert_array_equal(m.ft_bias, arrays["ft_bias"])
    np.testing.assert_array_equal(m.fc1_w, arrays["fc1_w"])
    np.testing.assert_array_equal(m.fc2_w, arrays["fc2_w"])
    # out_w/out_b are pre-multiplied by cp_scale=400 in the file.
    np.testing.assert_array_equal(m.out_w, arrays["out_w"] * np.float32(400.0))
    np.testing.assert_array_equal(m.out_b, np.array([100.0], dtype=np.float32))
    # The FT is quantized: dequantized values must be within one step.
    err = np.abs(m.ft.astype(np.float32) * m.ft_scale - arrays["ft"])
    assert err.max() <= m.ft_scale


def test_quantized_ft_roundtrip_and_clipping(tmp_path):
    arrays = _random_model_arrays(128)
    info = _write(tmp_path, "m.nnue", arrays)
    rt = info["roundtrip"]
    assert rt["ok"] is True
    assert rt["max_abs_cp_diff_vs_float_ft"] < 2.0

    tiny_scale = 1e-9
    info2 = _write(tmp_path, "clip.nnue", arrays, ft_scale=tiny_scale)
    assert info2["clipped_values"] > 0
    m = load_nnue(str(tmp_path / "clip.nnue"))
    assert int(m.ft.max()) <= 32767
    assert int(m.ft.min()) >= -32767


def test_checksum_and_reserved_header_are_verified(tmp_path):
    arrays = _random_model_arrays(128)
    _write(tmp_path, "m.nnue", arrays)
    good = (tmp_path / "m.nnue").read_bytes()

    bad = bytearray(good)
    bad[-1] ^= 0xFF
    (tmp_path / "bad.nnue").write_bytes(bytes(bad))
    with pytest.raises(ValueError, match="checksum"):
        load_nnue(str(tmp_path / "bad.nnue"))

    reserved = bytearray(good)
    struct.pack_into("<I", reserved, 28, 1)
    (tmp_path / "reserved.nnue").write_bytes(bytes(reserved))
    with pytest.raises(ValueError, match="reserved"):
        parse_header(bytes(reserved))


def test_policy_layout_is_input_major(tmp_path):
    arrays = _random_model_arrays(128)
    info = _write(tmp_path, "m.nnue", arrays, policy=True)
    assert info["policy_size"] == POLICY_SIZE
    m = load_nnue(str(tmp_path / "m.nnue"))
    assert m.policy_w.shape == (128, POLICY_SIZE)
    assert m.policy_b.shape == (POLICY_SIZE,)

    idx = fen_to_halfkp_indices(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "w"
    )
    logits = m.policy_logits(idx)
    assert logits is not None and logits.shape == (POLICY_SIZE,)
    assert np.all(np.isfinite(logits))


def test_reference_output_is_centipawns(tmp_path):
    """A zero-weight model with out_b=0.25 must return exactly 100 cp."""
    arrays = _random_model_arrays(128)
    for key in ("ft", "ft_bias", "fc1_w", "fc1_b", "fc2_w", "fc2_b", "out_w"):
        arrays[key][...] = 0
    arrays["out_b"][...] = 0.25
    _write(tmp_path, "zero.nnue", arrays)
    model = load_nnue(str(tmp_path / "zero.nnue"))
    idx = fen_to_halfkp_indices(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "w"
    )
    assert model.forward_indices(idx) == pytest.approx(100.0)
    assert model.eval_fen(START_FEN)["wdl"] == pytest.approx(
        1.0 / (1.0 + math.exp(-100.0 / 400.0))
    )
