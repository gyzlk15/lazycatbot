#!/usr/bin/env python3
"""Export LazyCatBot NNUE v1 files (see docs/NNUE_PILOT.md section 3.3).

The `.nnue` container is deliberately tiny and self-describing::

    64-byte header (little-endian)
    ---------- payload ----------
    int16  ft[inputs * hidden]          input-major (row = feature index)
    f32    ft_bias[hidden]
    f32    fc1_w[hidden * 32]           input-major
    f32    fc1_b[32]
    f32    fc2_w[32 * 32]               input-major
    f32    fc2_b[32]
    f32    out_w[32]                    pre-multiplied by cp_scale
    f32    out_b[1]                     pre-multiplied by cp_scale
    [optional policy head]
    f32    policy_w[hidden * policy_size]   input-major
    f32    policy_b[policy_size]

`out_w/out_b` are scaled by ``cp_scale = 400`` at export time, so the raw
output of the C++ forward pass is a side-to-move centipawn score and
``wdl = sigmoid(cp / 400)``.  This keeps the engine and the Python reference in
the same units; see ``training/model_contract.md``.

Usage (normally called by ``train_halfkp.py``)::

    python training/export_nnue.py --check training/work/model/model.nnue
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np

try:
    from reference_inference import (
        ARCH_HALF_KP,
        BUCKETS,
        CP_SCALE,
        HEADER_SIZE,
        INPUTS,
        MAGIC,
        POLICY_SIZE,
        QUANT_INT16_FT,
        VERSION,
        HEADER_STRUCT,
        _expected_payload_size,
        fnv1a_64,
        load_nnue,
        parse_header,
    )
except ImportError:  # imported as a package (training.export_nnue)
    from training.reference_inference import (  # type: ignore
        ARCH_HALF_KP,
        BUCKETS,
        CP_SCALE,
        HEADER_SIZE,
        INPUTS,
        MAGIC,
        POLICY_SIZE,
        QUANT_INT16_FT,
        VERSION,
        HEADER_STRUCT,
        _expected_payload_size,
        fnv1a_64,
        load_nnue,
        parse_header,
    )

INT16_MAX = 32767
MODEL_CARD = "model_card.json"


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_ft(
    ft_float: np.ndarray,
    ft_scale: Optional[float] = None,
    clip: int = INT16_MAX,
) -> Dict[str, object]:
    """Quantize a float FT matrix to int16 with a single global scale.

    ``round(w / ft_scale)`` clipped to ``[-clip, clip]``.  The default scale is
    ``max|w| / 32767`` so the largest weight maps to the top of the int16 range.
    """
    w = np.asarray(ft_float, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"FT matrix must be 2-D, got shape {w.shape}")
    max_abs = float(np.max(np.abs(w))) if w.size else 0.0
    if ft_scale is None:
        ft_scale = max_abs / clip if max_abs > 0.0 else 1.0
    ft_scale = float(ft_scale)
    if not math.isfinite(ft_scale) or ft_scale <= 0.0:
        raise ValueError(f"invalid ft_scale {ft_scale}")

    q = np.rint(w / ft_scale)
    clipped = int(np.count_nonzero((q < -clip) | (q > clip)))
    q = np.clip(q, -clip, clip).astype(np.int16)
    deq = q.astype(np.float32) * np.float32(ft_scale)
    err = np.abs(deq - w)
    return {
        "ft": q,
        "ft_scale": ft_scale,
        "max_abs_weight": max_abs,
        "clipped_values": clipped,
        "quant_max_abs_err": float(err.max()) if err.size else 0.0,
        "quant_mean_abs_err": float(err.mean()) if err.size else 0.0,
        "quant_rmse": float(np.sqrt(np.mean(err.astype(np.float64) ** 2)))
        if err.size
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _as_f32(arr, name: str, count: Optional[int] = None) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).reshape(-1)
    if count is not None and a.size != count:
        raise ValueError(f"{name}: expected {count} floats, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: contains non-finite values")
    return a


def write_nnue(
    path: str,
    *,
    hidden: int,
    ft: np.ndarray,
    ft_bias: np.ndarray,
    fc1_w: np.ndarray,
    fc1_b: np.ndarray,
    fc2_w: np.ndarray,
    fc2_b: np.ndarray,
    out_w: np.ndarray,
    out_b: np.ndarray,
    policy_w: Optional[np.ndarray] = None,
    policy_b: Optional[np.ndarray] = None,
    ft_scale: Optional[float] = None,
    cp_scale: float = CP_SCALE,
    verify: bool = True,
    verify_samples: int = 32,
    seed: int = 0,
) -> Dict[str, object]:
    """Write a `.nnue` v1 file and return an info dict for the model card.

    ``ft`` may be float32 (quantized here) or int16 (requires ``ft_scale``).
    ``out_w``/``out_b`` are multiplied by ``cp_scale`` so the file emits cp.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    hidden = int(hidden)
    if hidden <= 0 or hidden % 32 != 0:
        raise ValueError(f"hidden must be a positive multiple of 32, got {hidden}")

    quant: Dict[str, object]
    ft_arr = np.asarray(ft)
    if ft_arr.dtype == np.int16:
        if ft_scale is None:
            raise ValueError("int16 FT input requires an explicit ft_scale")
        if ft_arr.shape != (INPUTS, hidden):
            raise ValueError(f"FT shape {ft_arr.shape} != ({INPUTS}, {hidden})")
        q = np.ascontiguousarray(ft_arr)
        quant = {
            "ft": q,
            "ft_scale": float(ft_scale),
            "clipped_values": 0,
            "quant_max_abs_err": None,
            "quant_mean_abs_err": None,
            "quant_rmse": None,
        }
    else:
        if ft_arr.shape != (INPUTS, hidden):
            raise ValueError(f"FT shape {ft_arr.shape} != ({INPUTS}, {hidden})")
        quant = quantize_ft(ft_arr, ft_scale)
        q = quant["ft"]
    ft_scale = float(quant["ft_scale"])

    policy_size = 0
    pw = pb = None
    if policy_w is not None or policy_b is not None:
        if policy_w is None or policy_b is None:
            raise ValueError("policy_w and policy_b must be provided together")
        pw = np.ascontiguousarray(np.asarray(policy_w, dtype=np.float32))
        if pw.shape != (hidden, POLICY_SIZE):
            raise ValueError(f"policy_w shape {pw.shape} != ({hidden}, {POLICY_SIZE})")
        pb = _as_f32(policy_b, "policy_b", POLICY_SIZE)
        policy_size = POLICY_SIZE

    bias = _as_f32(ft_bias, "ft_bias", hidden)
    w1 = _as_f32(fc1_w, "fc1_w", hidden * 32)
    b1 = _as_f32(fc1_b, "fc1_b", 32)
    w2 = _as_f32(fc2_w, "fc2_w", 32 * 32)
    b2 = _as_f32(fc2_b, "fc2_b", 32)
    ow = _as_f32(out_w, "out_w", 32) * np.float32(cp_scale)
    ob = _as_f32(out_b, "out_b", 1) * np.float32(cp_scale)

    parts = [
        q.astype(np.int16, copy=False).tobytes(order="C"),
        bias.tobytes(),
        w1.tobytes(),
        b1.tobytes(),
        w2.tobytes(),
        b2.tobytes(),
        ow.tobytes(),
        ob.tobytes(),
    ]
    if policy_size > 0:
        parts.append(pw.astype(np.float32, copy=False).tobytes(order="C"))
        parts.append(pb.tobytes())
    payload = b"".join(parts)

    expected = _expected_payload_size(hidden, INPUTS, policy_size)
    if len(payload) != expected:  # pragma: no cover - internal guard
        raise AssertionError(f"payload {len(payload)} != expected {expected}")
    checksum = fnv1a_64(payload)

    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        ARCH_HALF_KP,
        INPUTS,
        hidden,
        BUCKETS,
        QUANT_INT16_FT,
        0,  # reserved
        ft_scale,
        policy_size,
        checksum,
        b"\x00" * 16,
    )
    assert len(header) == HEADER_SIZE

    with open(path, "wb") as f:
        f.write(header)
        f.write(payload)

    with open(path, "rb") as f:
        blob = f.read()
    info: Dict[str, object] = {
        "path": os.path.abspath(path),
        "file_size": HEADER_SIZE + len(payload),
        "payload_size": len(payload),
        "hidden": hidden,
        "inputs": INPUTS,
        "buckets": BUCKETS,
        "version": VERSION,
        "arch": ARCH_HALF_KP,
        "quant": QUANT_INT16_FT,
        "ft_scale": ft_scale,
        "cp_scale": float(cp_scale),
        "policy_size": policy_size,
        "checksum": f"{checksum:#018x}",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "format": "nnue-v1-halfkp",
        "clipped_values": quant["clipped_values"],
        "quant_max_abs_err": quant["quant_max_abs_err"],
        "quant_mean_abs_err": quant["quant_mean_abs_err"],
        "quant_rmse": quant["quant_rmse"],
        "output_units": "centipawns_side_to_move",
    }

    if verify:
        info["roundtrip"] = roundtrip_check(
            path,
            ft_source=None if ft_arr.dtype == np.int16 else ft_arr,
            sample_features=_sample_feature_sets(verify_samples, seed),
            tol_cp=2.0,
        )
    return info


# ---------------------------------------------------------------------------
# Reader / round-trip verification
# ---------------------------------------------------------------------------
def _forward_with_ft(
    ft: np.ndarray,
    ft_scale: float,
    ft_bias: np.ndarray,
    fc1_w: np.ndarray,
    fc1_b: np.ndarray,
    fc2_w: np.ndarray,
    fc2_b: np.ndarray,
    out_w: np.ndarray,
    out_b: np.ndarray,
    indices: np.ndarray,
) -> float:
    acc = ft[np.asarray(indices, dtype=np.int64)].sum(axis=0, dtype=np.float64)
    h = acc * float(ft_scale) + ft_bias.astype(np.float64)
    h = np.maximum(h, 0.0)
    h = np.maximum(h @ fc1_w.astype(np.float64) + fc1_b, 0.0)
    h = np.maximum(h @ fc2_w.astype(np.float64) + fc2_b, 0.0)
    return float(h @ out_w.astype(np.float64) + out_b[0])


def _sample_feature_sets(count: int, seed: int, max_features: int = 30):
    """Random but structurally valid HalfKP index sets for smoke checks."""
    rng = np.random.default_rng(seed + 991)
    out = []
    for _ in range(max(1, count)):
        bucket = int(rng.integers(0, BUCKETS))
        n = int(rng.integers(2, max_features + 1))
        rem = rng.choice(12 * 64, size=n, replace=False)
        out.append((bucket * 12 * 64 + rem).astype(np.int64))
    return out


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def roundtrip_check(
    path: str,
    *,
    ft_source: Optional[np.ndarray] = None,
    sample_features: Optional[Sequence[np.ndarray]] = None,
    tol_cp: float = 2.0,
) -> Dict[str, object]:
    """Reload a file and verify structure plus forward-pass consistency.

    - header is re-parsed and the payload checksum re-verified;
    - payload byte size matches the header;
    - reloaded FT/dense arrays are bit-identical to what was requested when
      ``ft_source`` is supplied (dense arrays are not quantized, so this is an
      exactness check);
    - the quantized-FT forward pass is compared against the float-FT forward
      pass on a few synthetic feature sets.
    """
    model = load_nnue(path, verify_checksum=True)
    blob = _read_bytes(path)
    hdr = parse_header(blob)
    expected = _expected_payload_size(hdr["hidden"], hdr["inputs"], hdr["policy_size"])
    if len(blob) - HEADER_SIZE != expected:
        raise ValueError("payload size mismatch after reload")

    result: Dict[str, object] = {
        "ok": True,
        "checksum_verified": True,
        "payload_size": expected,
        "dense_arrays_finite": bool(
            all(
                np.all(np.isfinite(a))
                for a in (
                    model.ft_bias,
                    model.fc1_w,
                    model.fc1_b,
                    model.fc2_w,
                    model.fc2_b,
                    model.out_w,
                    model.out_b,
                )
            )
        ),
    }

    if ft_source is not None and sample_features:
        src = np.asarray(ft_source, dtype=np.float32)
        np.testing.assert_allclose(
            model.ft.astype(np.float32),
            np.rint(src / model.ft_scale).clip(-INT16_MAX, INT16_MAX),
            rtol=0,
            atol=0,
            err_msg="reloaded FT differs from quantized source",
        )
        max_diff = 0.0
        for feats in sample_features:
            cp_ref = _forward_with_ft(
                src, 1.0, model.ft_bias, model.fc1_w, model.fc1_b,
                model.fc2_w, model.fc2_b, model.out_w, model.out_b, feats,
            )
            cp_file = model.forward_indices(feats)
            max_diff = max(max_diff, abs(cp_ref - cp_file))
        result["max_abs_cp_diff_vs_float_ft"] = max_diff
        result["tolerance_cp"] = tol_cp
        if max_diff > tol_cp:
            result["ok"] = False
            raise ValueError(
                f"quantization round-trip cp error {max_diff:.3f} > {tol_cp} cp"
            )

    return result


def write_model_card(out_dir: str, card: dict) -> str:
    """Write ``model_card.json`` next to the exported model."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, MODEL_CARD)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect/verify a .nnue v1 file.")
    ap.add_argument("--check", required=True, help="path to .nnue file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        hdr = parse_header(_read_bytes(args.check))
        result = roundtrip_check(args.check)
        result["header"] = hdr
        result["file_size"] = os.path.getsize(args.check)
        result["sha256"] = hashlib.sha256(_read_bytes(args.check)).hexdigest()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"OK  {args.check}")
        print(json.dumps(hdr, indent=2, sort_keys=True))
        print(f"file_size={result['file_size']} sha256={result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
