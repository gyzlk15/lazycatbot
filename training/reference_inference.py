#!/usr/bin/env python3
"""Reference HalfKP / NNUE inference for the LazyCatBot pilot.

This module is the *Python* side of the `.nnue` v1 contract
(`docs/NNUE_PILOT.md` section 3.3).  It has no PyTorch dependency: it loads an
exported model with NumPy only and reproduces the integer-accumulator forward
pass that the C++ path must implement.  `tools/nnue_validate.py` and the C++
comparison test should use it as the source of truth.

File layout (little-endian, 64-byte header)::

    offset 0   4s  magic b"LCNN"
    offset 4   u32 version (=1)
    offset 8   u32 arch    (=1, HalfKP)
    offset 12  u32 inputs  (=49152)
    offset 16  u32 hidden  (128/256/384/512 in the pilot)
    offset 20  u32 buckets (=64 king buckets)
    offset 24  u32 quant   (=1: int16 FT, float32 dense)
    offset 28  u32 reserved (=0)
    offset 32  f32 ft_scale
    offset 36  u32 policy_size (0 or 4352)
    offset 40  u64 checksum (FNV-1a 64 over everything after the header)
    offset 48  16 bytes reserved (=0)

Payload order::

    int16  ft[inputs * hidden]     row-major, input-major
    f32    ft_bias[hidden]
    f32    fc1_w[hidden * 32]      row-major, input-major
    f32    fc1_b[32]
    f32    fc2_w[32 * 32]          row-major, input-major
    f32    fc2_b[32]
    f32    out_w[32]
    f32    out_b[1]
    if policy_size > 0:
        f32 policy_w[hidden * policy_size]   row-major, input-major
        f32 policy_b[policy_size]

Forward pass::

    acc = sum(int16 ft rows for active features)      # int32
    h   = acc * ft_scale + ft_bias                    # float32
    h   = max(h, 0)
    fc1 -> max(0) -> fc2 -> max(0) -> out             # out == centipawns
    wdl = sigmoid(out / 400)

The exporter pre-multiplies the last layer by ``cp_scale = 400`` so the raw
file output is already a side-to-move centipawn score; the logit is
``out / 400``.  See ``training/model_contract.md``.

Feature function (section 3.1)::

    perspective = WHITE or BLACK
    ownKing     = king square of the perspective side
    bucket      = ownKing if WHITE else ownKing ^ 56
    for each piece (color, type, sq):
        skip the own king
        osq = sq if WHITE else sq ^ 56
        rel = 0 if color == perspective else 1
        idx = (bucket * 12 + rel * 6 + type) * 64 + osq
    inputs = 64 * 12 * 64 = 49152

Usage::

    python training/reference_inference.py --model training/work/model/model.nnue \
        --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    python training/reference_inference.py --model model.nnue --fen-file fens.txt --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # python-chess is a training dependency, not a runtime one.
    import chess
except ImportError as exc:  # pragma: no cover - exercised only on broken envs
    chess = None
    _CHESS_IMPORT_ERROR = exc
else:
    _CHESS_IMPORT_ERROR = None

MAGIC = b"LCNN"
VERSION = 1
ARCH_HALF_KP = 1
INPUTS = 64 * 12 * 64  # 49152
BUCKETS = 64
POLICY_SIZE = 4352
QUANT_INT16_FT = 1
HEADER_SIZE = 64
CP_SCALE = 400.0
FNV_OFFSET_64 = 14695981039346656037
FNV_PRIME_64 = 1099511628211

# piece type indices match the engine: PAWN=0 .. KING=5
MATERIAL = np.array([1, 3, 3, 5, 9, 0], dtype=np.int64)
PHASE_INC = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)

HEADER_STRUCT = struct.Struct("<4s7IfIQ16s")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def fnv1a_64(data: bytes) -> int:
    """FNV-1a 64-bit hash, matching the C++ implementation."""
    h = FNV_OFFSET_64
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME_64) & 0xFFFFFFFFFFFFFFFF
    return h


def _require_chess() -> None:
    if chess is None:  # pragma: no cover
        raise RuntimeError(
            f"python-chess is required for FEN handling ({_CHESS_IMPORT_ERROR})"
        )


def _engine_pt(piece_type: int) -> int:
    # python-chess: PAWN=1..KING=6; engine: PAWN=0..KING=5
    return int(piece_type) - 1


# ---------------------------------------------------------------------------
# Feature extraction (contract 3.1)
# ---------------------------------------------------------------------------
def fen_to_halfkp_indices(fen: str, perspective: Optional[str] = None) -> np.ndarray:
    """Return HalfKP feature indices for one perspective.

    ``perspective`` is ``"w"``, ``"b"`` or ``None`` (side to move).  The
    returned array is ``int64`` with one entry per non-king piece; the own
    king is represented by the bucket and is not listed.
    """
    _require_chess()
    board = chess.Board(fen)
    if perspective is None:
        perspective = "w" if board.turn == chess.WHITE else "b"
    perspective = perspective.lower()
    if perspective not in ("w", "b"):
        raise ValueError(f"perspective must be 'w', 'b' or None, got {perspective!r}")
    white = perspective == "w"
    own_color = chess.WHITE if white else chess.BLACK
    own_king = board.king(own_color)
    if own_king is None:
        raise ValueError(f"no {perspective} king on the board: {fen!r}")
    bucket = own_king if white else (own_king ^ 56)

    out: List[int] = []
    for color in (chess.WHITE, chess.BLACK):
        rel = 0 if color == own_color else 1
        for pt in range(1, 7):
            if color == own_color and pt == chess.KING:
                continue  # encoded by the bucket
            for sq in board.pieces(pt, color):
                osq = sq if white else (sq ^ 56)
                out.append((bucket * 12 + rel * 6 + _engine_pt(pt)) * 64 + osq)
    return np.asarray(out, dtype=np.int64)


def fen_to_halfkp_both(fen: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(white_indices, black_indices)`` for a FEN."""
    return (
        fen_to_halfkp_indices(fen, "w"),
        fen_to_halfkp_indices(fen, "b"),
    )


def material_from_feature_indices(indices: Sequence[int]) -> Tuple[int, int, int]:
    """Decode ``(own_material, opponent_material, phase)`` from feature indices.

    Material values are 1/3/3/5/9 and phase increments 0/1/1/2/4 per
    non-king piece, matching ``src/eval.cpp``.  Material is from the
    perspective encoded in the features (rel==0 means the perspective side).
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return 0, 0, 0
    rem = idx % 768
    rel = rem // 384
    pt = (rem % 384) // 64
    own = int(MATERIAL[pt][rel == 0].sum())
    opp = int(MATERIAL[pt][rel == 1].sum())
    phase = int(PHASE_INC[pt].sum())  # phase is perspective independent
    return own, opp, phase


# ---------------------------------------------------------------------------
# Policy index helpers (contract 3.2)
# ---------------------------------------------------------------------------
def encode_policy_index(from_sq: int, to_sq: int, promo: Optional[int] = None) -> int:
    """Encode a move as a policy class (0..4351)."""
    if promo is not None:
        if promo not in (2, 3, 4, 5):
            raise ValueError("promo must be 2=N, 3=B, 4=R, 5=Q")
        return 4096 + (promo - 2) * 64 + from_sq
    return from_sq * 64 + to_sq


_PROMO_CHAR = {2: "n", 3: "b", 4: "r", 5: "q"}


def decode_policy_index(index: int, white: bool = True) -> Optional[str]:
    """Decode a policy class to a UCI move string (``None`` for 0xFFFF).

    Promotion classes do not store the destination square; it follows from the
    from-square and the moving side, so ``white`` must be supplied for them
    (defaults to White).
    """
    if index is None or index < 0 or index == 0xFFFF:
        return None
    _require_chess()
    if index < 4096:
        from_sq, to_sq = divmod(index, 64)
        return chess.square_name(from_sq) + chess.square_name(to_sq)
    if index < POLICY_SIZE:
        off = index - 4096
        promo = off // 64 + 2  # 2=N, 3=B, 4=R, 5=Q
        from_sq = off % 64
        to_sq = from_sq + 8 if white else from_sq - 8
        if not (0 <= to_sq < 64):
            raise ValueError(f"impossible promotion class {index}")
        return (
            chess.square_name(from_sq)
            + chess.square_name(to_sq)
            + _PROMO_CHAR[promo]
        )
    raise ValueError(f"policy index out of range: {index}")


# ---------------------------------------------------------------------------
# Model reader / forward pass
# ---------------------------------------------------------------------------
@dataclass
class NNUEModel:
    hidden: int
    ft: np.ndarray  # int16 [INPUTS, hidden]
    ft_scale: float
    ft_bias: np.ndarray  # float32 [hidden]
    fc1_w: np.ndarray  # float32 [hidden, 32]
    fc1_b: np.ndarray  # float32 [32]
    fc2_w: np.ndarray  # float32 [32, 32]
    fc2_b: np.ndarray  # float32 [32]
    out_w: np.ndarray  # float32 [32], already in centipawn units
    out_b: np.ndarray  # float32 [1]
    version: int = VERSION
    arch: int = ARCH_HALF_KP
    inputs: int = INPUTS
    buckets: int = BUCKETS
    quant: int = QUANT_INT16_FT
    policy_size: int = 0
    policy_w: Optional[np.ndarray] = None  # float32 [hidden, policy_size]
    policy_b: Optional[np.ndarray] = None  # float32 [policy_size]
    checksum: int = 0
    path: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # -- forward ---------------------------------------------------------
    def _hidden(self, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= self.inputs):
            raise ValueError("feature index outside [0, inputs)")
        acc = self.ft[idx].sum(axis=0, dtype=np.int64).astype(np.float32)
        h = acc * np.float32(self.ft_scale) + self.ft_bias
        np.maximum(h, 0.0, out=h)
        return h

    @staticmethod
    def _dense(h: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        return h @ w + b  # w is [in, out] row-major/input-major

    def forward_hidden(self, indices: np.ndarray) -> np.ndarray:
        """Return the clipped FT activation for one position."""
        return self._hidden(indices)

    def forward_indices(self, indices: np.ndarray) -> float:
        """Return the side-to-move centipawn score for one position."""
        h = self._hidden(indices)
        h = self._dense(h, self.fc1_w, self.fc1_b)
        np.maximum(h, 0.0, out=h)
        h = self._dense(h, self.fc2_w, self.fc2_b)
        np.maximum(h, 0.0, out=h)
        out = self._dense(h, self.out_w, self.out_b)
        return float(out[0])

    def policy_logits(self, indices: np.ndarray) -> Optional[np.ndarray]:
        if self.policy_w is None:
            return None
        h = self._hidden(indices)
        return self._dense(h, self.policy_w, self.policy_b)

    def eval_fen(self, fen: str, perspective: Optional[str] = None) -> dict:
        """Evaluate a FEN.  CP is side-to-move centipawns (``out`` as stored)."""
        _require_chess()
        board = chess.Board(fen)
        if perspective is None:
            perspective = "w" if board.turn == chess.WHITE else "b"
        idx = fen_to_halfkp_indices(fen, perspective)
        cp = self.forward_indices(idx)
        result = {
            "fen": fen,
            "perspective": perspective,
            "cp": cp,
            "wdl": 1.0 / (1.0 + math.exp(-cp / CP_SCALE)),
            "logit": cp / CP_SCALE,
            "n_features": int(idx.size),
        }
        logits = self.policy_logits(idx)
        if logits is not None:
            k = min(5, logits.size)
            order = np.argsort(-logits)[:k]
            result["policy_topk"] = [
                {
                    "index": int(i),
                    "uci": decode_policy_index(
                        int(i), white=board.turn == chess.WHITE
                    ),
                    "logit": float(logits[i]),
                }
                for i in order
            ]
            result["policy_size"] = int(logits.size)
        return result


def _expected_payload_size(hidden: int, inputs: int, policy_size: int) -> int:
    size = (
        inputs * hidden * 2  # int16 FT
        + hidden * 4  # ft_bias
        + hidden * 32 * 4  # fc1_w
        + 32 * 4  # fc1_b
        + 32 * 32 * 4  # fc2_w
        + 32 * 4  # fc2_b
        + 32 * 4  # out_w
        + 4  # out_b
    )
    if policy_size > 0:
        size += hidden * policy_size * 4 + policy_size * 4
    return size


def parse_header(blob: bytes) -> dict:
    """Parse and validate a 64-byte `.nnue` v1 header."""
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"file too small for header: {len(blob)} bytes")
    (
        magic,
        version,
        arch,
        inputs,
        hidden,
        buckets,
        quant,
        reserved0,
        ft_scale,
        policy_size,
        checksum,
        reserved1,
    ) = HEADER_STRUCT.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}, expected {VERSION}")
    if arch != ARCH_HALF_KP:
        raise ValueError(f"unsupported arch {arch}, expected {ARCH_HALF_KP} (HalfKP)")
    if inputs != INPUTS:
        raise ValueError(f"inputs {inputs} != {INPUTS}")
    if hidden <= 0 or hidden % 32 != 0:
        raise ValueError(f"implausible hidden size {hidden}")
    if buckets != BUCKETS:
        raise ValueError(f"buckets {buckets} != {BUCKETS}")
    if quant != QUANT_INT16_FT:
        raise ValueError(f"unsupported quant mode {quant}, expected {QUANT_INT16_FT}")
    if reserved0 != 0 or reserved1 != b"\x00" * 16:
        raise ValueError("reserved header fields must be zero")
    if not math.isfinite(ft_scale) or ft_scale <= 0.0:
        raise ValueError(f"invalid ft_scale {ft_scale}")
    if policy_size not in (0, POLICY_SIZE):
        raise ValueError(f"policy_size must be 0 or {POLICY_SIZE}, got {policy_size}")
    return {
        "magic": magic.decode("ascii"),
        "version": version,
        "arch": arch,
        "inputs": inputs,
        "hidden": hidden,
        "buckets": buckets,
        "quant": quant,
        "ft_scale": ft_scale,
        "policy_size": policy_size,
        "checksum": checksum,
    }


def load_nnue(path: str, verify_checksum: bool = True) -> NNUEModel:
    """Read a `.nnue` v1 file into a NumPy model."""
    with open(path, "rb") as f:
        blob = f.read()
    hdr = parse_header(blob)
    hidden = hdr["hidden"]
    inputs = hdr["inputs"]
    policy_size = hdr["policy_size"]
    payload = memoryview(blob)[HEADER_SIZE:]
    expected = _expected_payload_size(hidden, inputs, policy_size)
    if len(payload) != expected:
        raise ValueError(
            f"payload size {len(payload)} != expected {expected} "
            f"(hidden={hidden}, policy_size={policy_size})"
        )
    if verify_checksum:
        actual = fnv1a_64(bytes(payload))
        if actual != hdr["checksum"]:
            raise ValueError(
                f"checksum mismatch: file {hdr['checksum']:#x}, computed {actual:#x}"
            )

    off = 0

    def take(count: int, dtype) -> np.ndarray:
        nonlocal off
        nbytes = count * np.dtype(dtype).itemsize
        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off).copy()
        off += nbytes
        return arr

    ft = take(inputs * hidden, np.int16).reshape(inputs, hidden)
    ft_bias = take(hidden, np.float32)
    fc1_w = take(hidden * 32, np.float32).reshape(hidden, 32)
    fc1_b = take(32, np.float32)
    fc2_w = take(32 * 32, np.float32).reshape(32, 32)
    fc2_b = take(32, np.float32)
    out_w = take(32, np.float32)
    out_b = take(1, np.float32)
    policy_w = policy_b = None
    if policy_size > 0:
        policy_w = take(hidden * policy_size, np.float32).reshape(hidden, policy_size)
        policy_b = take(policy_size, np.float32)
    assert off == expected, (off, expected)

    return NNUEModel(
        hidden=hidden,
        ft=ft,
        ft_scale=float(hdr["ft_scale"]),
        ft_bias=ft_bias,
        fc1_w=fc1_w,
        fc1_b=fc1_b,
        fc2_w=fc2_w,
        fc2_b=fc2_b,
        out_w=out_w,
        out_b=out_b,
        version=hdr["version"],
        arch=hdr["arch"],
        inputs=inputs,
        buckets=hdr["buckets"],
        quant=hdr["quant"],
        policy_size=policy_size,
        policy_w=policy_w,
        policy_b=policy_b,
        checksum=hdr["checksum"],
        path=os.path.abspath(path),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _iter_fens(args) -> Iterable[str]:
    if args.fen:
        yield args.fen
    if args.fen_file:
        with open(args.fen_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                yield line.split("|")[0].strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Reference HalfKP NNUE inference from FEN (NumPy only)."
    )
    ap.add_argument("--model", required=True, help="path to .nnue v1 file")
    ap.add_argument("--fen", default="", help="single FEN to evaluate")
    ap.add_argument("--fen-file", default="", help="file with one FEN per line")
    ap.add_argument("--perspective", choices=["w", "b", "side"], default="side")
    ap.add_argument("--json", action="store_true", help="print a JSON list")
    ap.add_argument("--no-checksum", action="store_true", help="skip checksum verification")
    ap.add_argument("--header", action="store_true", help="print the parsed header only")
    args = ap.parse_args(argv)

    try:
        model = load_nnue(args.model, verify_checksum=not args.no_checksum)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.header:
        with open(args.model, "rb") as f:
            hdr = parse_header(f.read(HEADER_SIZE))
        print(json.dumps(hdr, indent=2, sort_keys=True))
        return 0

    fens = list(_iter_fens(args))
    if not fens:
        print("error: pass --fen or --fen-file", file=sys.stderr)
        return 2

    perspective = None if args.perspective == "side" else args.perspective
    results = []
    for fen in fens:
        try:
            results.append(model.eval_fen(fen, perspective))
        except ValueError as exc:
            results.append({"fen": fen, "error": str(exc)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['fen']}\terror: {r['error']}")
            else:
                print(
                    f"cp={r['cp']:+8.2f} wdl={r['wdl']:.4f} "
                    f"persp={r['perspective']} nfeat={r['n_features']}  {r['fen']}"
                )
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
