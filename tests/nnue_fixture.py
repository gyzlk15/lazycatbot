#!/usr/bin/env python3
"""Generate deterministic synthetic `.nnue` v1 fixtures and reference evals.

The fixture is not a trained model.  It exercises the loader, the HalfKP
feature function, the incremental accumulator and the dense forward pass with
reproducible weights.  `tiny_reference.json` contains expected side-to-move
centipawns for 100 positions, computed with the same float32 math as the C++
inference path.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys

import chess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

INPUTS = 64 * 12 * 64
BUCKETS = 64
POLICY_SIZE = 4352

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def fnv1a(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def make_weights(hidden: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ft_scale = np.float32(0.0015)
    ft = rng.integers(-1200, 1200, size=(INPUTS, hidden), dtype=np.int16)
    ft_bias = rng.normal(0.0, 0.05, size=hidden).astype(np.float32)
    fc1_w = rng.normal(0.0, 0.004, size=(hidden, 32)).astype(np.float32)
    fc1_b = rng.normal(0.0, 0.01, size=32).astype(np.float32)
    fc2_w = rng.normal(0.0, 0.010, size=(32, 32)).astype(np.float32)
    fc2_b = rng.normal(0.0, 0.01, size=32).astype(np.float32)
    # The v1 exporter pre-multiplies the output layer by cp_scale=400, so the
    # file stores centipawns directly.  Mirror that here.
    out_w = (rng.normal(0.0, 0.020, size=32) * np.float32(400.0)).astype(
        np.float32
    )
    out_b = (rng.normal(0.0, 0.01, size=1) * np.float32(400.0)).astype(
        np.float32
    )
    return {
        "ft": ft,
        "ft_scale": ft_scale,
        "ft_bias": ft_bias,
        "fc1_w": fc1_w,
        "fc1_b": fc1_b,
        "fc2_w": fc2_w,
        "fc2_b": fc2_b,
        "out_w": out_w,
        "out_b": out_b,
    }


def payload_bytes(w: dict) -> bytes:
    return b"".join(
        [
            w["ft"].tobytes(order="C"),
            w["ft_bias"].tobytes(order="C"),
            w["fc1_w"].tobytes(order="C"),
            w["fc1_b"].tobytes(order="C"),
            w["fc2_w"].tobytes(order="C"),
            w["fc2_b"].tobytes(order="C"),
            w["out_w"].tobytes(order="C"),
            w["out_b"].tobytes(order="C"),
        ]
    )


def write_nnue(path: str, hidden: int, seed: int) -> dict:
    w = make_weights(hidden, seed)
    payload = payload_bytes(w)
    hdr = bytearray(64)
    struct.pack_into("<4sIIIIIIIfI", hdr, 0, b"LCNN", 1, 1, INPUTS, hidden,
                     BUCKETS, 1, 0, float(w["ft_scale"]), 0)
    struct.pack_into("<Q", hdr, 40, fnv1a(payload))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(hdr))
        f.write(payload)
    return w


def features_for(board: chess.Board, perspective: bool) -> list[int]:
    king_sq = board.king(perspective)
    bucket = king_sq if perspective == chess.WHITE else (king_sq ^ 56)
    out = []
    for sq, piece in board.piece_map().items():
        if piece.color == perspective and piece.piece_type == chess.KING:
            continue
        osq = sq if perspective == chess.WHITE else (sq ^ 56)
        rel = 0 if piece.color == perspective else 1
        pt = piece.piece_type - 1  # python-chess PAWN=1 .. KING=6
        out.append((bucket * 12 + rel * 6 + pt) * 64 + osq)
    return out


def forward(w: dict, board: chess.Board) -> int:
    acc = np.zeros(w["ft"].shape[1], dtype=np.int32)
    for idx in features_for(board, board.turn):
        acc += w["ft"][idx]
    h = np.maximum(
        acc.astype(np.float32) * np.float32(w["ft_scale"]) + w["ft_bias"],
        np.float32(0.0),
    )
    l1 = np.maximum(w["fc1_b"] + h @ w["fc1_w"], np.float32(0.0))
    l2 = np.maximum(w["fc2_b"] + l1 @ w["fc2_w"], np.float32(0.0))
    out = float(w["out_b"][0] + l2 @ w["out_w"])
    cp = out  # out_w/out_b already carry the cp_scale=400 factor
    if cp >= 30000.0:
        cp = 30000.0
    elif cp <= -30000.0:
        cp = -30000.0
    return int(math.floor(cp + 0.5)) if cp >= 0 else int(math.ceil(cp - 0.5))


def generate_fens(n: int) -> list[str]:
    rng = np.random.default_rng(424242)
    fens: list[str] = []
    starts = [
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ]
    for start in starts:
        board = chess.Board(start)
        for _ in range(n // len(starts) + 1):
            if board.is_game_over(claim_draw=False):
                break
            fens.append(board.fen(en_passant="fen"))
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(0, len(moves)))])
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for fen in fens:
        if fen not in seen:
            seen.add(fen)
            unique.append(fen)
    return unique[:n]


def main() -> int:
    os.makedirs(FIXTURES, exist_ok=True)

    w = write_nnue(os.path.join(FIXTURES, "tiny.nnue"), 24, 20261126)
    write_nnue(os.path.join(FIXTURES, "tiny_odd.nnue"), 13, 20261127)
    # Hidden 32 is the smallest model that exercises the int8 FT + int8 FC1
    # fast path (FC1 requires a multiple of 32 hidden units).
    write_nnue(os.path.join(FIXTURES, "tiny32.nnue"), 32, 20261128)

    refs = {}
    for fen in generate_fens(100):
        board = chess.Board(fen)
        refs[fen] = forward(w, board)
    with open(os.path.join(FIXTURES, "tiny_reference.json"), "w",
              encoding="utf-8") as f:
        json.dump(
            {
                "hidden": 24,
                "ft_scale": float(w["ft_scale"]),
                "fens": refs,
            },
            f,
            indent=1,
            sort_keys=True,
        )
    print(f"wrote tiny.nnue (hidden=24), tiny_odd.nnue (hidden=13), "
          f"{len(refs)} reference FENs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
