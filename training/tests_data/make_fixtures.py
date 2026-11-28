#!/usr/bin/env python3
"""Generate deterministic synthetic fixtures for the data pipeline tests.

The real Lichess dump is 20.9 GiB; these fixtures are small enough to commit
and cover every branch of the downloader / extractor:

  * White-POV cp that follows material (so the POV correlation test passes);
  * both sides to move, including Black-to-move positions where White is better;
  * Chess960-encoded castling PV tokens (king -> rook) and standard castling;
  * promotion, long PV, PV without cp, mate, shallow depth, |cp| > 1500,
    invalid FEN, duplicate position, duplicate PV move, missing PV line;
  * legacy ``FEN|target`` lines.

Run from the repository root:
  .venv\\Scripts\\python.exe training/tests_data/make_fixtures.py
"""
from __future__ import annotations

import json
import os
import random

import chess

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261126
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def material_white_minus_black(board: chess.Board) -> int:
    total = 0
    for piece in board.piece_map().values():
        v = VALUE[piece.piece_type]
        total += v if piece.color == chess.WHITE else -v
    return total


def random_pv(board: chess.Board, rng: random.Random, min_len=3, max_len=9):
    """Return UCI_Chess960-encoded moves from a board (castling = king->rook)."""
    pv = []
    copy = board.copy(stack=False)
    for _ in range(rng.randint(min_len, max_len)):
        moves = list(copy.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        pv.append(copy.uci(mv, chess960=True))
        copy.push(mv)
    return pv


def random_material_board(rng: random.Random, seen: set[str] | None = None):
    """Random legal-ish position with a random material imbalance."""
    for _ in range(500):
        board = chess.Board()
        squares = [sq for sq, p in board.piece_map().items()
                   if p.piece_type != chess.KING]
        k = rng.randint(1, 6)
        for sq in rng.sample(squares, k):
            board.remove_piece_at(sq)
        board.turn = rng.choice([True, False])
        board.castling_rights = board.clean_castling_rights()
        if not board.is_valid():
            continue
        pv = random_pv(board, rng)
        if len(pv) < 2:
            continue
        if seen is not None:
            if board.fen() in seen:
                continue
            seen.add(board.fen())
        return board, pv
    raise RuntimeError("could not build a valid random position")


def cp_for(board: chess.Board, rng: random.Random) -> int:
    cp = 100 * material_white_minus_black(board) + rng.randint(-40, 40)
    return max(-1400, min(1400, cp))


def record(fen: str, depth: int, cp=None, mate=None, line=None, knodes=12345,
           extra_evals=None):
    pv = {}
    if cp is not None:
        pv["cp"] = cp
    if mate is not None:
        pv["mate"] = mate
    if line is not None:
        pv["line"] = line
    evals = [{"depth": depth, "knodes": knodes, "pvs": [pv]}]
    if extra_evals:
        evals = extra_evals
    return {"fen": fen, "evals": evals}


def build_records() -> tuple[list[dict], list[str], list[str]]:
    rng = random.Random(SEED)
    records: list[dict] = []
    seen: set[str] = set()

    # --- 26 material-driven positions: the POV signal.
    for _ in range(26):
        board, pv = random_material_board(rng, seen)
        fen = board.fen()
        records.append(record(fen, rng.randint(22, 32), cp=cp_for(board, rng),
                              line=" ".join(pv)))

    # --- standard castling encoded as UCI_Chess960 (e1h1 / e8h8).
    cb = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    mv = cb.parse_uci("e1g1")
    castle_960 = cb.uci(mv, chess960=True)  # e1h1
    cb.push(mv)
    pv = [castle_960] + random_pv(cb, rng, 2, 5)
    records.append(record("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
                          28, cp=35, line=" ".join(pv)))

    bb = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    mv = bb.parse_uci("e8g8")
    castle_960_b = bb.uci(mv, chess960=True)
    bb.push(mv)
    pv_b = [castle_960_b] + random_pv(bb, rng, 2, 5)
    records.append(record("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 27, cp=-20,
                          line=" ".join(pv_b)))

    # --- promotion (policy index uses the promotion formula).
    pb = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    pv = ["a7a8q", "h1g1", "a8a4"]
    records.append(record(pb.fen(), 26, cp=900, line=" ".join(pv)))

    # --- long PV: must be capped to 8 moves.
    lb, long_pv = random_material_board(rng, seen)
    line = " ".join(random_pv(lb, rng, 12, 14))
    records.append(record(lb.fen(), 24, cp=cp_for(lb, rng), line=line))

    # --- PV with a duplicated move token.
    db, dpv = random_material_board(rng, seen)
    dup_line = " ".join([dpv[0], dpv[0]] + dpv[1:])
    records.append(record(db.fen(), 25, cp=cp_for(db, rng), line=dup_line))

    # --- highest-depth selection, not first eval.
    tb, tpv = random_material_board(rng, seen)
    records.append({
        "fen": tb.fen(),
        "evals": [
            {"depth": 20, "knodes": 1, "pvs": [{"cp": 500, "line": " ".join(tpv)}]},
            {"depth": 30, "knodes": 2, "pvs": [{"cp": -40, "line": " ".join(tpv)}]},
        ],
    })

    # --- same depth twice: first wins.
    sb, spv = random_material_board(rng, seen)
    records.append({
        "fen": sb.fen(),
        "evals": [
            {"depth": 25, "knodes": 1, "pvs": [{"cp": 10, "line": " ".join(spv)}]},
            {"depth": 25, "knodes": 9, "pvs": [{"cp": 90, "line": " ".join(spv)}]},
        ],
    })

    # --- duplicate FEN of the first record (dedup must keep one).
    records.append(dict(records[0]))

    # --- PV without a line (kept, policy absent).
    records.append(record("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                          30, cp=18, line=None))

    # --- drops / edge cases.
    records.append(record("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                          30, mate=3, line="e2e4 e7e5"))
    records.append(record("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                          12, cp=20, line="e2e4"))          # shallow
    records.append(record("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                          30, cp=2000, line="e2e4"))         # cp range
    records.append({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "evals": [{"depth": 30, "pvs": []}]})    # no pvs
    records.append({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "evals": [{"depth": 30, "pvs": [{"line": "e2e4"}]}]})  # no cp
    records.append(record("8/8/8/8/8/8/8/8 w - - 0 1", 30, cp=10,
                          line="a1a2"))                     # invalid FEN
    records.append({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "evals": []})                            # empty evals

    # --- text fixtures -----------------------------------------------------
    legacy = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|0.5320000",
        "8/8/8/8/8/8/8/K6k w - - 0 1|0.5000000",
        "8/8/8/8/8/8/8/K6k b - - 0 1|0.4800000",
        # v2 line with a Black-to-move negative cp and a Chess960 castle.
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1|0.4810000|-40|e8h8|e8h8 e1h1",
        # duplicated PV token and promotion
        "8/P7/8/8/8/8/8/K6k w - - 0 1|0.9920000|900|a7a8q|a7a8q a7a8q a8a4",
        # regression: the h1 rook and the e1 king move first, so the later e1h1
        # is an ordinary king move to an empty square, not a 960 castle.
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1|0.5000000|0||"
        "h1h5 e8e7 e1e2 e7f6 e2e1 f6e7 e1h1",
        "not a fen|0.5",
    ]
    resume = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|0.5",
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1|0.5",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1|0.5",
        "8/P7/8/8/8/8/8/K6k w - - 0 1|0.9",
        "8/8/8/8/8/8/8/K6k w - - 0 1|0.5",
        "8/8/8/8/8/8/8/K6k b - - 0 1|0.5",
    ]
    return records, legacy, resume


def main() -> int:
    records, legacy, resume = build_records()
    with open(os.path.join(HERE, "fixture_evals.jsonl"), "w", encoding="utf-8",
              newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    with open(os.path.join(HERE, "fixture_legacy.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        for line in legacy:
            fh.write(line + "\n")
    # 24 distinct simple positions for the resume test.
    with open(os.path.join(HERE, "fixture_resume.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        for mask in range(24):
            # Kings on a1/h8 plus a distinct subset of white pawns on rank 2:
            # 2^8 masks, 24 used. All positions are valid and distinct.
            board = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
            for i in range(8):
                if mask >> i & 1:
                    board.set_piece_at(chess.square(i, 1),
                                       chess.Piece(chess.PAWN, chess.WHITE))
            assert board.is_valid(), board.fen()
            fh.write(f"{board.fen()}|0.5000000\n")
    print(f"wrote {len(records)} eval records, {len(legacy)} legacy lines, "
          f"24 resume lines to {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
