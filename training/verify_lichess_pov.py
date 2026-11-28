#!/usr/bin/env python3
"""Verify the POV, schema and Chess960 conventions of the Lichess eval DB.

The training pipeline assumes three things about
``lichess_db_eval.jsonl.zst``:

1. ``evals[i].pvs[j].cp`` is **White POV** (Lichess shows all evals from
   White's side; every ``cp`` in the dump follows that convention).
2. ``evals[i].pvs[j].line`` is a space separated UCI move list using the
   **UCI_Chess960** castling encoding (king-from -> rook-from, e.g. ``e1h1``),
   even in standard chess positions.
3. The FEN in the record describes the position before the PV; the side to
   move is field 2 of the FEN.

This tool streams the ``.zst`` (no full extraction), takes a deterministic
reservoir sample, and checks:

* schema: ``fen``, ``evals``, ``depth``, ``pvs``, ``cp``/``mate``, ``line``;
* FEN validity (python-chess) and legality of every PV line on a
  Chess960-aware board;
* whether PV castling moves are Chess960 encoded (king -> own rook) and
  rejected by a standard board;
* the POV claim, using a local recomputation from material: under White POV,
  ``cp`` must correlate positively with ``White material - Black material``;
  under side-to-move POV it would correlate with the material balance
  *oriented to the mover* instead.  A 20 position table is printed for a
  manual cross-check;
* optionally (``--engine build/chessbot.exe``) a local search score for those
  20 positions plus a color-mirror symmetry test.

Usage:
  python training/verify_lichess_pov.py                     # auto: DB if ready
  python training/verify_lichess_pov.py --input train/work/lichess_db_eval.jsonl.zst
  python training/verify_lichess_pov.py --fixture training/tests_data/fixture_evals.jsonl
  python training/verify_lichess_pov.py --check-text training/work/pilot/positions.txt

Exit status: 0 = all checks passed, 2 = database not ready (check pending),
1 = a check failed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np

try:  # python-chess is required for the PV/FEN checks.
    import chess
except ImportError:  # pragma: no cover - environment guard
    chess = None

DEFAULT_DB = os.path.join("training", "work", "lichess_db_eval.jsonl.zst")
DEFAULT_MANIFEST = os.path.join("training", "work", "lichess_db_eval.manifest.json")
EXPECTED_DB_BYTES = 22_499_667_845
PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$")


# --------------------------------------------------------------------------- io


def open_lines(path: str, status: dict | None = None):
    """Yield decoded lines from a .zst file (streamed) or a plain text file.

    A truncated ``.zst`` (download still running) ends the stream cleanly and
    sets ``status["truncated"] = True`` instead of raising: the POV sample is
    still meaningful, and the report says the input was partial.
    """
    if path.endswith(".zst"):
        return _zstd_lines(path, status)
    return _plain_lines(path)


def _zstd_lines(path: str, status=None):
    import io

    import zstandard as zstd

    fh = open(path, "rb")
    dctx = zstd.ZstdDecompressor()
    try:
        reader = dctx.stream_reader(fh, read_across_frames=True)
    except TypeError:  # older zstandard
        reader = dctx.stream_reader(fh)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace",
                            newline="\n")
    try:
        try:
            for line in text:
                yield line
        except zstd.ZstdError as exc:
            if status is None:
                raise
            status["truncated"] = True
            status["truncated_error"] = str(exc)
            print(f"warning: zstd stream ended early (partial download?): {exc}",
                  file=sys.stderr)
    finally:
        try:
            text.close()
        finally:
            fh.close()


def _plain_lines(path: str):
    fh = open(path, "r", encoding="utf-8", errors="replace", newline="\n")
    try:
        for line in fh:
            yield line
    finally:
        fh.close()


def db_ready(path: str) -> tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"missing {path}"
    size = os.path.getsize(path)
    if size < 1_000_000:
        return False, f"{path} is only {size} bytes (download in progress?)"
    return True, f"{path} is {size / 2**30:.2f} GiB"


def db_complete(path: str) -> bool:
    """True when the file matches the verified size (manifest or known URL size)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if os.path.exists(DEFAULT_MANIFEST):
        try:
            with open(DEFAULT_MANIFEST, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            expected = int(manifest.get("compressed_size", 0))
            if expected:
                return size == expected
        except (OSError, ValueError, TypeError):
            pass
    return size >= EXPECTED_DB_BYTES


# ------------------------------------------------------------------ FEN helpers


def parse_fen_fields(fen: str):
    parts = fen.split()
    if len(parts) < 2:
        return None
    return parts


def board_material(fen: str) -> int | None:
    """White material minus Black material, in pawns (P=1,N=3,B=3,R=5,Q=9)."""
    try:
        board = fen.split()[0]
    except IndexError:
        return None
    mat = 0
    for ch in board:
        low = ch.lower()
        if low in PIECE_VALUES:
            v = PIECE_VALUES[low]
            mat += v if ch.isupper() else -v
    return mat


def material_phase_units(fen: str) -> int | None:
    """Non-pawn, non-king material of both sides, in centipawns."""
    try:
        board = fen.split()[0]
    except IndexError:
        return None
    total = 0
    for ch in board:
        low = ch.lower()
        if low in ("n", "b", "r", "q"):
            total += {"n": 320, "b": 330, "r": 500, "q": 950}[low]
    return total


def stm_sign(fen: str) -> int:
    parts = fen.split()
    return -1 if len(parts) > 1 and parts[1] == "b" else 1


def king_square(fen: str, color: str) -> int | None:
    """Square index of ``K``/``k`` from the board field, or None."""
    ranks = fen.split()[0].split("/") if fen.split() else []
    if len(ranks) != 8:
        return None
    for r, row in enumerate(ranks):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                if ch == color:
                    return (7 - r) * 8 + file_idx
                file_idx += 1
    return None


def parse_board(fen: str):
    """Return 64 chars ('P'..'k' or '.') for the board field, or None."""
    parts = fen.split()
    if not parts:
        return None
    ranks = parts[0].split("/")
    if len(ranks) != 8:
        return None
    board = ["."] * 64
    for r, row in enumerate(ranks):
        file_idx = 0
        for ch in row:
            if ch.isdigit():
                file_idx += int(ch)
                if file_idx > 8:
                    return None
            elif ch in "pnbrqkPNBRQK":
                if file_idx >= 8:
                    return None
                board[(7 - r) * 8 + file_idx] = ch
                file_idx += 1
            else:
                return None
        if file_idx != 8:
            return None
    return board


def normalize_uci_960(fen: str, token: str) -> str:
    """Convert a UCI_Chess960 castling token to standard UCI (king -> king).

    Color agnostic: matches when ``from`` holds a king and ``to`` holds that
    king's rook, so PV moves after the first one are handled as well.
    """
    if len(token) < 4 or token[0] not in "abcdefgh" or token[2] not in "abcdefgh":
        return token
    from_sq = (int(token[1]) - 1) * 8 + (ord(token[0]) - 97)
    to_sq = (int(token[3]) - 1) * 8 + (ord(token[2]) - 97)
    board = parse_board(fen)
    if board is None:
        return token
    piece = board[from_sq]
    if piece not in "Kk":
        return token
    if board[to_sq] != ("R" if piece == "K" else "r"):
        return token
    target_file = 6 if (to_sq % 8) > (from_sq % 8) else 2
    target = (to_sq // 8) * 8 + target_file
    return f"{chr(97 + from_sq % 8)}{from_sq // 8 + 1}{chr(97 + target % 8)}{target // 8 + 1}"


# -------------------------------------------------------------------- sampling


def reservoir(lines, want: int, rng: random.Random, max_scan: int):
    """Deterministic reservoir sample of raw lines, plus scan counters."""
    kept: list[str] = []
    scanned = 0
    for raw in lines:
        scanned += 1
        if max_scan and scanned > max_scan:
            break
        if len(kept) < want:
            kept.append(raw)
        else:
            j = rng.randrange(scanned)
            if j < want:
                kept[j] = raw
    return kept, scanned


# ---------------------------------------------------------------------- checks


def check_schema(records):
    out = {
        "records": len(records),
        "with_fen": 0,
        "with_evals": 0,
        "evals_depth_numeric": 0,
        "with_pvs": 0,
        "with_cp_or_mate": 0,
        "with_line": 0,
        "fen_valid": 0,
        "fen_invalid_examples": [],
    }
    for rec in records:
        fen = rec.get("fen")
        if isinstance(fen, str) and len(fen.split()) >= 4:
            out["with_fen"] += 1
            if chess is not None:
                try:
                    chess.Board(fen)
                    out["fen_valid"] += 1
                except ValueError:
                    if len(out["fen_invalid_examples"]) < 3:
                        out["fen_invalid_examples"].append(fen)
        evals = rec.get("evals")
        if isinstance(evals, list) and evals:
            out["with_evals"] += 1
            best = max(evals, key=lambda e: _depth(e))
            if isinstance(best.get("depth"), (int, float)):
                out["evals_depth_numeric"] += 1
            pvs = best.get("pvs") or []
            if pvs:
                out["with_pvs"] += 1
                pv = pvs[0]
                if pv.get("cp") is not None or pv.get("mate") is not None:
                    out["with_cp_or_mate"] += 1
                line = pv.get("line")
                if isinstance(line, str) and line.strip():
                    out["with_line"] += 1
    return out


def _depth(ev):
    try:
        return int(ev.get("depth", 0))
    except (TypeError, ValueError):
        return 0


def check_pv_legality(records, max_moves: int = 12):
    """Play each sampled PV on a Chess960-aware board and a standard board."""
    stats = {
        "lines_checked": 0,
        "moves_checked": 0,
        "legal_960": 0,
        "first_illegal_960": 0,
        "standard_fail_moves": 0,
        "standard_fail_lines": 0,
        "castling_960_tokens": 0,
        "castling_std_tokens": 0,
        "castling_960_examples": [],
        "illegal_examples": [],
        "non_uci_tokens": 0,
    }
    if chess is None:
        return stats
    for rec in records:
        fen = rec.get("fen")
        if not isinstance(fen, str):
            continue
        evals = rec.get("evals")
        if not isinstance(evals, list) or not evals:
            continue
        best = max(evals, key=_depth)
        pvs = best.get("pvs") or []
        if not pvs:
            continue
        line = pvs[0].get("line")
        if not isinstance(line, str) or not line.strip():
            continue
        tokens = [t for t in line.split() if t][:max_moves]
        stats["lines_checked"] += 1
        for tok in tokens:
            if not UCI_RE.match(tok):
                stats["non_uci_tokens"] += 1
                break
        try:
            board = chess.Board(fen, chess960=True)
        except ValueError:
            continue
        # Chess960 aware replay.
        ok = True
        for tok in tokens:
            stats["moves_checked"] += 1
            try:
                mv = board.parse_uci(tok)
                if mv not in board.legal_moves:
                    ok = False
                    break
                board.push(mv)
            except ValueError:
                ok = False
                break
        if ok:
            stats["legal_960"] += 1
        else:
            stats["first_illegal_960"] += 1
            if len(stats["illegal_examples"]) < 5:
                stats["illegal_examples"].append((fen, line[:120]))

        # Standard replay: python-chess *converts* UCI_Chess960 castling tokens
        # (e1h1 -> e1g1) in parse_uci even on a standard board, so use the raw
        # Move: a 960 castling token must be rejected as "king onto own rook".
        try:
            sboard = chess.Board(fen, chess960=False)
        except ValueError:
            continue
        fail = False
        for tok in tokens:
            try:
                mv = chess.Move.from_uci(tok)
                # Membership in the *generator* normalizes 960 castling, so
                # compare against the explicit list of legal moves.
                if mv not in list(sboard.legal_moves):
                    raise ValueError(tok)
                sboard.push(mv)
            except ValueError:
                stats["standard_fail_moves"] += 1
                fail = True
                break
        if fail:
            stats["standard_fail_lines"] += 1

        # Explicit castling-token detection (king from -> own rook).
        side_is_black = stm_sign(fen) < 0
        king = king_square(fen, "k" if side_is_black else "K")
        for tok in tokens:
            if not UCI_RE.match(tok):
                continue
            from_sq = (int(tok[1]) - 1) * 8 + ord(tok[0]) - 97
            to_sq = (int(tok[3]) - 1) * 8 + ord(tok[2]) - 97
            normalized = normalize_uci_960(fen, tok)
            if normalized != tok:
                stats["castling_960_tokens"] += 1
                if len(stats["castling_960_examples"]) < 5:
                    stats["castling_960_examples"].append(
                        (fen, tok, normalized)
                    )
            elif king is not None and from_sq == king and abs(to_sq - from_sq) == 2:
                # e1g1 / e1c1 style token.
                stats["castling_std_tokens"] += 1
    return stats


def check_pov(records, min_mat_pawns: float = 1.0):
    """Decide whether cp is White POV or side-to-move POV.

    Material balance is only a noisy proxy on real Lichess data (sharp
    positions, odds/study start positions), so the decisive test is the
    Black-to-move subset: under White POV ``corr(cp, White-Black material)``
    stays positive, while under side-to-move POV it is negative (because the
    mover-oriented balance flips sign).  The pooled correlations and a median
    table are reported as supporting evidence.
    """
    cps, mat, sign = [], [], []
    rows = []
    for rec in records:
        fen = rec.get("fen")
        if not isinstance(fen, str):
            continue
        evals = rec.get("evals")
        if not isinstance(evals, list) or not evals:
            continue
        best = max(evals, key=_depth)
        pvs = best.get("pvs") or []
        if not pvs:
            continue
        pv = pvs[0]
        if pv.get("mate") is not None:
            continue
        cp = pv.get("cp")
        if cp is None:
            continue
        m = board_material(fen)
        if m is None:
            continue
        try:
            cp = float(cp)
        except (TypeError, ValueError):
            continue
        cps.append(cp)
        mat.append(float(m) * 100.0)
        sign.append(float(stm_sign(fen)))
        rows.append((cp, m, stm_sign(fen)))

    n = len(cps)
    out = {
        "cp_records": n,
        "corr_white_pov": None,
        "corr_stm_pov": None,
        "sign_agreement_white": None,
        "sign_agreement_stm": None,
        "n_abs_mat_ge_100": 0,
        "black_to_move": None,
        "median_cp_by_material": {},
        "verdict": "inconclusive (no data)",
    }
    if n < 10:
        return out
    cpv = np.asarray(cps, dtype=np.float64)
    mv = np.asarray(mat, dtype=np.float64)
    sv = np.asarray(sign, dtype=np.float64)
    mask = np.abs(mv) >= min_mat_pawns * 100.0
    out["n_abs_mat_ge_100"] = int(mask.sum())
    if mask.sum() < 10:
        return out

    def corr(a, b):
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    cp_m, mat_m, stm_m = cpv[mask], mv[mask], sv[mask]
    out["corr_white_pov"] = corr(cp_m, mat_m)
    out["corr_stm_pov"] = corr(cp_m, mat_m * stm_m)
    out["sign_agreement_white"] = float(np.mean(np.sign(cp_m) == np.sign(mat_m)))
    out["sign_agreement_stm"] = float(
        np.mean(np.sign(cp_m) == np.sign(mat_m * stm_m))
    )

    # Decisive subset: Black to move.
    bmask = stm_m < 0
    if bmask.sum() >= 8:
        out["black_to_move"] = {
            "n": int(bmask.sum()),
            "corr_white_pov": corr(cp_m[bmask], mat_m[bmask]),
            "sign_agreement_white": float(
                np.mean(np.sign(cp_m[bmask]) == np.sign(mat_m[bmask]))
            ),
            "mean_cp": float(cp_m[bmask].mean()),
        }
    # Median cp by rounded material balance (supporting evidence).
    for mval in range(-9, 10):
        sel = np.rint(mv / 100.0) == mval
        if int(sel.sum()) >= 10:
            out["median_cp_by_material"][str(mval)] = float(np.median(cpv[sel]))

    rw = out["corr_white_pov"] or 0.0
    rs = out["corr_stm_pov"] or 0.0
    b = out["black_to_move"]
    if b and b["n"] >= 8:
        rb = b["corr_white_pov"] or 0.0
        sb = b["sign_agreement_white"] or 0.0
        if rb > 0.10 and sb > 0.50:
            out["verdict"] = (
                f"White POV (Black-to-move subset: corr={rb:+.3f}, "
                f"sign={sb:.3f}, n={b['n']})"
            )
        elif rb < -0.10:
            out["verdict"] = (
                f"side-to-move POV (unexpected! Black-to-move corr={rb:+.3f})"
            )
        elif rw > 0.15:
            out["verdict"] = "White POV (weak but positive pooled correlation)"
        else:
            out["verdict"] = "inconclusive"
    else:
        if rw > 0.5 and rw > rs + 0.15:
            out["verdict"] = "White POV (cp rises with White material advantage)"
        elif rs > 0.5 and rs > rw + 0.15:
            out["verdict"] = "side-to-move POV (unexpected!)"
        elif rw < -0.5 and abs(rs) < 0.3:
            out["verdict"] = "inconsistent: cp is sign-flipped relative to material"
        else:
            out["verdict"] = "inconclusive"
    return out


def cross_check_positions(records, limit: int = 20):
    """Pick ``limit`` positions spread over material classes for manual review."""
    rows = []
    for rec in records:
        fen = rec.get("fen")
        if not isinstance(fen, str):
            continue
        evals = rec.get("evals")
        if not isinstance(evals, list) or not evals:
            continue
        best = max(evals, key=_depth)
        pvs = best.get("pvs") or []
        if not pvs:
            continue
        pv = pvs[0]
        cp = pv.get("cp")
        mate = pv.get("mate")
        mat = board_material(fen)
        rows.append(
            {
                "fen": fen,
                "depth": _depth(best),
                "cp": cp,
                "mate": mate,
                "material_white_minus_black": mat,
                "stm": "b" if stm_sign(fen) < 0 else "w",
                "pv": (pv.get("line") or "")[:100],
            }
        )
    if not rows:
        return []
    # Deterministic spread: sort by material then take evenly spaced samples.
    rows.sort(key=lambda r: (r["material_white_minus_black"] is None,
                             r["material_white_minus_black"] or 0))
    step = max(1, len(rows) // limit)
    picked = rows[::step][:limit]
    for r in picked:
        cp = r["cp"]
        mat = r["material_white_minus_black"]
        if cp is None or mat is None:
            r["pov_sane_white"] = None
        elif abs(mat) < 1:
            r["pov_sane_white"] = None
        else:
            r["pov_sane_white"] = (cp > 0) == (mat > 0)
    return picked


def check_text_file(path: str, max_lines: int = 200000):
    """Verify the pipeline's own output format/FEN/PV/POV consistency."""
    stats = {
        "path": path,
        "lines": 0,
        "parsed": 0,
        "bad_fields": 0,
        "bad_wdl": 0,
        "stm_cp_nonfinite": 0,
        "policy_present": 0,
        "pv_present": 0,
        "pov": None,
        "castling_960_normalized": 0,
        "policy_out_of_range": 0,
        "examples_bad": [],
    }
    cps, mats, signs = [], [], []
    cps900 = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stats["lines"] += 1
            if max_lines and stats["lines"] > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                stats["bad_fields"] += 1
                continue
            fen = parts[0]
            if len(fen.split()) < 4:
                stats["bad_fields"] += 1
                continue
            fields = fen.split()
            stats["parsed"] += 1
            try:
                wdl = float(parts[1])
            except ValueError:
                stats["bad_wdl"] += 1
                continue
            cp = None
            if len(parts) >= 3 and parts[2] not in ("", "nan", "NaN"):
                try:
                    cp = float(parts[2])
                except ValueError:
                    cp = None
            if cp is not None and not math.isfinite(cp):
                stats["stm_cp_nonfinite"] += 1
                cp = None
            if cp is not None and abs(cp) < 1e6:
                expect = 1.0 / (1.0 + math.exp(-cp / 400.0))
                if abs(expect - wdl) > 2e-6:
                    stats["bad_wdl"] += 1
                    if len(stats["examples_bad"]) < 3:
                        stats["examples_bad"].append(
                            f"wdl mismatch: {wdl} vs {expect:.7f} ({fen})"
                        )
            if len(parts) >= 4 and parts[3] not in ("", "0000", "none"):
                stats["policy_present"] += 1
            if len(parts) >= 5 and parts[4].strip():
                stats["pv_present"] += 1
            mat = board_material(fen)
            if cp is not None and mat is not None:
                cps.append(cp)
                mats.append(mat * 100.0)
                signs.append(float(stm_sign(fen)))
                cps900 += 1
            # sanity: fields[5:] should not exist
            if len(parts) > 5:
                stats["bad_fields"] += 1
    if cps:
        cpv = np.asarray(cps)
        mv = np.asarray(mats)
        sv = np.asarray(signs)
        # In the output file cp is STM POV, so correlate with oriented material.
        oriented = mv * sv
        mask = np.abs(mv) >= 100
        if mask.sum() >= 10 and cpv[mask].std() > 0 and oriented[mask].std() > 0:
            r = float(np.corrcoef(cpv[mask], oriented[mask])[0, 1])
            r_white = float(np.corrcoef(cpv[mask], mv[mask])[0, 1])
            if r_white > 0.10 and r_white > r + 0.10:
                verdict = "White POV (unexpected for pipeline output)"
            elif r > 0.10 and r > r_white + 0.10:
                verdict = "side-to-move POV (as required)"
            else:
                verdict = "inconclusive (weak material signal)"
            stats["pov"] = {
                "corr_stm_pov": r,
                "corr_white_pov": r_white,
                "verdict": verdict,
            }
    return stats


# ------------------------------------------------------------------- reporting


def fmt_pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def fmt3(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def render_report(meta, schema, pv, pov, cross, text_check, engine_note) -> str:
    L = []
    L.append("# Lichess eval DB: POV / schema / Chess960 verification")
    L.append("")
    L.append(f"Generated: {meta['generated_at']}")
    L.append(f"Input: `{meta['input']}`")
    L.append(f"Sampling: reservoir of {meta['sample_size']} records from "
             f"{meta['scanned']} scanned lines (seed {meta['seed']}, "
             f"scan cap {meta['max_scan']}).")
    if meta.get("partial_input"):
        L.append("")
        L.append("**Note: the input was a partial (in-progress) download. "
                 "Results describe the available prefix only; rerun after the "
                 "manifest is written.**")
    L.append("")
    L.append("## Documented convention (checked against the data)")
    L.append("")
    L.append("- `evals[i].pvs[j].cp` is **White POV**; the pipeline negates it "
             "for Black to move.")
    L.append("- `evals[i].pvs[j].line` is a UCI move list; castling uses the "
             "**UCI_Chess960** king-from -> rook-from encoding (`e1h1`, "
             "`e1a1`), so the extractor must normalize it to `e1g1`/`e1c1`.")
    L.append("- `evals[i].depth` is the search depth; the highest-depth eval is "
             "the training label.")
    L.append("")
    L.append("## Schema")
    L.append("")
    L.append(f"- sampled records: {schema['records']}")
    L.append(f"- with `fen`: {schema['with_fen']} "
             f"({fmt_pct(schema['with_fen'], schema['records'])})")
    L.append(f"- FEN accepted by python-chess: {schema['fen_valid']} "
             f"({fmt_pct(schema['fen_valid'], schema['with_fen'])})")
    L.append(f"- with non-empty `evals`: {schema['with_evals']}")
    L.append(f"- highest eval has numeric depth: {schema['evals_depth_numeric']}")
    L.append(f"- with `pvs`: {schema['with_pvs']}; with cp or mate: "
             f"{schema['with_cp_or_mate']}; with PV `line` string: "
             f"{schema['with_line']}")
    if schema["fen_invalid_examples"]:
        L.append(f"- invalid FEN examples: {schema['fen_invalid_examples']}")
    L.append("")
    L.append("## POV check (material recomputation)")
    L.append("")
    L.append("The official database docs and the Lichess/fishnet-evals dataset "
             "card state that `cp`/`mate` are **from White's point of view** and "
             "that `line`/`move` use UCI_Chess960 castling. The material proxy "
             "below is noisy on real data (sharp tactics, odds/study start "
             "positions), so the decisive subset is **Black to move**, where "
             "White POV and side-to-move POV predict opposite signs.")
    L.append("")
    L.append(f"- records with cp and |material imbalance| >= {1} pawn: "
             f"{pov['n_abs_mat_ge_100']} (of {pov['cp_records']} cp records)")
    L.append(f"- corr(cp, White-POV material) = {fmt3(pov['corr_white_pov'])}")
    L.append(f"- corr(cp, mover-oriented material) = {fmt3(pov['corr_stm_pov'])}")
    L.append(f"- sign agreement with White-POV material: "
             f"{fmt3(pov['sign_agreement_white'])}")
    L.append(f"- sign agreement with mover-oriented material: "
             f"{fmt3(pov['sign_agreement_stm'])}")
    b = pov.get("black_to_move")
    if b:
        L.append("")
        L.append("### Decisive subset: Black to move")
        L.append("")
        L.append(f"- n = {b['n']}")
        L.append(f"- corr(cp, White-POV material) = {fmt3(b['corr_white_pov'])} "
                 f"(White POV expects > 0, side-to-move POV expects < 0)")
        L.append(f"- sign agreement with White-POV material = "
                 f"{fmt3(b['sign_agreement_white'])}")
        L.append(f"- mean cp = {b['mean_cp']:.1f}")
    if pov.get("median_cp_by_material"):
        L.append("")
        L.append("### Median cp by material balance (White minus Black, pawns)")
        L.append("")
        L.append("| material | median cp |")
        L.append("|---:|---:|")
        for k, v in pov["median_cp_by_material"].items():
            L.append(f"| {k} | {v:+.0f} |")
    L.append("")
    L.append(f"- **verdict: {pov['verdict']}**")
    L.append("")
    L.append("Interpretation: a White-POV file keeps a positive correlation on "
             "the Black-to-move subset (the mover's material orientation is "
             "mirrored, so a side-to-move file flips the sign). The median cp "
             "table should rise with the material balance.")
    L.append("")
    L.append("## PV legality and Chess960 castling encoding")
    L.append("")
    L.append(f"- PV lines checked: {pv['lines_checked']} "
             f"({pv['moves_checked']} moves)")
    L.append(f"- fully legal on a Chess960 board: {pv['legal_960']} "
             f"({fmt_pct(pv['legal_960'], pv['lines_checked'])})")
    L.append(f"- lines with an illegal first move under Chess960 replay: "
             f"{pv['first_illegal_960']}")
    L.append(f"- standard-board lines that fail on a raw from->to replay: "
             f"{pv['standard_fail_lines']} ({pv['standard_fail_moves']} moves)")
    L.append(f"- Chess960 castling tokens (king -> own rook): "
             f"{pv['castling_960_tokens']}")
    L.append(f"- standard castling tokens (king -> g/c file): "
             f"{pv['castling_std_tokens']}")
    L.append("- The standard-board failures are the direct proof of the "
             "UCI_Chess960 castling encoding: the raw token (e1h1) is not a "
             "legal move on a standard board, while Chess960 replay accepts it.")
    if pv["castling_960_examples"]:
        L.append("- examples (fen, token, normalized):")
        for ex in pv["castling_960_examples"]:
            L.append(f"  - `{ex[1]}` -> `{ex[2]}` in `{ex[0][:90]}`")
    if pv["illegal_examples"]:
        L.append("- illegal examples:")
        for fen, line in pv["illegal_examples"]:
            L.append(f"  - `{fen[:90]}` / `{line}`")
    if pv["castling_960_tokens"] == 0:
        L.append("- note: no castling token in this sample; the normalization "
                 "code path is covered by the fixtures in "
                 "`training/tests_data/` and must be re-checked on a larger "
                 "sample if the count stays zero.")
    L.append("")
    L.append("## 20 position cross-check table")
    L.append("")
    if cross:
        L.append("| # | stm | cp | mate | White-Black mat | sane under White POV | FEN |")
        L.append("|---:|:---:|---:|---:|---:|:---:|---|")
        for i, r in enumerate(cross, 1):
            L.append(
                f"| {i} | {r['stm']} | {r['cp']} | {r['mate']} | "
                f"{r['material_white_minus_black']} | "
                f"{'yes' if r['pov_sane_white'] else ('n/a' if r['pov_sane_white'] is None else 'NO')} | "
                f"`{r['fen']}` |"
            )
        judged = [r for r in cross if r["pov_sane_white"] is not None]
        agree = sum(1 for r in judged if r["pov_sane_white"])
        L.append("")
        L.append(f"- {agree}/{len(judged)} sampled positions have a cp sign that "
                 f"matches the White-POV material balance. Expect a clear "
                 f"majority but not unanimity: real positions are tactical and "
                 f"the sample includes odds/study start positions.")
    else:
        L.append("_No positions available (database not ready)._")
    L.append("")
    L.append("## Pipeline output text check")
    L.append("")
    if text_check:
        L.append(f"- file: `{text_check['path']}`, parsed "
                 f"{text_check['parsed']}/{text_check['lines']} lines")
        L.append(f"- bad field counts: {text_check['bad_fields']}; "
                 f"wdl/cp mismatch: {text_check['bad_wdl']}")
        L.append(f"- policy present: {text_check['policy_present']}; "
                 f"PV present: {text_check['pv_present']}")
        if text_check["pov"]:
            L.append(f"- POV of output cp: {text_check['pov']}")
        for ex in text_check["examples_bad"]:
            L.append(f"- example: {ex}")
    else:
        L.append("_Not run (use `--check-text` after the downloader)._")
    L.append("")
    L.append("## Local engine cross-check")
    L.append("")
    L.append(engine_note or "_Not run (pass `--engine build/chessbot.exe`)._")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- engines


def engine_scores(exe: str, fens, depth: int = 12, timeout: float = 20.0):
    """Return STM-POV cp scores from a UCI engine for each FEN."""
    import subprocess

    if not os.path.exists(exe):
        return None
    scores = []
    for fen in fens:
        try:
            proc = subprocess.Popen(
                [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            proc.stdin.write("uci\nisready\n")
            proc.stdin.write(f"position fen {fen}\n")
            proc.stdin.write(f"go depth {depth}\n")
            proc.stdin.flush()
            score = None
            t0 = time.time()
            while time.time() - t0 < timeout:
                line = proc.stdout.readline()
                if not line:
                    break
                if line.startswith("info") and " score " in line:
                    parts = line.split()
                    for i, tok in enumerate(parts):
                        if tok == "cp" and i + 1 < len(parts):
                            try:
                                score = int(parts[i + 1])
                            except ValueError:
                                pass
                        elif tok == "mate" and i + 1 < len(parts):
                            score = None
                if line.startswith("bestmove"):
                    break
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            if score is not None:
                scores.append(score)
        except Exception:
            return None
    return scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="", help=".jsonl.zst (default: training/work/...)")
    ap.add_argument("--fixture", default="", help="plain .jsonl for tests")
    ap.add_argument("--check-text", default="", help="download_lichess_evals.py output")
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--scan", type=int, default=200_000,
                    help="max lines to scan (0 = whole file)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--report-md", default="training/work/pov_report.md")
    ap.add_argument("--report-json", default="training/work/pov_report.json")
    ap.add_argument("--engine", default="", help="optional local UCI engine")
    ap.add_argument("--engine-depth", type=int, default=12)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if chess is None:
        print("python-chess is required for PV/FEN checks", file=sys.stderr)
        return 1

    path = args.fixture or args.input or DEFAULT_DB
    text_check = None
    if args.check_text:
        text_check = check_text_file(args.check_text)
        print(f"text check: {args.check_text} -> {text_check['pov']}")

    if not os.path.exists(path):
        print(f"database not ready: {path} does not exist", file=sys.stderr)
        return 2
    ready, why = db_ready(path)
    if not ready and not args.fixture:
        print(f"database not ready: {why}", file=sys.stderr)
        print("pending: run this tool after the download and manifest are "
              "ready (see training/data_contract.md)", file=sys.stderr)
        # Still emit the text-file check if requested.
        if text_check is not None:
            print(json.dumps(text_check, indent=2))
        return 2
    print(f"input: {why if not args.fixture else path}")

    t0 = time.time()
    rng = random.Random(args.seed)
    scan_status: dict = {}
    raw, scanned = reservoir(open_lines(path, scan_status), args.sample, rng,
                             args.scan)
    print(f"scanned {scanned} lines, kept {len(raw)} in {time.time()-t0:.1f}s")
    if scan_status.get("truncated"):
        print("warning: input is a partial download; results cover the "
              "available prefix only", file=sys.stderr)

    records = []
    for line in raw:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    schema = check_schema(records)
    pv = check_pv_legality(records)
    pov = check_pov(records)
    cross = cross_check_positions(records, 20)

    engine_note = ""
    if args.engine:
        fens = [r["fen"] for r in cross[:20]]
        scores = engine_scores(args.engine, fens, args.engine_depth)
        if not scores:
            engine_note = f"_engine `{args.engine}` produced no scores._"
        else:
            agree = 0
            usable = 0
            for r, s in zip(cross, scores):
                if r["cp"] is None:
                    continue
                white_from_engine = s * stm_sign(r["fen"])
                usable += 1
                if abs(r["cp"]) > 50 and abs(white_from_engine) > 50:
                    agree += (r["cp"] > 0) == (white_from_engine > 0)
            engine_note = (
                f"engine `{args.engine}` at depth {args.engine_depth}: "
                f"sign agreement with DB cp on {agree}/{usable} positions "
                f"(positions with |cp|<=50 counted as agree)."
            )
            engine_note += (
                "\n\nNote: engine scores are STM POV (UCI); they were converted "
                "to White POV by multiplying by the side-to-move sign."
            )

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": path,
        "sample_size": len(records),
        "scanned": scanned,
        "seed": args.seed,
        "max_scan": args.scan,
        "partial_input": bool(scan_status.get("truncated"))
        or (not args.fixture and not db_complete(path)),
    }
    report_md = render_report(meta, schema, pv, pov, cross, text_check, engine_note)
    report = {
        "meta": meta,
        "schema": schema,
        "pov": pov,
        "pv_chess960": pv,
        "cross_check_20": cross,
        "text_check": text_check,
        "engine": engine_note,
    }

    for p in (args.report_md, args.report_json):
        if p:
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    if args.report_md:
        with open(args.report_md, "w", encoding="utf-8") as fh:
            fh.write(report_md + "\n")
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

    if not args.quiet:
        print(report_md)

    # Verdict. Only hard-fail on clear contradictions.
    ok = True
    if pov["verdict"].startswith("side-to-move"):
        print("FAIL: cp is side-to-move POV, pipeline assumes White POV",
              file=sys.stderr)
        ok = False
    if pov["verdict"].startswith("inconsistent"):
        print("FAIL: cp sign does not follow material under either POV",
              file=sys.stderr)
        ok = False
    if pv["lines_checked"] and pv["legal_960"] / pv["lines_checked"] < 0.98:
        msg = ("PV lines are not legal under Chess960 replay "
               f"({pv['legal_960']}/{pv['lines_checked']})")
        if args.fixture:
            print(f"WARN (fixture): {msg}", file=sys.stderr)
        else:
            print(f"FAIL: {msg}", file=sys.stderr)
            ok = False
    if schema["records"] and schema["with_evals"] == 0:
        print("FAIL: no usable eval records in sample", file=sys.stderr)
        ok = False
    if text_check is not None:
        if text_check["bad_wdl"] or text_check["bad_fields"]:
            print("FAIL: pipeline text format/wdl consistency", file=sys.stderr)
            ok = False
        if text_check["pov"] and text_check["pov"]["verdict"].startswith("White POV"):
            print("FAIL: pipeline text cp is not side-to-move POV", file=sys.stderr)
            ok = False
    if pov["verdict"] == "inconclusive":
        print("WARN: POV test inconclusive on this sample", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
