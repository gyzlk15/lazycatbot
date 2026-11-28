#!/usr/bin/env python3
"""End-to-end tests for the HalfKP data pipeline (synthetic fixtures only).

Run from the repository root:

  .venv\\Scripts\\python.exe -m pytest training/tests_data -q

The extractor tests need ``training/work/extract_halfkp.exe`` (or the path in
``HALFKP_EXTRACTOR``); when it is missing they skip with the build command
instead of failing, so the downloader / verifier / stats tests still run on a
machine without MSVC.

No network access: every test passes ``--no-download``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "training"
HERE = Path(__file__).resolve().parent
PY = sys.executable
DOWNLOADER = TRAIN / "download_lichess_evals.py"
VERIFIER = TRAIN / "verify_lichess_pov.py"
STATS = TRAIN / "data_stats.py"
EXTRACTOR = Path(os.environ.get("HALFKP_EXTRACTOR",
                                str(TRAIN / "work" / "extract_halfkp.exe")))
FIXTURE_JSONL = HERE / "fixture_evals.jsonl"
FIXTURE_LEGACY = HERE / "fixture_legacy.txt"
FIXTURE_RESUME = HERE / "fixture_resume.txt"

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
         chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
POLICY_ABSENT = 0xFFFF


# --------------------------------------------------------------------- helpers


def run(script: Path, *args: str, expect: int = 0) -> str:
    proc = subprocess.run(
        [str(PY), str(script), *args],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != expect:
        raise AssertionError(
            f"{script.name} {' '.join(args)} -> {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def run_exe(exe: Path, *args: str, expect: int = 0) -> str:
    proc = subprocess.run([str(exe), *args], cwd=str(ROOT),
                          capture_output=True, text=True)
    if proc.returncode != expect:
        raise AssertionError(
            f"{exe} {' '.join(args)} -> {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def read_positions(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            fen, wdl, cp, best, pv = line.split("|")
            rows.append({"fen": fen, "wdl": float(wdl),
                         "cp": None if cp == "nan" else int(cp),
                         "best": best, "pv": pv.split()})
    return rows


def material(fen: str) -> int:
    board = fen.split()[0]
    total = 0
    for ch in board:
        low = ch.lower()
        if low in "pnbrqk":
            total += VALUE[chess.Piece.from_symbol(low).piece_type] * (
                1 if ch.isupper() else -1)
    return total


def fnv1a32(text: str) -> int:
    h = 2166136261
    for b in text.encode():
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def expected_selection(records, min_depth=20, max_cp=1500, limit=0):
    """Independent implementation of the documented downloader rules."""
    seen = set()
    out = []
    for rec in records:
        if limit and len(out) >= limit:
            break
        fen = rec.get("fen", "")
        parts = fen.split()
        if len(parts) < 4 or parts[1] not in ("w", "b"):
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if not board.is_valid():
            continue
        evals = rec.get("evals") or []
        if not evals:
            continue
        best = max(evals, key=lambda e: int(e.get("depth", 0)))
        if int(best.get("depth", 0)) < min_depth:
            continue
        pvs = best.get("pvs") or []
        if not pvs:
            continue
        pv = pvs[0]
        if pv.get("mate") is not None or pv.get("cp") is None:
            continue
        cp_white = int(pv["cp"])
        if abs(cp_white) > max_cp:
            continue
        key = " ".join(parts[:4])
        if key in seen:
            continue
        seen.add(key)
        cp_stm = cp_white if board.turn == chess.WHITE else -cp_white
        wdl = 1.0 / (1.0 + np.exp(-cp_stm / 400.0))
        out.append({"fen": fen, "cp": cp_stm, "wdl": float(wdl),
                    "pv": (pv.get("line") or "").split()})
    return out


def first_move_legal(fen: str, token: str) -> bool:
    try:
        board = chess.Board(fen, chess960=True)
        mv = board.parse_uci(token)
        return board.is_legal(mv)
    except ValueError:
        return False


def reference_policy(fen: str, tokens):
    """Independent reimplementation of the extractor's policy/PV encoding."""
    board = chess.Board(fen, chess960=True)
    chars = ["."] * 64
    for sq, pc in board.piece_map().items():
        chars[sq] = pc.symbol()
    side = board.turn
    kings = {chess.WHITE: board.king(chess.WHITE),
             chess.BLACK: board.king(chess.BLACK)}
    rooks = {
        chess.WHITE: {sq for sq, p in board.piece_map().items()
                      if p.color == chess.WHITE and p.piece_type == chess.ROOK},
        chess.BLACK: {sq for sq, p in board.piece_map().items()
                      if p.color == chess.BLACK and p.piece_type == chess.ROOK},
    }
    king_moved = {chess.WHITE: False, chess.BLACK: False}
    rook_moved: set[int] = set()
    pv = []
    seen = set()
    for i, tok in enumerate(tokens[:8]):
        if i == 0 and not first_move_legal(fen, tok):
            break
        try:
            mv = chess.Move.from_uci(tok)
        except ValueError:
            break
        frm, to, promo = mv.from_square, mv.to_square, mv.promotion
        mover = side if i % 2 == 0 else not side
        orig_to = to
        is_castle = (
            frm == kings[mover]
            and not king_moved[mover]
            and to in rooks[mover]
            and to not in rook_moved
            and frm // 8 == to // 8
        )
        if is_castle:
            to = (to // 8) * 8 + (6 if to % 8 > frm % 8 else 2)
        if frm == kings[mover]:
            king_moved[mover] = True
        if frm in rooks[mover]:
            rook_moved.add(frm)
        if is_castle:
            rook_moved.add(orig_to)
        # A move onto an original rook square removes that rook (capture or a
        # different piece arriving after the rook left).
        if chars[orig_to] in "Rr":
            rook_moved.add(orig_to)
        idx = (4096 + (promo - 2) * 64 + frm) if promo else (frm * 64 + to)
        if i > 0 and idx in seen:
            # Truncate on a duplicate: dropping one move breaks the colour
            # alternation of the rest of the PV.
            break
        seen.add(idx)
        pv.append(idx)
    return (pv[0] if pv else POLICY_ABSENT), pv


def reference_features(fen: str):
    board = chess.Board(fen)
    us = board.turn
    king = board.king(us)
    white = us == chess.WHITE
    bucket = king if white else king ^ 56
    feats = []
    for sq, pc in board.piece_map().items():
        if pc.color == us and pc.piece_type == chess.KING:
            continue
        osq = sq if white else sq ^ 56
        rel = 0 if pc.color == us else 1
        feats.append((bucket * 12 + rel * 6 + (pc.piece_type - 1)) * 64 + osq)
    return sorted(feats)


def load_shards(root: Path):
    feats = np.fromfile(root / "features.u16", dtype=np.uint16).reshape(-1, 32)
    n = len(np.fromfile(root / "counts.u8", dtype=np.uint8))
    return {
        "n": n,
        "features": feats,
        "counts": np.fromfile(root / "counts.u8", dtype=np.uint8),
        "targets": np.fromfile(root / "targets.f32", dtype=np.float32),
        "cp": np.fromfile(root / "cp.f32", dtype=np.float32),
        "moves": np.fromfile(root / "moves.u16", dtype=np.uint16),
        "pv": np.fromfile(root / "pv.u16", dtype=np.uint16).reshape(n, 8),
        "groups": np.fromfile(root / "groups.u32", dtype=np.uint32),
    }


def import_downloader():
    spec = importlib.util.spec_from_file_location("dl", DOWNLOADER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ downloader


def test_fixtures_present():
    assert FIXTURE_JSONL.exists() and FIXTURE_LEGACY.exists()
    assert FIXTURE_RESUME.exists()


def test_downloader_selection_matches_reference(tmp_path):
    out = tmp_path / "positions.txt"
    run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(out),
        "--no-download")
    rows = read_positions(out)
    records = [json.loads(line) for line in
               FIXTURE_JSONL.read_text(encoding="utf-8").splitlines()]
    expected = expected_selection(records)
    assert len(rows) == len(expected), (len(rows), len(expected))
    for got, want in zip(rows, expected):
        assert got["fen"] == want["fen"]
        assert got["cp"] == want["cp"], (got["fen"], got["cp"], want["cp"])
        assert abs(got["wdl"] - want["wdl"]) < 1e-6
    # Every line must have exactly five fields and a sane wdl.
    for row in rows:
        assert 0.0 <= row["wdl"] <= 1.0
        if row["cp"] is not None:
            assert abs(row["wdl"] - 1.0 / (1.0 + np.exp(-row["cp"] / 400.0))) < 2e-6
        assert len(row["pv"]) <= 8


def test_downloader_pov_conversion(tmp_path):
    out = tmp_path / "positions.txt"
    run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(out),
        "--no-download")
    rows = read_positions(out)
    records = [json.loads(line) for line in
               FIXTURE_JSONL.read_text(encoding="utf-8").splitlines()]
    by_fen = {r["fen"]: r for r in rows}
    checked = 0
    for rec in records:
        board = chess.Board(rec["fen"]) if "evals" in rec and rec["evals"] else None
        if board is None or board.turn != chess.BLACK:
            continue
        if rec["fen"] not in by_fen:
            continue
        pv = max(rec["evals"], key=lambda e: int(e.get("depth", 0)))["pvs"][0]
        if pv.get("cp") is None:
            continue
        assert by_fen[rec["fen"]]["cp"] == -int(pv["cp"])
        checked += 1
    assert checked >= 5


def test_downloader_chess960_and_promotion(tmp_path):
    out = tmp_path / "positions.txt"
    run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(out),
        "--no-download")
    rows = read_positions(out)
    text = "\n".join(" ".join(r["pv"]) for r in rows)
    # Both castling encodings from the fixtures must be normalized.
    assert "e1g1" in text and "e8g8" in text
    assert "e1h1" not in text and "e8h8" not in text
    promos = [t for r in rows for t in r["pv"] if len(t) == 5]
    assert "a7a8q" in promos


def test_downloader_legacy_input(tmp_path):
    out = tmp_path / "legacy.txt"
    run(DOWNLOADER, "--input", str(FIXTURE_LEGACY), "--out", str(out),
        "--no-download")
    rows = read_positions(out)
    assert len(rows) == 6
    assert sum(1 for r in rows if r["cp"] is None) == 3
    # The v2 line's two castles (black then white) are both normalized.
    assert "e8g8" in " ".join(rows[3]["pv"])
    assert "e1g1" in " ".join(rows[3]["pv"])
    # Promotion and the duplicate PV token.
    assert rows[4]["pv"][0] == "a7a8q"
    assert rows[4]["pv"].count("a7a8q") == 1
    # Regression: the final e1h1 follows king/rook movement in the PV and is
    # an ordinary king move, not a castle, so it must not become e1g1.
    assert rows[5]["pv"][0] == "h1h5"
    assert rows[5]["pv"][-1] == "e1h1"
    stats = json.loads((tmp_path / "legacy.txt.stats.json").read_text("utf-8"))
    assert stats["reject"]["json"] == 0
    assert stats["reject"]["fen"] == 1
    assert stats["castling_960_normalized"] == 2


def test_downloader_sampling_is_deterministic(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    c = tmp_path / "c.txt"
    for p in (a, b, c):
        run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(p),
            "--no-download", "--sample-stride", "3")
    assert a.read_text("utf-8") == b.read_text("utf-8")
    records = [json.loads(x) for x in
               FIXTURE_JSONL.read_text("utf-8").splitlines()]
    # The downloader keeps lines 3, 6, 9, ... (1-based line index % stride).
    assert len(read_positions(a)) == len(expected_selection(records[2::3]))


def test_downloader_resume(tmp_path):
    out = tmp_path / "resume.txt"
    run(DOWNLOADER, "--input", str(FIXTURE_RESUME), "--out", str(out),
        "--no-download", "--max-positions", "0")
    full = out.read_text("utf-8")
    assert full.count("\n") == 24

    lines = full.splitlines(keepends=True)
    partial = "".join(lines[:10])
    out.write_text(partial, encoding="utf-8", newline="\n")
    dl = import_downloader()
    ns = argparse.Namespace(min_depth=20, max_cp=1500, no_dedup=False,
                            sample_stride=1, sample_rate=1.0, seed=1234,
                            max_positions=0)
    ckpt = {
        "line_index": 10,
        "written": 10,
        "out_bytes": len(partial.encode("utf-8")),
        "signature": dl.signature(ns),
        "stats": {},
    }
    Path(str(out) + ".resume.json").write_text(json.dumps(ckpt), encoding="utf-8")
    run(DOWNLOADER, "--input", str(FIXTURE_RESUME), "--out", str(out),
        "--no-download", "--max-positions", "0")
    assert out.read_text("utf-8") == full
    assert not Path(str(out) + ".resume.json").exists()


def test_u64set_dedup():
    dl = import_downloader()
    s = dl.U64Set(1024)
    values = [dl.hash64(f"fen-{i}") for i in range(2000)]
    for i, h in enumerate(values):
        assert s.add(h) == (i < len(set(values[: i + 1])))
    for h in values:
        assert not s.add(h)


# -------------------------------------------------------------------- extractor


needs_exe = pytest.mark.skipif(
    not EXTRACTOR.exists(),
    reason=f"extractor not built ({EXTRACTOR}); build it with "
           f"cmd /c \"...vcvars64.bat\" && cl /std:c++20 /O2 /EHsc /I src "
           f"training\\extract_halfkp.cpp src\\bitboards.cpp src\\position.cpp "
           f"src\\eval.cpp src\\nnue_weights.cpp src\\nnue.cpp "
           f"/Fe:training\\work\\extract_halfkp.exe",
)


@needs_exe
def test_extractor_selftest():
    src = TRAIN / "extract_halfkp.cpp"
    if src.exists():
        assert EXTRACTOR.stat().st_mtime >= src.stat().st_mtime, (
            f"{EXTRACTOR} is older than {src}; rebuild it before trusting "
            f"the shard tests (a failed build leaves the old exe in place)"
        )
    out = run_exe(EXTRACTOR, "--selftest")
    assert "SELFTEST OK" in out


@needs_exe
def test_extractor_shards_match_reference(tmp_path):
    text = tmp_path / "positions.txt"
    shards = tmp_path / "halfkp"
    shards.mkdir()
    run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(text),
        "--no-download")
    run_exe(EXTRACTOR, str(text), str(shards),
            "--source-stats", str(text) + ".stats.json")
    rows = read_positions(text)
    sh = load_shards(shards)
    assert sh["n"] == len(rows)
    assert len(sh["groups"]) == sh["n"]
    assert len(sh["targets"]) == sh["n"]

    for i, row in enumerate(rows):
        want = reference_features(row["fen"])
        n = int(sh["counts"][i])
        got = sorted(int(x) for x in sh["features"][i, :n])
        assert got == want, (i, row["fen"], got, want)
        assert np.all(sh["features"][i, n:] == 0)
        assert abs(float(sh["targets"][i]) - np.float32(row["wdl"])) < 1e-6
        if row["cp"] is None:
            assert not np.isfinite(sh["cp"][i])
        else:
            assert np.float32(row["cp"]) == sh["cp"][i]

        want_move, want_pv = reference_policy(row["fen"], row["pv"])
        assert int(sh["moves"][i]) == want_move, (i, row["fen"], row["pv"])
        got_pv = [int(x) for x in sh["pv"][i]]
        assert got_pv[: len(want_pv)] == want_pv
        assert all(x == POLICY_ABSENT for x in got_pv[len(want_pv):])
        want_group = fnv1a32(" ".join(row["fen"].split()[:4]))
        assert int(sh["groups"][i]) == want_group

    meta = json.loads((shards / "meta.json").read_text("utf-8"))
    assert meta["count"] == len(rows)
    assert meta["format"] == "halfkp-v1"
    assert meta["inputs"] == 49152
    assert meta["policy_size"] == 4352
    assert meta["pov"]["output"] == "side_to_move"
    assert meta["source_stats"]["written"] == len(rows)
    assert meta["castling_960_normalized"] >= 0


@needs_exe
def test_extractor_legacy_and_invalid(tmp_path):
    shards = tmp_path / "halfkp"
    shards.mkdir()
    text = tmp_path / "mixed.txt"
    text.write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|0.5\n"
        "8/8/8/8/8/8/8/8 w - - 0 1|0.5\n"
        "not-a-fen|0.5\n"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1|0.5|20|"
        "e2e4|e2e4 e7e5\n",
        encoding="utf-8", newline="\n",
    )
    run_exe(EXTRACTOR, str(text), str(shards))
    sh = load_shards(shards)
    assert sh["n"] == 2
    assert int(sh["moves"][0]) == POLICY_ABSENT       # legacy has no policy
    assert int(sh["moves"][1]) == 12 * 64 + 28        # e2e4
    assert int(sh["pv"][1][0]) == 12 * 64 + 28
    assert int(sh["cp"][1]) == 20
    meta = json.loads((shards / "meta.json").read_text("utf-8"))
    assert meta["skip_reasons"]["bad_fen"] == 2
    assert meta["skip_reasons"]["bad_fields"] == 0


@needs_exe
def test_extractor_cp_mode(tmp_path):
    shards = tmp_path / "halfkp"
    shards.mkdir()
    text = tmp_path / "cp.txt"
    text.write_text("8/8/8/8/8/8/8/K6k w - - 0 1|400\n", encoding="utf-8")
    run_exe(EXTRACTOR, str(text), str(shards), "--cp")
    sh = load_shards(shards)
    assert abs(float(sh["targets"][0]) - 1.0 / (1.0 + np.exp(-1.0))) < 1e-6
    assert int(sh["cp"][0]) == 400


# ------------------------------------------------------------------ data stats


@needs_exe
def test_data_stats_pass_and_detects_duplicates(tmp_path):
    text = tmp_path / "positions.txt"
    shards = tmp_path / "halfkp"
    shards.mkdir()
    run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(text),
        "--no-download")
    run_exe(EXTRACTOR, str(text), str(shards))
    report = tmp_path / "stats.json"
    run(STATS, "--data", str(shards), "--json", str(report), "--quiet")
    st = json.loads(report.read_text("utf-8"))
    assert st["ok"], [c for c in st["checks"] if not c["ok"]]
    assert st["count"] == len(read_positions(text))
    assert st["policy"]["coverage"] and st["policy"]["coverage"] > 0.9
    assert st["pv"]["coverage"] and st["pv"]["coverage"] > 0.9
    assert st["split"]["group_overlap"] == 0

    # Duplicate a line on purpose: this is a real duplicate (35 rows cannot
    # produce a 32-bit collision), so the duplicate-budget check must fail.
    lines = text.read_text("utf-8").splitlines()
    dupdir = tmp_path / "dup_shards"
    dupdir.mkdir()
    dup_text = tmp_path / "dup.txt"
    dup_text.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
    run_exe(EXTRACTOR, str(dup_text), str(dupdir))
    report2 = tmp_path / "stats2.json"
    run(STATS, "--data", str(dupdir), "--json", str(report2), "--quiet",
        expect=1)
    st2 = json.loads(report2.read_text("utf-8"))
    assert not st2["ok"]
    assert st2["groups"]["duplicate_rows"] == 1
    assert st2["groups"]["duplicate_rate"] > 0
    assert st2["groups"]["excess_duplicates"] >= 0.99
    failed = [c["name"] for c in st2["checks"] if not c["ok"]]
    assert any("duplicate groups" in name for name in failed)
    assert st2["split"]["group_overlap"] == 0

    # External holdout overlap: one shared position, one new one.
    holdout = tmp_path / "holdout.epd"
    holdout.write_text(
        f"{lines[0].split('|')[0]} bm e2e4;\n"
        "8/8/8/8/8/8/8/K6k w - - 0 1 bm a1a2;\n",
        encoding="utf-8",
    )
    report3 = tmp_path / "stats3.json"
    run(STATS, "--data", str(shards), "--holdout", str(holdout),
        "--json", str(report3), "--quiet")
    st3 = json.loads(report3.read_text("utf-8"))
    assert st3["holdout"]["positions"] == 2
    assert st3["holdout"]["overlapping_positions"] == 1
    assert st3["holdout"]["leakage_detected"] is True


def test_data_stats_rejects_broken_shards(tmp_path):
    shards = tmp_path / "bad"
    shards.mkdir()
    (shards / "counts.u8").write_bytes(b"\x01\x02")
    (shards / "features.u16").write_bytes(b"\x00" * 64)  # should be 2*32
    for name in ("targets.f32", "cp.f32", "moves.u16", "groups.u32"):
        (shards / name).write_bytes(b"\x00" * 8)
    (shards / "pv.u16").write_bytes(b"\xff" * 32)
    proc = subprocess.run([str(PY), str(STATS), "--data", str(shards)],
                          cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 1
    assert "disagree" in proc.stderr or "expected" in proc.stderr


# ---------------------------------------------------------------------- pov


def test_verify_pov_on_fixture(tmp_path):
    report = tmp_path / "pov.json"
    run(VERIFIER, "--fixture", str(FIXTURE_JSONL), "--scan", "0",
        "--report-md", str(tmp_path / "pov.md"),
        "--report-json", str(report), "--quiet")
    r = json.loads(report.read_text("utf-8"))
    assert r["pov"]["verdict"].startswith("White POV")
    assert r["pv_chess960"]["castling_960_tokens"] >= 2
    assert r["pv_chess960"]["standard_fail_lines"] >= 2
    assert len(r["cross_check_20"]) == 20
    assert r["schema"]["fen_valid"] == r["schema"]["records"]


def test_verify_pov_synthetic_side_to_move_fixture(tmp_path):
    """A deliberate side-to-move fixture must be rejected, not accepted."""
    records = []
    for line in FIXTURE_JSONL.read_text("utf-8").splitlines():
        rec = json.loads(line)
        if not rec.get("evals"):
            records.append(rec)
            continue
        board = chess.Board(rec["fen"]) if len(rec["fen"].split()) >= 4 else None
        if board is None:
            records.append(rec)
            continue
        for ev in rec["evals"]:
            for pv in ev.get("pvs", []):
                if pv.get("cp") is not None:
                    # White POV -> side-to-move POV.
                    sign = 1 if board.turn == chess.WHITE else -1
                    pv["cp"] = int(pv["cp"]) * sign
        records.append(rec)
    path = tmp_path / "stm.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                    encoding="utf-8")
    report = tmp_path / "stm.json"
    proc = subprocess.run(
        [str(PY), str(VERIFIER), "--fixture", str(path), "--scan", "0",
         "--report-json", str(report), "--report-md", "", "--quiet"],
        cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 1
    assert "side-to-move" in proc.stderr.lower()
    r = json.loads(report.read_text("utf-8"))
    assert r["pov"]["verdict"].startswith("side-to-move")


def test_verify_check_text(tmp_path):
    text = tmp_path / "positions.txt"
    run(DOWNLOADER, "--input", str(FIXTURE_JSONL), "--out", str(text),
        "--no-download")
    report = tmp_path / "pov.json"
    run(VERIFIER, "--fixture", str(FIXTURE_JSONL), "--scan", "0",
        "--check-text", str(text), "--report-json", str(report),
        "--report-md", "", "--quiet")
    r = json.loads(report.read_text("utf-8"))
    assert r["text_check"]["pov"]["verdict"].startswith("side-to-move POV")
    assert r["text_check"]["bad_wdl"] == 0
    assert r["text_check"]["bad_fields"] == 0


# -------------------------------------------------------------------- manifest


def test_pilot_manifest_tool(tmp_path):
    root = tmp_path / "ds"
    (root / "halfkp").mkdir(parents=True)
    (root / "positions.txt").write_text("fen|0.5|0||\n", encoding="utf-8")
    (root / "positions.txt.stats.json").write_text(
        json.dumps({"written": 1, "lines": 1}), encoding="utf-8")
    (root / "halfkp" / "meta.json").write_text(
        json.dumps({"count": 1, "skipped": 0}), encoding="utf-8")
    (root / "halfkp" / "counts.u8").write_bytes(b"\x01")
    tool = HERE / "make_pilot_manifest.py"
    run(tool, "--dir", str(root), "--final-path", "training/work/pilot",
        "--source", str(tmp_path / "missing.zst"))
    m = json.loads((root / "manifest.json").read_text("utf-8"))
    assert m["status"] == "stable"
    assert m["path"] == "training/work/pilot"
    assert m["counts"]["shard_rows"] == 1
    paths = {f["path"] for f in m["files"]}
    assert "positions.txt" in paths and "halfkp/meta.json" in paths
    # Recompute one hash to ensure the manifest is self-consistent.
    import hashlib
    f = next(x for x in m["files"] if x["path"] == "positions.txt")
    assert hashlib.sha256((root / "positions.txt").read_bytes()).hexdigest() == f["sha256"]
