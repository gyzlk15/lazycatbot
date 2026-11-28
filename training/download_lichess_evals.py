#!/usr/bin/env python3
"""Stream the Lichess eval database into a compact HalfKP training text file.

Output line format (``halfkp-text-v2``):

    FEN|wdl|cp|bestmove|pv

    FEN       position before the move, 4..6 fields
    wdl       side-to-move win probability in [0,1] = sigmoid(cp/400)
    cp        side-to-move centipawns (integer; White POV negated for Black)
    bestmove  first PV move, standard UCI, "" when unknown
    pv        up to 8 normalized standard-UCI moves, space separated, "" when
              unknown

Selection
  * highest-depth eval per record (ties: first one);
  * drop mates, missing cp, ``depth < --min-depth``, ``|cp| > --max-cp``;
  * 64-bit hash dedup of the FEN (board+stm+castling+ep, move counters
    ignored) with a compact numpy open-addressing set (no Python str sets);
  * optional deterministic sampling: ``--sample-stride`` and/or
    ``--sample-rate`` (CRC32 of the raw line + seed), so resume is exact.

Chess960 normalization
  Lichess PVs use the UCI_Chess960 castling encoding (king-from -> rook-from,
  e.g. ``e1h1``). It is rewritten to standard UCI (``e1g1``/``e1c1``,
  ``e8g8``/``e8c8``) before writing; the policy target uses the normalized move.

Resume
  ``--out`` and ``<out>.resume.json`` are written every few seconds. Re-running
  the same command continues where it stopped: the output file is truncated to
  the checkpoint byte offset, already written FENs are reloaded into the dedup
  table, and decoding restarts from the checkpointed line index. Filters are
  part of the checkpoint signature; a mismatch is an error.

Legacy input
  If the input file is not ``.zst`` it is read as text and both
  ``FEN|target`` (old downloader / quiet-labeled) and
  ``FEN|wdl|cp|bestmove|pv`` lines are accepted.

Examples:
  python training/download_lichess_evals.py --max-positions 3000000 \
      --sample-stride 2 --out training/work/pilot/positions.txt
  python training/download_lichess_evals.py \
      --input training/tests_data/fixture_evals.jsonl \
      --out temp/fixture_positions.txt --min-depth 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zlib

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("numpy is required. Run: .venv\\Scripts\\python.exe -m pip install numpy",
          file=sys.stderr)
    raise SystemExit(1)

try:
    import orjson

    def loads(data):
        return orjson.loads(data)
except ImportError:  # pragma: no cover - fallback
    def loads(data):
        return json.loads(data)

try:
    import zstandard as _zstd

    ZstdError = _zstd.ZstdError
except ImportError:  # pragma: no cover - open_text_lines reports it
    ZstdError = None


URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$")


# --------------------------------------------------------------------------- io


def download(url: str, path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"using existing {path} ({os.path.getsize(path) / 2**30:.2f} GiB)")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"downloading {url}\n  -> {path} (resume supported)")
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-L", "--fail", "--retry", "5", "--retry-delay", "10",
             "-C", "-", "-o", path, url],
            check=True,
        )
    else:
        print("curl not found; using urllib (no resume)", file=sys.stderr)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)


def open_text_lines(path: str, status: dict | None = None,
                    allow_truncated: bool = False):
    """Yield decoded text lines, streaming zstd when needed.

    A truncated ``.zst`` aborts with a clear message unless
    ``--allow-truncated`` is set (used only for provisional sampling of a
    download in progress).
    """
    if path.endswith(".zst"):
        import io

        import zstandard as zstd

        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        try:
            reader = dctx.stream_reader(fh, read_across_frames=True)
        except TypeError:  # older zstandard
            reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="\n")
        try:
            try:
                for line in text:
                    yield line
            except zstd.ZstdError as exc:
                if status is not None:
                    status["truncated"] = True
                    status["truncated_error"] = str(exc)
                if not allow_truncated:
                    print(f"error: compressed input ended early: {exc}\n"
                          f"  (download still running? rerun when complete, or "
                          f"pass --allow-truncated for a provisional sample)",
                          file=sys.stderr)
                    raise
                print(f"warning: stopped at truncated end of input: {exc}",
                      file=sys.stderr)
        finally:
            text.close()
            fh.close()
    else:
        with open(path, "r", encoding="utf-8", errors="replace", newline="\n") as fh:
            for line in fh:
                yield line


# --------------------------------------------------------------- 64-bit hashing


def hash64(text: str) -> int:
    """Deterministic 64-bit hash (process independent, unlike builtin hash)."""
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest(),
        "little",
    )


class U64Set:
    """Compact open-addressing set of uint64 values (linear probing).

    A Python ``set[int]`` costs roughly 60+ bytes per entry; this costs about
    16 bytes per slot at a 0.6 load factor.  Insertions are deterministic and
    a 64-bit hash collision (~1e-7 at 3M entries) only costs one dropped
    duplicate.
    """

    __slots__ = ("keys", "mask", "size")

    def __init__(self, expected: int = 1 << 20):
        cap = 16
        want = max(16, int(expected / 0.6))
        while cap < want:
            cap <<= 1
        self.keys = np.zeros(cap, dtype=np.uint64)
        self.mask = cap - 1
        self.size = 0

    def add(self, h: int) -> bool:
        keys = self.keys
        mask = self.mask
        k = np.uint64(h if h else 1)
        i = h & mask
        while True:
            cur = keys[i]
            if cur == 0:
                keys[i] = k
                self.size += 1
                if self.size * 10 > len(keys) * 6:
                    self._grow()
                return True
            if cur == k:
                return False
            i = (i + 1) & mask

    def _grow(self) -> None:
        old = self.keys
        cap = len(old) * 2
        keys = np.zeros(cap, dtype=np.uint64)
        mask = cap - 1
        for cur in old:
            if cur == 0:
                continue
            i = int(cur) & mask
            while keys[i] != 0:
                i = (i + 1) & mask
            keys[i] = cur
        self.keys = keys
        self.mask = mask


# ---------------------------------------------------------------- chess helpers


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


def validate_fen_light(fen: str) -> bool:
    parts = fen.split()
    if len(parts) < 4 or len(parts) > 6:
        return False
    if parts[1] not in ("w", "b"):
        return False
    board = parse_board(fen)
    if board is None:
        return False
    if board.count("K") != 1 or board.count("k") != 1:
        return False
    castling = parts[2]
    if castling != "-":
        if len(castling) > 4 or any(
            c not in "KQkqABCDEFGHabcdefgh" for c in castling
        ):
            return False
    ep = parts[3]
    if ep != "-" and not re.match(r"^[a-h][36]$", ep):
        return False
    if len(parts) >= 5 and not parts[4].isdigit():
        return False
    if len(parts) >= 6 and not parts[5].isdigit():
        return False
    return True


def side_to_move(fen: str) -> str:
    parts = fen.split()
    return parts[1] if len(parts) > 1 and parts[1] in ("w", "b") else "w"


def normalize_pv(fen: str, tokens, max_moves: int = 8):
    """Normalize a Chess960 UCI PV to standard UCI; drop duplicates/garbage.

    Castling detection tracks which king/rooks have already moved in the PV,
    so a *continuation* king or rook move onto an original back-rank square is
    not mistaken for a castle. Moves alternate colours starting with the FEN
    side to move (Lichess PVs are legal).

    Returns ``(moves, castling_960_count)``.
    """
    board = parse_board(fen)
    if board is None:
        return [], 0
    side = side_to_move(fen)
    kings = {"w": None, "b": None}
    rooks = {"w": set(), "b": set()}
    for sq, ch in enumerate(board):
        if ch == "K":
            kings["w"] = sq
        elif ch == "k":
            kings["b"] = sq
        elif ch == "R":
            rooks["w"].add(sq)
        elif ch == "r":
            rooks["b"].add(sq)
    king_moved = {"w": False, "b": False}
    rook_moved: set[int] = set()
    out: list[str] = []
    seen = set()
    castled = 0
    for i, tok in enumerate(tokens):
        if not isinstance(tok, str) or not UCI_RE.match(tok):
            break
        frm = (int(tok[1]) - 1) * 8 + (ord(tok[0]) - 97)
        to = (int(tok[3]) - 1) * 8 + (ord(tok[2]) - 97)
        mover = side if i % 2 == 0 else ("b" if side == "w" else "w")
        orig_to = to
        is_castle = (
            frm == kings[mover]
            and not king_moved[mover]
            and to in rooks[mover]
            and to not in rook_moved
            and (frm >> 3) == (to >> 3)
        )
        if is_castle:
            rank = to // 8
            to = rank * 8 + (6 if (to % 8) > (frm % 8) else 2)
            if to != orig_to:
                castled += 1
        # Track movement of the original king and rooks (conservative).
        if frm == kings[mover]:
            king_moved[mover] = True
        if frm in rooks[mover]:
            rook_moved.add(frm)
        if is_castle:
            rook_moved.add(orig_to)
        # Any move onto an original rook square (capture by either side, or
        # another piece after the rook left) removes that rook.
        if board[orig_to] in "Rr":
            rook_moved.add(orig_to)
        if frm == to and not is_castle:
            break
        norm = (
            f"{chr(97 + frm % 8)}{frm // 8 + 1}"
            f"{chr(97 + to % 8)}{to // 8 + 1}"
        )
        if len(tok) == 5:
            norm += tok[4].lower()
        if norm in seen:
            # Truncate: dropping one move would break the colour alternation
            # of the stored PV and every later castle/legality decision.
            break
        seen.add(norm)
        out.append(norm)
        if len(out) >= max_moves:
            break
    return out, castled


def first_move_from_friendly(fen: str, move: str) -> bool:
    """Cheap sanity check: the first PV move starts on a friendly piece."""
    if not move or not UCI_RE.match(move):
        return False
    board = parse_board(fen)
    if board is None:
        return False
    frm = (int(move[1]) - 1) * 8 + (ord(move[0]) - 97)
    piece = board[frm]
    return piece.isupper() if side_to_move(fen) == "w" else piece.islower()


def sigmoid_cp(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


# ------------------------------------------------------------------ checkpoints


def load_checkpoint(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def save_checkpoint(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def signature(args) -> dict:
    return {
        "min_depth": args.min_depth,
        "max_cp": args.max_cp,
        "dedup": not args.no_dedup,
        "sample_stride": args.sample_stride,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
        "max_positions": args.max_positions,
    }


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Lichess eval DB -> FEN|wdl|cp|bestmove|pv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-positions", "--positions", dest="max_positions",
                    type=int, default=3_000_000)
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--max-cp", type=int, default=1500)
    ap.add_argument("--sample-stride", type=int, default=1,
                    help="consider every Nth decompressed line (>=1)")
    ap.add_argument("--sample-rate", type=float, default=1.0,
                    help="keep a deterministic fraction of lines (0..1]")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--pv-moves", type=int, default=8)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--input", default="",
                    help="existing .jsonl.zst or plain text file")
    ap.add_argument("--out", default="training/work/lichess_positions.txt")
    ap.add_argument("--cache", default="training/work/lichess_db_eval.jsonl.zst")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--allow-truncated", action="store_true",
                    help="provisional sampling: stop at a truncated .zst "
                         "instead of failing")
    ap.add_argument("--format", choices=["auto", "jsonl", "text"], default="auto",
                    help="input format; auto = .zst -> jsonl, else sniff '{'")
    ap.add_argument("--progress-every", type=float, default=5.0)
    args = ap.parse_args()

    if args.sample_stride < 1:
        ap.error("--sample-stride must be >= 1")
    if not 0.0 < args.sample_rate <= 1.0:
        ap.error("--sample-rate must be in (0, 1]")
    if args.max_positions < 0:
        ap.error("--max-positions must be >= 0")
    if args.pv_moves < 1:
        ap.error("--pv-moves must be >= 1")

    src = args.input or args.cache
    if not args.input and not args.no_download:
        try:
            download(URL, args.cache)
        except Exception as exc:
            print(f"download failed: {exc}", file=sys.stderr)
            return 1
    if not os.path.exists(src):
        print(f"missing input {src}", file=sys.stderr)
        return 1
    src_size = os.path.getsize(src)
    if src_size == 0 or (not args.input and src_size < 1000):
        print(f"{src} is empty or partial (download still running?)", file=sys.stderr)
        return 1

    json_mode = src.endswith(".zst")
    if args.format != "auto":
        json_mode = args.format == "jsonl"
    elif not json_mode:
        try:
            with open(src, "rb") as fh:
                head = fh.read(4096).lstrip()
            json_mode = head.startswith(b"{")
        except OSError:
            pass

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt_path = args.out + ".resume.json"
    sig = signature(args)
    ckpt = None if args.no_resume else load_checkpoint(ckpt_path)
    start_line = 0
    written = 0
    if ckpt:
        if ckpt.get("signature") != sig:
            print(f"checkpoint {ckpt_path} has different filters; "
                  f"rerun with --no-resume to restart", file=sys.stderr)
            return 1
        if not os.path.exists(args.out):
            print(f"checkpoint exists but {args.out} is missing; "
                  f"rerun with --no-resume", file=sys.stderr)
            return 1
        if os.path.getsize(args.out) < ckpt.get("out_bytes", 0):
            print(f"{args.out} is smaller than the checkpoint; "
                  f"rerun with --no-resume", file=sys.stderr)
            return 1
        with open(args.out, "r+b") as fh:
            fh.truncate(ckpt["out_bytes"])
        start_line = int(ckpt["line_index"])
        written = int(ckpt["written"])
        print(f"resuming at line {start_line}, {written} positions written")

    dedup = None if args.no_dedup else U64Set(max(1024, args.max_positions or (1 << 20)))
    if dedup is not None and written:
        # Rebuild exact dedup state from the existing output so a resumed run
        # cannot duplicate an earlier position.
        with open(args.out, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                fen = line.split("|", 1)[0].strip()
                if fen:
                    dedup.add(hash64(" ".join(fen.split()[:4])))

    stats = {
        "lines": 0,
        "sampled_out": 0,
        "accepted": 0,
        "written": 0,
        "dedup_hits": 0,
        "reject": {
            "json": 0, "fen": 0, "no_evals": 0, "no_pvs": 0,
            "mate": 0, "no_cp": 0, "shallow": 0, "cp_range": 0,
        },
        "depth_hist": {},
        "pv_len_hist": {},
        "castling_960_normalized": 0,
        "policy_present": 0,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": src,
        "filters": sig,
        "pov": "white->side_to_move",
        "format": "halfkp-text-v2",
    }
    if ckpt:
        for k, v in ckpt.get("stats", {}).items():
            if k in ("depth_hist", "pv_len_hist", "reject"):
                stats[k] = dict(v)
            elif k not in ("started", "source", "filters", "pov", "format", "lines"):
                stats[k] = v

    out = open(args.out, "a", encoding="utf-8", newline="\n")
    buf: list[str] = []
    input_status: dict = {}
    t0 = time.time()
    last_log = t0
    last_ckpt = t0
    line_index = 0
    rate_num = int(args.sample_rate * 1_000_000)

    def write_checkpoint() -> None:
        out.flush()
        os.fsync(out.fileno())
        stats["lines"] = line_index
        stats["written"] = written
        with open(args.out + ".stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, sort_keys=True)
        save_checkpoint(ckpt_path, {
            "line_index": line_index,
            "written": written,
            "out_bytes": out.tell(),
            "signature": sig,
            "stats": stats,
        })

    try:
        for raw in open_text_lines(src, input_status, args.allow_truncated):
            line_index += 1
            if line_index <= start_line:
                continue
            if args.max_positions and written >= args.max_positions:
                break
            if not raw:
                continue
            if args.sample_stride > 1 and line_index % args.sample_stride:
                continue
            if rate_num < 1_000_000:
                # CRC32 of the raw line: deterministic, fast, seedable.
                if (zlib.crc32(raw.encode("utf-8", "replace"), args.seed)
                        % 1_000_000) >= rate_num:
                    continue
            stats["sampled_out"] += 1
            line = raw.strip()
            if not line:
                continue

            if json_mode:
                parsed = _parse_json_line(line, stats)
            else:
                parsed = _parse_text_line(line, stats)
            if parsed is None:
                continue
            fen, wdl, cp, depth, raw_pv = parsed

            if not validate_fen_light(fen):
                stats["reject"]["fen"] += 1
                continue
            norm, castled = normalize_pv(fen, raw_pv, args.pv_moves)
            best = norm[0] if norm else ""
            if best and not first_move_from_friendly(fen, best):
                best = ""
            key = hash64(" ".join(fen.split()[:4]))
            if dedup is not None and not dedup.add(key):
                stats["dedup_hits"] += 1
                continue

            stats["castling_960_normalized"] += castled
            if best:
                stats["policy_present"] += 1
            if depth:
                d = str(depth)
                stats["depth_hist"][d] = stats["depth_hist"].get(d, 0) + 1
            stats["pv_len_hist"][str(len(norm))] = (
                stats["pv_len_hist"].get(str(len(norm)), 0) + 1
            )
            cp_s = str(int(round(cp))) if math.isfinite(cp) else "nan"
            buf.append(f"{fen}|{wdl:.7f}|{cp_s}|{best}|{' '.join(norm)}\n")
            written += 1
            stats["accepted"] += 1
            if len(buf) >= 8192:
                out.write("".join(buf))
                buf.clear()
            now = time.time()
            if now - last_log >= args.progress_every:
                elapsed = max(now - t0, 1e-9)
                print(
                    f"{written} positions | {line_index} lines | "
                    f"{written / elapsed:.0f} pos/s | {elapsed:.0f}s",
                    flush=True,
                )
                last_log = now
            if now - last_ckpt >= 20.0:
                write_checkpoint()
                last_ckpt = now
    except KeyboardInterrupt:
        print("interrupted; writing checkpoint", file=sys.stderr)
        if buf:
            out.write("".join(buf))
            buf.clear()
        write_checkpoint()
        out.close()
        return 130
    except Exception as exc:
        if ZstdError is not None and isinstance(exc, ZstdError):
            if buf:
                out.write("".join(buf))
                buf.clear()
            write_checkpoint()
            out.close()
            return 1
        raise
    finally:
        if buf:
            out.write("".join(buf))
        if not out.closed:
            out.flush()
            os.fsync(out.fileno())
            out.close()

    stats["lines"] = line_index
    stats["written"] = written
    stats["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stats["elapsed_seconds"] = round(time.time() - t0, 1)
    if input_status.get("truncated"):
        stats["input_truncated"] = True
        stats["input_truncated_error"] = input_status.get("truncated_error")
    with open(args.out + ".stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
    try:
        if not input_status.get("truncated"):
            os.remove(ckpt_path)
    except OSError:
        pass

    dt = time.time() - t0
    if input_status.get("truncated"):
        print(f"stopped: {written} positions from {line_index} lines at the "
              f"truncated end of the input ({dt:.0f}s) -> {args.out}")
    else:
        print(f"done: {written} positions from {line_index} lines in {dt:.0f}s "
              f"-> {args.out}")
    print(f"stats: {args.out}.stats.json")
    return 0 if written else 1


def _depth_of(ev) -> int:
    try:
        return int(ev.get("depth", 0))
    except (TypeError, ValueError):
        return 0


def _parse_json_line(line: str, stats):
    """Lichess JSON record -> (fen, wdl, cp_stm, depth, pv_tokens) or None."""
    try:
        rec = loads(line)
    except (ValueError, TypeError):
        stats["reject"]["json"] += 1
        return None
    if not isinstance(rec, dict):
        stats["reject"]["json"] += 1
        return None
    fen = rec.get("fen")
    if not isinstance(fen, str):
        stats["reject"]["json"] += 1
        return None
    evals = rec.get("evals")
    if not isinstance(evals, list) or not evals:
        stats["reject"]["no_evals"] += 1
        return None
    best = max(evals, key=_depth_of)
    depth = _depth_of(best)
    if depth < int(stats["filters"]["min_depth"]):
        stats["reject"]["shallow"] += 1
        return None
    pvs = best.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        stats["reject"]["no_pvs"] += 1
        return None
    pv = pvs[0]
    if pv.get("mate") is not None:
        stats["reject"]["mate"] += 1
        return None
    cp = pv.get("cp")
    if cp is None:
        stats["reject"]["no_cp"] += 1
        return None
    try:
        cp = float(cp)
    except (TypeError, ValueError):
        stats["reject"]["no_cp"] += 1
        return None
    if not math.isfinite(cp):
        stats["reject"]["no_cp"] += 1
        return None
    if abs(cp) > int(stats["filters"]["max_cp"]):
        stats["reject"]["cp_range"] += 1
        return None
    if side_to_move(fen) == "b":
        cp = -cp  # White POV -> side-to-move POV
    line_str = pv.get("line")
    tokens = line_str.split() if isinstance(line_str, str) else []
    return fen, sigmoid_cp(cp), cp, depth, tokens


def _parse_text_line(line: str, stats):
    """Parse ``FEN|target`` or ``FEN|wdl|cp|bestmove|pv`` text lines."""
    parts = line.split("|")
    if len(parts) == 2:
        # Legacy FEN|target: target is a win probability or a game result.
        fen, target = parts[0].strip(), parts[1].strip()
        try:
            wdl = float(target)
        except ValueError:
            stats["reject"]["json"] += 1
            return None
        if not 0.0 <= wdl <= 1.0:
            stats["reject"]["json"] += 1
            return None
        return fen, wdl, float("nan"), 0, []
    if len(parts) >= 5:
        fen = parts[0].strip()
        try:
            wdl = float(parts[1])
        except ValueError:
            stats["reject"]["json"] += 1
            return None
        if not 0.0 <= wdl <= 1.0:
            stats["reject"]["json"] += 1
            return None
        cp_txt = parts[2].strip().lower()
        if cp_txt in ("", "nan", "none"):
            cp = float("nan")
        else:
            try:
                cp = float(cp_txt)
            except ValueError:
                cp = float("nan")
        pv_tokens = parts[4].split() if parts[4].strip() else []
        return fen, wdl, cp, 0, pv_tokens
    stats["reject"]["json"] += 1
    return None


if __name__ == "__main__":
    raise SystemExit(main())
