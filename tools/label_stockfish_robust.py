#!/usr/bin/env python3
"""Robust Stockfish static-eval labeler.

Like tools/label_stockfish.py, but restarts Stockfish on a crash/timeout and
skips the offending FEN instead of aborting the whole batch. Writes output
incrementally and logs progress. Local-only.
"""
from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
import time

import chess

FINAL_RE = re.compile(r"Final evaluation\s+([+-]?\d+(?:\.\d+)?)\s+\(white side\)")


class SfEval:
    def __init__(self, path: str):
        self.path = path
        self.proc = None
        self.q = None
        self.reader = None
        self.start()

    def start(self):
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        self.q = queue.Queue()
        self.reader = threading.Thread(target=self._reader, daemon=True)
        self.reader.start()
        self.send("uci")
        self.wait_for("uciok", 60)
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 32")
        self.send("isready")
        self.wait_for("readyok", 60)

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line.rstrip("\r\n"))
        self.q.put(None)

    def send(self, s: str):
        self.proc.stdin.write(s + "\n")
        self.proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: float):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.q.get(timeout=max(0.01, deadline - time.time()))
            if line is None:
                raise RuntimeError("stockfish ended")
            if line.startswith(prefix):
                return line
        raise TimeoutError(prefix)

    def evaluate(self, fen: str, timeout: float = 5.0):
        self.send(f"position fen {fen}")
        self.send("eval")
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.q.get(timeout=max(0.01, deadline - time.time()))
            if line is None:
                raise RuntimeError("stockfish ended")
            m = FINAL_RE.search(line)
            if m:
                return float(m.group(1))
        raise TimeoutError(fen)

    def close(self):
        if self.proc is None:
            return
        try:
            self.send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def restart(self):
        self.close()
        self.start()


def read_fens(path: str):
    seen = set()
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            fen = line.split("|", 1)[0].strip()
            if not fen or fen in seen:
                continue
            try:
                b = chess.Board(fen)
            except ValueError:
                continue
            if not b.is_valid() or b.is_game_over() or b.is_check():
                continue
            seen.add(fen)
            out.append(fen)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    fens = read_fens(args.input)
    if args.limit:
        fens = fens[: args.limit]
    print(f"loaded {len(fens)} positions", flush=True)

    sf = SfEval(args.stockfish)
    written = skipped = 0
    skipped_fens = []
    t0 = time.time()
    with open(args.output, "w", encoding="utf-8", newline="\n") as out:
        for i, fen in enumerate(fens):
            try:
                val = sf.evaluate(fen, timeout=args.timeout)
            except Exception as exc:
                skipped += 1
                skipped_fens.append((fen, str(exc)))
                print(f"skip at {i}: {exc}: {fen}", flush=True)
                try:
                    sf.restart()
                except Exception as exc2:
                    print(f"restart failed: {exc2}", file=sys.stderr, flush=True)
                    break
                continue
            cp = max(-2000, min(2000, int(round(val * 100))))
            if chess.Board(fen).turn == chess.BLACK:
                cp = -cp
            out.write(f"{fen}|{cp}\n")
            written += 1
            if (i + 1) % 1000 == 0:
                out.flush()
                print(
                    f"{i+1}/{len(fens)} written={written} skipped={skipped} "
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )
    sf.close()
    print(f"done: {written} written, {skipped} skipped -> {args.output}", flush=True)
    if skipped_fens:
        with open(args.output + ".skipped.txt", "w", encoding="utf-8") as f:
            for fen, why in skipped_fens:
                f.write(f"{fen}\t{why}\n")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
