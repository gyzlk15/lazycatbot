#!/usr/bin/env python3
"""Validate the C++ NNUE evaluation path against the Python reference.

This tool drives a real UCI engine over stdin/stdout.  For every FEN it sends::

    setoption name UseNNUE value true
    setoption name EvalFile value <model.nnue>
    setoption name EvalBackend value cpu
    position fen <fen>
    eval

and parses the engine's ``info string`` evaluation diagnostics.  It then

* checks the *sign* of the search evaluation on material / mate sanity FENs,
* compares the C++ network value with ``training/reference_inference.py``
  (tolerance 2 cp by default), if a model and the reference module exist,
* compares the network value with the classical evaluation for gross
  disagreements, and
* verifies that the engine did not crash and did not fall back to classical
  silently when a model was requested.

The tool never plays games and never touches the network.  It is safe to run
on a build machine.

Typical use::

    .venv\\Scripts\\python.exe tools/nnue_validate.py \
        --engine build/chessbot.exe \
        --model training/work/model/halfkp_v1.nnue \
        --reference training/reference_inference.py \
        --json temp/nnue_validate.json

A FEN file contains one FEN per line.  Blank lines and ``#`` comments are
ignored; an optional name can be supplied as ``name | fen`` and an optional
deep-search label as ``name | fen | cp``.  With a label and ``|cp| >= 50`` a
sign expectation is added.  Built-in sanity cases are always checked in
addition to ``--fen-file`` entries.

Exit codes: 0 = all checks passed, 1 = validation failures, 2 = setup error
(missing engine/model, engine that cannot load a model, unreadable reference).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINE = os.path.join(ROOT, "build", "chessbot.exe")
ENGINE_CANDIDATES = [
    os.path.join(ROOT, "build", "chessbot.exe"),
    os.path.join(ROOT, "build-baseline", "chessbot_baseline.exe"),
    os.path.join(ROOT, "chessbot.exe"),
    os.path.join(ROOT, "chessbot_res.exe"),
]
DEFAULT_REFERENCE = os.path.join(ROOT, "training", "reference_inference.py")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UciError(RuntimeError):
    """A protocol-level engine problem (timeout, crash or bad output)."""

    def __init__(self, message: str, kind: str = "protocol"):
        super().__init__(message)
        self.kind = kind


class SetupError(RuntimeError):
    """The harness itself cannot run (missing binary, model, reference)."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Engine eval lines look like (current correction-mode NNUE):
#   info string classical(stm) 12 neural -3 total 9 nnue_available 1
#       enabled 1 weight 25 clip 80 direct 0
# A direct-eval build may add/replace keys such as ``net``/``nnue_cp``.
_ALIASES: dict[str, str] = {
    "classical(stm)": "classical",
    "classical": "classical",
    "neural": "neural",
    "nnue": "neural",
    "nnue_cp": "net",
    "nnue_raw": "net",
    "net": "net",
    "total": "total",
    "eval": "total",
    "final": "total",
    "nnue_available": "available",
    "available": "available",
    "nnue_enabled": "enabled",
    "enabled": "enabled",
    "direct": "direct",
    "direct_eval": "direct",
    "weight": "weight",
    "clip": "clip",
}

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_INFO_PREFIX = re.compile(r"^info string\s*", re.IGNORECASE)
_BENCH_RE = re.compile(
    r"\bbench nodes\s+(\d+)\s+time\s+(\d+)\s*ms"
    r"(?:\s+nps\s+(\d+))?",
    re.IGNORECASE,
)


def parse_info_string(line: str) -> dict[str, float]:
    """Parse ``info string`` diagnostics into canonical numeric fields.

    Unknown tokens are ignored.  A key with a non-numeric value (for example a
    file path) is skipped rather than raising.  Only the first occurrence of a
    canonical key is kept.
    """
    text = line.strip()
    match = _INFO_PREFIX.match(text)
    if match:
        text = text[match.end() :]
    elif text.startswith("info "):
        text = text[len("info ") :]
    else:
        raise ValueError(f"not an info line: {line!r}")

    tokens = text.split()
    fields: dict[str, float] = {}
    i = 0
    while i + 1 < len(tokens):
        key = _ALIASES.get(tokens[i].strip().lower())
        if key is None:
            i += 1
            continue
        value = tokens[i + 1]
        if not _NUMBER.match(value):
            i += 1
            continue
        if key not in fields:
            fields[key] = float(value)
        i += 2
    return fields


def is_eval_line(line: str) -> bool:
    """Return True for an ``info string`` line that carries evaluation data."""
    low = line.lower()
    if not low.startswith("info string"):
        return False
    return (
        "classical" in low
        or "nnue_available" in low
        or re.search(r"\btotal\b", low) is not None
        or re.search(r"\bnet\b", low) is not None
    )


def parse_eval_line(line: str) -> dict[str, float]:
    fields = parse_info_string(line)
    if not fields:
        raise ValueError(f"info line has no evaluation fields: {line!r}")
    return fields


def engine_net_cp(
    fields: dict[str, float], source: str = "auto"
) -> Optional[int]:
    """Return the *network* value in centipawns, or None if unavailable.

    ``source`` may be ``auto`` (default), ``net``, ``total`` or ``neural``.
    Use an explicit source when the C++ diagnostics use a different label than
    this adapter expects; ``auto`` prefers:

    1. an explicit ``net``/``nnue_cp`` token;
    2. ``total`` when the engine reports ``direct 1`` (network is the eval);
    3. ``neural`` when the engine is in the legacy correction mode.

    In legacy correction mode ``neural`` is a correction term, not a full
    network evaluation, so callers that compare against the reference must
    also check :func:`is_direct_eval` (or pass an explicit ``source``).
    """
    if source not in ("auto", "net", "total", "neural"):
        raise ValueError(f"unknown net source {source!r}")
    if source != "auto":
        if source not in fields:
            return None
        return int(round(fields[source]))
    if "net" in fields:
        return int(round(fields["net"]))
    direct = fields.get("direct")
    if direct is not None and direct >= 1 and "total" in fields:
        return int(round(fields["total"]))
    enabled = fields.get("enabled")
    if enabled is not None and enabled < 1:
        # NNUE is off: ``neural`` is the zeroed correction term, not a value.
        return None
    if "neural" in fields:
        return int(round(fields["neural"]))
    if "total" in fields:
        return int(round(fields["total"]))
    return None


def is_direct_eval(fields: dict[str, float]) -> bool:
    """True when the engine reports a direct network evaluation."""
    if "net" in fields:
        return True
    return bool(fields.get("direct", 0) >= 1)


def search_eval_cp(fields: dict[str, float]) -> Optional[int]:
    """Return the evaluation the search would use (``total`` if present)."""
    if "total" in fields:
        return int(round(fields["total"]))
    return engine_net_cp(fields)


def parse_bench_line(line: str) -> dict[str, Optional[int]]:
    """Parse ``bench nodes N time T ms [nps X]``."""
    match = _BENCH_RE.search(line)
    if not match:
        raise ValueError(f"not a bench result line: {line!r}")
    nodes = int(match.group(1))
    time_ms = int(match.group(2))
    nps: Optional[int]
    if match.group(3) is not None:
        nps = int(match.group(3))
    elif time_ms > 0:
        nps = (nodes * 1000) // time_ms
    else:
        nps = None
    return {"nodes": nodes, "time_ms": time_ms, "nps": nps}


# ---------------------------------------------------------------------------
# Sanity FEN cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expectation:
    """Sign expectation for a static evaluation, from the side to move."""

    kind: str = "none"  # positive | negative | zero | none
    threshold: float = 0.0

    def check(self, value: Optional[float]) -> bool:
        if self.kind == "none":
            return True
        if value is None:
            return False
        if self.kind == "positive":
            return value > self.threshold
        if self.kind == "negative":
            return value < -self.threshold
        if self.kind == "zero":
            return abs(value) <= self.threshold
        raise ValueError(f"unknown expectation kind {self.kind!r}")

    def describe(self) -> str:
        if self.kind == "none":
            return "record only"
        if self.kind == "positive":
            return f"> {self.threshold:g}"
        if self.kind == "negative":
            return f"< {-self.threshold:g}"
        return f"|x| <= {self.threshold:g}"


@dataclass(frozen=True)
class FenCase:
    name: str
    fen: str
    expectation: Expectation = field(default_factory=Expectation)
    tags: tuple[str, ...] = ()
    label_cp: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            board = chess.Board(self.fen)
        except ValueError as exc:
            raise ValueError(f"{self.name}: invalid FEN {self.fen!r}: {exc}") from exc
        if not board.is_valid():
            raise ValueError(f"{self.name}: FEN is not a legal position: {self.fen!r}")


KQK_W = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
KQK_B_STRONG = "3qk3/8/8/8/8/8/8/4K3 b - - 0 1"
KQK_W_WEAK = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1"

BUILTIN_CASES: list[FenCase] = [
    FenCase(
        "startpos",
        chess.STARTING_FEN,
        Expectation("zero", 120),
        ("material", "balanced"),
    ),
    FenCase(
        "k-vs-k",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        Expectation("zero", 30),
        ("material", "balanced"),
    ),
    FenCase(
        "kqk-white-to-move",
        KQK_W,
        Expectation("positive", 0),
        ("material", "decisive"),
    ),
    FenCase(
        "kqk-black-to-move-strong",
        KQK_B_STRONG,
        Expectation("positive", 0),
        ("material", "decisive", "mirror"),
    ),
    FenCase(
        "kqk-white-to-move-weak",
        KQK_W_WEAK,
        Expectation("negative", 0),
        ("material", "decisive", "mirror"),
    ),
    FenCase(
        "krk-white-to-move",
        "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
        Expectation("positive", 0),
        ("material", "decisive"),
    ),
    FenCase(
        "knk-white-to-move",
        "4k3/8/8/8/8/8/8/N3K3 w - - 0 1",
        Expectation("positive", 0),
        ("material", "decisive"),
    ),
    FenCase(
        "kpvk-white-to-move",
        "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1",
        Expectation("none"),
        ("material",),
    ),
    FenCase(
        "kpvk-black-pawn-white-to-move",
        "4k3/4p3/8/8/8/8/8/4K3 w - - 0 1",
        Expectation("none"),
        ("material",),
    ),
    FenCase(
        "white-down-queen-white-to-move",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        Expectation("negative", 0),
        ("material", "decisive"),
    ),
    FenCase(
        "black-down-queen-black-to-move",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        Expectation("negative", 0),
        ("material", "decisive", "mirror"),
    ),
    FenCase(
        "mate-in-1-white",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        Expectation("positive", 0),
        ("tactical", "mate-in-1"),
    ),
    FenCase(
        "mate-in-1-black",
        "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
        Expectation("positive", 0),
        ("tactical", "mate-in-1", "mirror"),
    ),
    FenCase(
        "e4-e5-opening",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        Expectation("zero", 200),
        ("opening",),
    ),
    FenCase(
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        Expectation("none"),
        ("middlegame", "no-crash"),
    ),
    FenCase(
        "rook-endgame",
        "8/5pk1/6p1/8/8/6P1/5PK1/7R w - - 0 1",
        Expectation("none"),
        ("endgame", "no-crash"),
    ),
]


def parse_fen_cases(text: str, source: str = "<string>") -> list[FenCase]:
    """Parse a FEN list.

    Accepted line formats::

        fen
        name | fen
        name | fen | cp_label      # deep-search cp, side to move

    With a ``cp_label``, a sign expectation is added when ``|cp| >= 50``.
    Deep-search labels can disagree with a static evaluation on tactical
    positions; the report keeps them separate from the C++/Python parity
    check.  Blank lines and ``#`` comments are ignored.
    """
    cases: list[FenCase] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        label: Optional[float] = None
        if len(parts) == 1:
            name = f"{source}:{lineno}"
            fen = parts[0]
        elif len(parts) == 2:
            name = parts[0] or f"{source}:{lineno}"
            fen = parts[1]
        elif len(parts) == 3:
            name = parts[0] or f"{source}:{lineno}"
            fen = parts[1]
            try:
                label = float(parts[2])
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{lineno}: bad cp label {parts[2]!r}"
                ) from exc
        else:
            raise ValueError(f"{source}:{lineno}: too many '|' fields")
        if label is None or abs(label) < 50:
            expectation = Expectation("none")
        else:
            expectation = Expectation(
                "positive" if label > 0 else "negative", 0
            )
        try:
            cases.append(FenCase(name, fen, expectation, (), label))
        except ValueError as exc:
            raise ValueError(f"{source}:{lineno}: {exc}") from exc
    return cases


def load_fen_file(path: str) -> list[FenCase]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    return parse_fen_cases(text, source=os.path.basename(path))


# ---------------------------------------------------------------------------
# Reference inference adapter
# ---------------------------------------------------------------------------


class ReferenceEvaluator:
    """Evaluate FENs with ``training/reference_inference.py`` (or compatible).

    Accepted module APIs, in order:

    1. ``load_nnue(model_path)`` returning an object with ``eval_fen(fen)``
       (this is the actual Agent C module); the result may be a dict with a
       ``cp`` key;
    2. a module-level ``evaluate_fen(fen)`` / ``eval_fen(fen)`` /
       ``predict(fen)`` function returning centipawns;
    3. the module CLI (``python reference_inference.py --model ... --fen ...
       --json``), used as a slow fallback.

    All values are side-to-move centipawns, matching the engine contract.
    """

    _FUNCTION_NAMES = (
        "evaluate_fen",
        "eval_fen",
        "predict_fen",
        "infer_fen",
        "evaluate",
        "predict",
        "infer",
    )

    def __init__(self, path: str, model_path: Optional[str]):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise SetupError(f"reference module not found: {self.path}")
        self.model_path = os.path.abspath(model_path) if model_path else None
        self._cache: dict[str, float] = {}
        self._model = None
        self._model_eval: Optional[Callable[[str], object]] = None
        self._function: Optional[Callable[..., object]] = None
        self._module = None
        self._load()

    def _load(self) -> None:
        spec = importlib.util.spec_from_file_location("nnue_reference_inference", self.path)
        if spec is None or spec.loader is None:
            raise SetupError(f"cannot import reference module: {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - surface any import failure
            raise SetupError(f"reference module failed to import: {exc}") from exc
        self._module = module

        if self.model_path and hasattr(module, "load_nnue"):
            try:
                self._model = module.load_nnue(self.model_path)
            except Exception as exc:  # noqa: BLE001
                raise SetupError(
                    f"reference load_nnue({self.model_path!r}) failed: {exc}"
                ) from exc
            evaluator = getattr(self._model, "eval_fen", None)
            if callable(evaluator):
                self._model_eval = evaluator

        if self._model_eval is None:
            for name in self._FUNCTION_NAMES:
                fn = getattr(module, name, None)
                if callable(fn):
                    self._function = fn
                    break

        if self._model_eval is None and self._function is None:
            # Last resort: the command line interface.
            self._function = self._evaluate_via_cli

    @staticmethod
    def _coerce(value: object) -> float:
        if isinstance(value, dict):
            for key in ("cp", "value", "score", "eval", "nnue_cp"):
                if key in value:
                    return ReferenceEvaluator._coerce(value[key])
            raise ValueError(f"reference dict has no cp/value/score key: {value!r}")
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("reference returned an empty sequence")
            return ReferenceEvaluator._coerce(value[0])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reference returned a non-numeric value: {value!r}")
        if not math.isfinite(float(value)):
            raise ValueError(f"reference returned a non-finite value: {value!r}")
        return float(value)

    def _evaluate_via_cli(self, fen: str) -> float:
        assert self.model_path is not None, "CLI reference needs a model path"
        command = [
            sys.executable,
            self.path,
            "--model",
            self.model_path,
            "--fen",
            fen,
            "--json",
        ]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"reference CLI failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        data = json.loads(proc.stdout)
        if not data or "error" in data[0]:
            raise RuntimeError(f"reference CLI error: {data!r}")
        return self._coerce(data[0])

    def evaluate(self, fen: str) -> float:
        if fen in self._cache:
            return self._cache[fen]
        if self._model_eval is not None:
            value = self._coerce(self._model_eval(fen))
        elif self._function is not None:
            try:
                value = self._coerce(self._function(fen))
            except TypeError:
                if self.model_path is None:
                    raise
                value = self._coerce(self._function(fen, self.model_path))
        else:  # pragma: no cover - __init__ guarantees one path
            raise RuntimeError("reference evaluator is not configured")
        self._cache[fen] = value
        return value

    def describe(self) -> str:
        if self._model_eval is not None:
            return f"{self.path} (load_nnue + eval_fen)"
        if self._function is not None and self._function is not self._evaluate_via_cli:
            return f"{self.path} ({self._function.__name__})"
        return f"{self.path} (CLI fallback)"

    @property
    def is_cli_fallback(self) -> bool:
        return self._function is self._evaluate_via_cli


# ---------------------------------------------------------------------------
# UCI process
# ---------------------------------------------------------------------------


class UciSession:
    """Minimal blocking UCI session used for ``eval``/``bench`` commands."""

    def __init__(
        self,
        path: str,
        *,
        name: str = "engine",
        extra_args: Optional[Iterable[str]] = None,
        startup_timeout: float = 30.0,
        trace: bool = False,
    ):
        self.path = os.fspath(path)
        self.name = name
        self.extra_args = list(extra_args or [])
        self.startup_timeout = float(startup_timeout)
        self.trace = trace
        self.id_name: Optional[str] = None
        self.option_types: dict[str, str] = {}
        self.proc: Optional[subprocess.Popen[str]] = None
        self._queue: Optional[queue.Queue[Optional[str]]] = None
        self._reader: Optional[threading.Thread] = None
        self._start()

    # -- lifecycle ------------------------------------------------------
    def _command(self) -> list[str]:
        if self.path.lower().endswith(".py"):
            command = [sys.executable, self.path]
        else:
            command = [self.path]
        command.extend(self.extra_args)
        return command

    def _start(self) -> None:
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise SetupError(f"{self.name}: cannot start {self.path!r}: {exc}") from exc

        self._queue = queue.Queue()
        proc = self.proc
        assert proc.stdout is not None
        queue_ = self._queue

        def reader() -> None:
            try:
                for line in proc.stdout:
                    if self.trace:
                        print(f"[{self.name} <<] {line.rstrip()}", file=sys.stderr)
                    queue_.put(line.rstrip("\r\n"))
            except Exception:  # noqa: BLE001 - pipe closed during shutdown
                pass
            finally:
                queue_.put(None)

        self._reader = threading.Thread(
            target=reader, name=f"{self.name}-reader", daemon=True
        )
        self._reader.start()
        try:
            self._handshake()
        except Exception:
            self.close(graceful=False)
            raise

    def _write(self, text: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise UciError(f"{self.name}: process is not running", kind="crash")
        if self.proc.poll() is not None:
            raise UciError(
                f"{self.name}: process exited with {self.proc.returncode}",
                kind="crash",
            )
        if self.trace:
            print(f"[{self.name} >>] {text}", file=sys.stderr)
        try:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise UciError(f"{self.name}: write failed ({exc})", kind="crash") from exc

    def send(self, text: str) -> None:
        self._write(text)

    def _read_line(self, timeout: float) -> str:
        assert self._queue is not None
        try:
            line = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty as exc:
            raise UciError(
                f"{self.name}: timeout waiting for output", kind="timeout"
            ) from exc
        if line is None:
            raise UciError(f"{self.name}: process ended", kind="crash")
        return line

    def wait_for(
        self,
        predicate: Callable[[str], bool],
        timeout: float,
        what: str,
        collect: Optional[list[str]] = None,
    ) -> str:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise UciError(
                    f"{self.name}: timeout waiting for {what}", kind="timeout"
                )
            line = self._read_line(left)
            if collect is not None:
                collect.append(line)
            if predicate(line):
                return line

    def _handshake(self) -> None:
        lines: list[str] = []
        self.send("uci")
        self.wait_for(lambda l: l.strip() == "uciok", self.startup_timeout, "uciok", lines)
        for line in lines:
            if line.startswith("id name "):
                self.id_name = line[len("id name ") :].strip()
            match = re.match(r"option name (.+?) type (\w+)", line)
            if match:
                self.option_types[match.group(1)] = match.group(2).lower()

    # -- UCI commands ---------------------------------------------------
    def set_option(self, name: str, value: Optional[str] = None) -> None:
        if value is None:
            self.send(f"setoption name {name}")
        else:
            self.send(f"setoption name {name} value {value}")

    def has_option(self, name: str) -> bool:
        return any(existing.lower() == name.lower() for existing in self.option_types)

    def isready(self, timeout: Optional[float] = None) -> list[str]:
        lines: list[str] = []
        self.send("isready")
        self.wait_for(
            lambda l: l.strip() == "readyok",
            self.startup_timeout if timeout is None else timeout,
            "readyok",
            lines,
        )
        return lines

    def drain(self) -> None:
        """Discard output that arrived while the engine should have been idle."""
        assert self._queue is not None
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                return
            if line is None:
                raise UciError(f"{self.name}: process ended while draining", kind="crash")

    def eval_fen(self, fen: str, timeout: float = 10.0) -> tuple[dict[str, float], list[str]]:
        self.drain()
        self.send(f"position fen {fen}")
        lines: list[str] = []
        self.send("eval")
        line = self.wait_for(is_eval_line, timeout, "eval output", lines)
        return parse_eval_line(line), lines

    def bench(self, depth: int, timeout: float = 600.0) -> tuple[dict[str, Optional[int]], float]:
        self.drain()
        start = time.perf_counter()
        self.send(f"bench {depth}")
        line = self.wait_for(
            lambda l: l.startswith("bench ") and "nodes" in l,
            timeout,
            "bench output",
        )
        elapsed = time.perf_counter() - start
        return parse_bench_line(line), elapsed

    def search_nodes(
        self, fen: str, nodes: int, timeout: float = 600.0
    ) -> dict[str, Optional[int]]:
        """Run ``go nodes N`` on one FEN and return the engine's node count."""
        self.drain()
        self.send(f"position fen {fen}")
        lines: list[str] = []
        self.send(f"go nodes {int(nodes)}")
        bestmove = self.wait_for(
            lambda l: l.startswith("bestmove"), timeout, "bestmove", lines
        )
        searched: Optional[int] = None
        for line in lines:
            match = re.search(r"\bnodes\s+(\d+)", line)
            if match:
                searched = int(match.group(1))
        return {"nodes": searched if searched is not None else int(nodes),
                "bestmove": bestmove}

    # -- shutdown -------------------------------------------------------
    def close(self, graceful: bool = True, timeout: float = 3.0) -> None:
        proc = self.proc
        if proc is None:
            return
        if graceful and proc.poll() is None:
            try:
                self._write("quit")
                proc.wait(timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    proc.wait(timeout=2.0)
                except Exception:  # noqa: BLE001
                    pass
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass
        if self._reader is not None:
            try:
                self._reader.join(timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
        self.proc = None
        self._queue = None


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

_FAILURE_WORDS = (
    "fail",
    "error",
    "invalid",
    "cannot",
    "unable",
    "not found",
    "missing file",
    "could not",
)


def diagnostic_lines(lines: Iterable[str]) -> list[str]:
    return [line for line in lines if line.lower().startswith("info string")]


def diagnostic_is_failure(line: str) -> bool:
    low = line.lower()
    return any(word in low for word in _FAILURE_WORDS)


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    case: FenCase
    fields: dict[str, float]
    search_cp: Optional[int]
    net_cp: Optional[int]
    classical_cp: Optional[int]
    reference_cp: Optional[float]
    delta_reference: Optional[float]
    sign_ok: bool
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "name": self.case.name,
            "fen": self.case.fen,
            "tags": list(self.case.tags),
            "expectation": self.case.expectation.kind,
            "threshold": self.case.expectation.threshold,
            "label_cp": self.case.label_cp,
            "fields": {key: value for key, value in sorted(self.fields.items())},
            "search_cp": self.search_cp,
            "net_cp": self.net_cp,
            "classical_cp": self.classical_cp,
            "reference_cp": self.reference_cp,
            "delta_reference": self.delta_reference,
            "sign_ok": self.sign_ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def discover_engine(explicit: Optional[str]) -> str:
    """Resolve an engine path; raise SetupError with candidates when missing."""
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise SetupError(
                f"engine not found: {path}\n"
                "candidates checked:\n  "
                + "\n  ".join(ENGINE_CANDIDATES)
            )
        return path
    for candidate in ENGINE_CANDIDATES:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise SetupError(
        "no engine binary found; checked:\n  " + "\n  ".join(ENGINE_CANDIDATES)
    )


def _configure_session(
    session: UciSession,
    *,
    model: Optional[str],
    use_nnue: bool,
    backend: str,
    extra_options: Optional[dict[str, str]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """Apply NNUE options and return ``(diagnostics, warnings, errors)``."""
    warnings: list[str] = []
    errors: list[str] = []

    # Caller-supplied options come first so the NNUE-critical settings below
    # (EvalFile/UseNNUE) cannot be accidentally disabled.  A typical use is
    # ``EvalQuant=int16`` to force the reference-exact accumulator path.
    for name, value in (extra_options or {}).items():
        if session.has_option(name):
            session.set_option(name, value)
        else:
            warnings.append(
                f"engine does not advertise option {name!r}; ignoring value {value!r}"
            )

    if model:
        if not session.has_option("EvalFile"):
            errors.append(
                "engine does not advertise an EvalFile option; it cannot load "
                "a .nnue model (Agent D loader pending?)"
            )
        else:
            session.set_option("EvalFile", model)
        if session.has_option("EvalBackend"):
            session.set_option("EvalBackend", backend)
        else:
            warnings.append("engine does not advertise EvalBackend; using its default")

    if session.has_option("UseNNUE"):
        session.set_option("UseNNUE", "true" if use_nnue else "false")
    elif use_nnue:
        warnings.append("engine does not advertise UseNNUE")

    ready_lines = session.isready()
    diagnostics = diagnostic_lines(ready_lines)
    for line in diagnostics:
        if diagnostic_is_failure(line):
            (errors if model else warnings).append(line)
    return diagnostics, warnings, errors


def run_validation(
    engine_path: str,
    *,
    model: Optional[str] = None,
    cases: Optional[Iterable[FenCase]] = None,
    reference_path: Optional[str] = None,
    tolerance: float = 2.0,
    use_nnue: Optional[bool] = None,
    backend: str = "cpu",
    net_source: str = "auto",
    extra_options: Optional[dict[str, str]] = None,
    eval_timeout: float = 10.0,
    startup_timeout: float = 30.0,
    require_reference: bool = False,
    trace: bool = False,
) -> dict:
    """Run the validation and return a JSON-serialisable report."""
    engine = discover_engine(engine_path)
    model_abs = os.path.abspath(os.path.expanduser(model)) if model else None
    if model_abs is not None and not os.path.isfile(model_abs):
        raise SetupError(f"model file not found: {model_abs}")

    if use_nnue is None:
        use_nnue = model_abs is not None

    all_cases = list(BUILTIN_CASES) + list(cases or [])
    report: dict = {
        "engine": engine,
        "engine_name": None,
        "model": model_abs,
        "use_nnue": bool(use_nnue),
        "backend": backend,
        "net_source": net_source,
        "extra_options": dict(extra_options or {}),
        "tolerance_cp": float(tolerance),
        "reference": os.path.abspath(reference_path) if reference_path else None,
        "reference_available": False,
        "reference_mode": None,
        "startup_diagnostics": [],
        "warnings": [],
        "errors": [],
        "cases": [],
        "summary": {},
    }

    reference: Optional[ReferenceEvaluator] = None
    if reference_path:
        if not os.path.isfile(reference_path):
            raise SetupError(f"reference module not found: {reference_path}")
        if model_abs is None:
            report["warnings"].append(
                "reference inference requires a model; reference comparison skipped"
            )
        elif not use_nnue:
            report["warnings"].append(
                "reference inference requires UseNNUE=true; reference comparison skipped"
            )
        else:
            reference = ReferenceEvaluator(reference_path, model_abs)
            report["reference_available"] = True
            report["reference_mode"] = reference.describe()
            if reference.is_cli_fallback:
                report["warnings"].append(
                    "reference module has no usable in-process API; using slow "
                    "CLI fallback (one process per FEN)"
                )

    if require_reference and model_abs is not None and use_nnue and reference is None:
        report["errors"].append(
            "a reference module is required but none was configured/found"
        )

    session: Optional[UciSession] = None
    try:
        session = UciSession(
            engine,
            startup_timeout=startup_timeout,
            trace=trace,
        )
        report["engine_name"] = session.id_name or os.path.basename(engine)
        diagnostics, warnings, setup_errors = _configure_session(
            session,
            model=model_abs,
            use_nnue=bool(use_nnue),
            backend=backend,
            extra_options=extra_options,
        )
        report["startup_diagnostics"] = diagnostics
        report["warnings"].extend(warnings)
        report["errors"].extend(setup_errors)

        if setup_errors and model_abs is not None:
            # Without EvalFile there is nothing to validate.  Return early
            # instead of emitting a report full of misleading classical rows.
            report["summary"] = _summarise([], report)
            return report

        results: list[CaseResult] = []
        for case in all_cases:
            try:
                fields, lines = session.eval_fen(case.fen, timeout=eval_timeout)
            except UciError as exc:
                results.append(
                    CaseResult(
                        case=case,
                        fields={},
                        search_cp=None,
                        net_cp=None,
                        classical_cp=None,
                        reference_cp=None,
                        delta_reference=None,
                        sign_ok=False,
                        errors=[f"eval failed: {exc}"],
                    )
                )
                continue

            search_cp = search_eval_cp(fields)
            net_cp = engine_net_cp(fields, net_source)
            classical_cp = (
                int(round(fields["classical"])) if "classical" in fields else None
            )
            sign_ok = case.expectation.check(search_cp)
            errors: list[str] = []
            notes: list[str] = []

            for line in diagnostic_lines(lines):
                if diagnostic_is_failure(line):
                    errors.append(f"engine diagnostic: {line}")
                else:
                    notes.append(line)

            if not sign_ok:
                errors.append(
                    f"sign check failed: expected {case.expectation.describe()}, "
                    f"got {search_cp}"
                )
            if (
                case.expectation.kind in ("positive", "negative")
                and sign_ok
                and search_cp is not None
                and abs(search_cp) < 100
                and classical_cp is not None
                and abs(classical_cp) >= 200
            ):
                notes.append(
                    "network compresses material: "
                    f"{search_cp} cp vs classical {classical_cp} cp"
                )
            if case.expectation.kind != "none" and "total" not in fields:
                notes.append("engine did not report a total evaluation")

            if model_abs is not None:
                enabled = fields.get("enabled")
                if enabled is not None and enabled < 1:
                    errors.append(
                        "engine reports enabled=0 after UseNNUE=true; model not active"
                    )
                available = fields.get("available")
                if available is not None and available < 1:
                    errors.append(
                        "engine reports nnue_available=0 after loading a model"
                    )

            # Gross disagreement between the classical and network values is
            # a useful smoke signal, but not a hard failure on quiet positions.
            if (
                classical_cp is not None
                and search_cp is not None
                and case.expectation.kind in ("positive", "negative")
            ):
                classical_ok = case.expectation.check(classical_cp)
                network_ok = case.expectation.check(search_cp)
                if classical_ok != network_ok:
                    notes.append(
                        "classical and NNUE evaluations disagree on the expected sign "
                        f"(classical={classical_cp}, eval={search_cp})"
                    )
                if abs(search_cp - classical_cp) > 2000:
                    notes.append(
                        f"large classical/eval gap: {search_cp - classical_cp} cp"
                    )

            reference_cp: Optional[float] = None
            delta_reference: Optional[float] = None
            if reference is not None and net_cp is not None:
                comparable = is_direct_eval(fields) or net_source != "auto"
                if not comparable:
                    notes.append(
                        "engine is in legacy correction mode (direct=0); the "
                        "network correction is not a full evaluation, so the "
                        "reference comparison was skipped (use --net-source "
                        "total/neural to force a field)"
                    )
                else:
                    try:
                        reference_cp = reference.evaluate(case.fen)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"reference evaluation failed: {exc}")
                    else:
                        delta_reference = net_cp - reference_cp
                        if abs(delta_reference) > tolerance:
                            errors.append(
                                f"reference mismatch: engine {net_cp} cp vs reference "
                                f"{reference_cp:.2f} cp (|delta|={abs(delta_reference):.2f} "
                                f"> {tolerance:g})"
                            )
            elif reference is not None and net_cp is None:
                notes.append("no network value reported; reference comparison skipped")

            results.append(
                CaseResult(
                    case=case,
                    fields=fields,
                    search_cp=search_cp,
                    net_cp=net_cp,
                    classical_cp=classical_cp,
                    reference_cp=reference_cp,
                    delta_reference=delta_reference,
                    sign_ok=sign_ok,
                    errors=errors,
                    notes=notes,
                )
            )

        report["cases"] = [result.as_dict() for result in results]
        if reference is not None:
            compared = [r for r in results if r.reference_cp is not None]
            if not compared and model_abs is not None:
                report["errors"].append(
                    "no case produced a comparable direct network value; the "
                    "C++/Python reference check did not run"
                )
        report["summary"] = _summarise(results, report)
    finally:
        if session is not None:
            session.close()

    return report


def _summarise(results: list[CaseResult], report: dict) -> dict:
    passed = sum(1 for result in results if result.ok)
    failed = len(results) - passed
    compared = [result for result in results if result.reference_cp is not None]
    mismatched = [
        result
        for result in compared
        if result.delta_reference is not None
        and abs(result.delta_reference) > report.get("tolerance_cp", 2.0)
    ]
    max_delta = max(
        (abs(result.delta_reference) for result in compared if result.delta_reference is not None),
        default=None,
    )
    return {
        "cases": len(results),
        "passed": passed,
        "failed": failed,
        "warnings": len(report.get("warnings", [])),
        "errors": len(report.get("errors", [])),
        "reference_compared": len(compared),
        "reference_failed": len(mismatched),
        "max_abs_reference_delta": max_delta,
        "ok": failed == 0 and not report.get("errors"),
    }


def print_report(report: dict, verbose: bool = False) -> None:
    print("=== NNUE validation ===")
    print(f"engine:    {report['engine']}")
    if report.get("engine_name"):
        print(f"id:        {report['engine_name']}")
    print(f"model:     {report.get('model') or '(none; classical only)'}")
    print(
        f"mode:      {'NNUE' if report.get('use_nnue') else 'classical'}"
        f" / backend {report.get('backend')}"
        f" / net-source {report.get('net_source', 'auto')}"
    )
    if report.get("reference"):
        print(f"reference: {report['reference']}")
        print(f"           {report.get('reference_mode') or 'unavailable'}")
    print()

    header = f"{'case':34s} {'eval':>7s} {'net':>7s} {'class':>7s} {'ref':>8s} {'delta':>7s}  status"
    print(header)
    print("-" * len(header))
    for item in report.get("cases", []):
        def fmt(value, width=7):
            if value is None:
                return f"{'-':>{width}s}"
            if isinstance(value, float):
                return f"{value:>{width}.1f}"
            return f"{value:>{width}d}"

        status = "ok" if not item["errors"] else "FAIL"
        print(
            f"{item['name'][:34]:34s} "
            f"{fmt(item['search_cp'])} {fmt(item['net_cp'])} "
            f"{fmt(item['classical_cp'])} {fmt(item['reference_cp'], 8)} "
            f"{fmt(item['delta_reference'])}  {status}"
        )
        if verbose:
            for note in item.get("notes", []):
                print(f"    note: {note}")
            for error in item.get("errors", []):
                print(f"    error: {error}")

    summary = report.get("summary", {})
    print()
    print("--- summary ---")
    print(
        f"cases: {summary.get('cases', 0)}  passed: {summary.get('passed', 0)}  "
        f"failed: {summary.get('failed', 0)}"
    )
    if report.get("reference_available"):
        print(
            f"reference compared: {summary.get('reference_compared', 0)}  "
            f"failed: {summary.get('reference_failed', 0)}  "
            f"max |delta|: {summary.get('max_abs_reference_delta')}"
        )
    else:
        print("reference compared: 0 (model or reference unavailable)")
    if report.get("warnings"):
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    if report.get("errors"):
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")
    if report.get("startup_diagnostics"):
        print("startup diagnostics:")
        for line in report["startup_diagnostics"]:
            print(f"  {line}")
    print()
    if summary.get("ok"):
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a UCI engine's NNUE evaluation against the Python "
        "reference and material sanity FENs."
    )
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="path to the UCI engine")
    parser.add_argument(
        "--model",
        default=None,
        help="path to a .nnue v1 model; omit for a classical-only smoke run",
    )
    parser.add_argument(
        "--fen-file",
        default=None,
        help="extra FENs, one per line (optional 'name | fen' or "
        "'name | fen | deep_cp')",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help=f"Python reference module (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="do not load the Python reference even if it exists",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="maximum |C++ - Python| in centipawns (default 2)",
    )
    parser.add_argument(
        "--nnue", dest="use_nnue", action="store_true", default=None,
        help="force UseNNUE=true",
    )
    parser.add_argument(
        "--classical", dest="use_nnue", action="store_false",
        help="force UseNNUE=false (sanity only)",
    )
    parser.add_argument("--backend", default="cpu", choices=["cpu", "none"])
    parser.add_argument(
        "--net-source",
        default="auto",
        choices=["auto", "net", "total", "neural"],
        help="which info-string field holds the network value (default: auto)",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra UCI option applied before EvalFile (repeatable), e.g. "
        "EvalQuant=int16",
    )
    parser.add_argument("--eval-timeout", type=float, default=10.0, help="seconds per eval")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--require-reference",
        action="store_true",
        help="fail when a model is supplied but the reference module is missing",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="write the full report as JSON to this path",
    )
    parser.add_argument("--verbose", action="store_true", help="print notes and diagnostics")
    parser.add_argument("--trace", action="store_true", help="dump raw UCI traffic")
    return parser


def _parse_uci_options(items: Iterable[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"bad UCI option {item!r}; expected NAME=VALUE")
        options[name.strip()] = value.strip()
    return options


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        extra_options = _parse_uci_options(args.option)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    reference_path: Optional[str] = None
    if not args.no_reference:
        if args.reference:
            reference_path = args.reference
        elif os.path.isfile(DEFAULT_REFERENCE):
            reference_path = DEFAULT_REFERENCE

    # The Python reference is only meaningful when a model is supplied.
    if args.model is None:
        reference_path = None
    elif reference_path is not None and not os.path.isfile(reference_path):
        if args.require_reference:
            print(
                f"error: reference module not found: {reference_path}",
                file=sys.stderr,
            )
            return 2
        print(
            f"warning: reference module not found: {reference_path}; "
            "reference comparison skipped",
            file=sys.stderr,
        )
        reference_path = None

    if (
        args.require_reference
        and args.model
        and (reference_path is None or not os.path.isfile(reference_path))
    ):
        print(
            "error: --require-reference was set but the reference module is missing",
            file=sys.stderr,
        )
        return 2

    extra_cases: list[FenCase] = []
    if args.fen_file:
        try:
            extra_cases = load_fen_file(args.fen_file)
        except (OSError, ValueError) as exc:
            print(f"error: cannot read FEN file: {exc}", file=sys.stderr)
            return 2
        if not extra_cases:
            print(f"error: no FENs in {args.fen_file}", file=sys.stderr)
            return 2

    try:
        report = run_validation(
            args.engine,
            model=args.model,
            cases=extra_cases,
            reference_path=reference_path,
            tolerance=args.tolerance,
            use_nnue=args.use_nnue,
            backend=args.backend,
            net_source=args.net_source,
            extra_options=extra_options,
            eval_timeout=args.eval_timeout,
            startup_timeout=args.startup_timeout,
            require_reference=args.require_reference,
            trace=args.trace,
        )
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except UciError as exc:
        print(f"error: engine protocol failure: {exc}", file=sys.stderr)
        return 2

    print_report(report, verbose=args.verbose)

    if args.json:
        path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"json: {path}")

    if report.get("errors"):
        return 2
    return 0 if report["summary"].get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
