#!/usr/bin/env python3
"""Local engine-vs-engine match harness for the NNUE pilot.

This is a thin wrapper around ``tools/arena.py``: it reuses the hardened UCI
process wrapper, ``play_game`` (full move history, mirrored colour pairs,
timeout recovery) and the Elo/CI helpers, then adds:

* a preflight that proves the NNUE process actually loaded the requested model
  (``EvalFile`` advertised, ``UseNNUE`` enabled, no ``info string`` load error);
* fixed-nodes *and* fixed-movetime search modes;
* W/D/L from the candidate's point of view, score, Elo estimate with a pair
  level confidence interval, and a likelihood-of-superiority (LOS) estimate;
* an optional sequential SPRT stop (normal approximation, documented in
  ``docs/reports/VALIDATION.md``).

Only local binaries are started.  No Lichess, online bot or network game is
used anywhere in this tool.

Examples::

    .venv\\Scripts\\python.exe tools/nnue_sprt.py ^
        --baseline build-baseline/chessbot_baseline.exe ^
        --nnue build/chessbot.exe ^
        --model training/work/model/halfkp_v1.nnue ^
        --pairs 50 --nodes 8000 --json temp/nnue_sprt_nodes.json

    .venv\\Scripts\\python.exe tools/nnue_sprt.py --pairs 50 --movetime 100

Exit codes: 0 = completed without engine errors, 1 = validation/match failed
(including engine errors), 2 = fatal engine/setup error, 3 = ``--strict`` and
at least one engine error/timeout/illegal move occurred.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

from nnue_validate import (
    DEFAULT_ENGINE,
    SetupError,
    UciError,
    UciSession,
    diagnostic_is_failure,
    diagnostic_lines,
    discover_engine,
    is_direct_eval,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = os.path.join(ROOT, "build-baseline", "chessbot_baseline.exe")


def _load_arena():
    """Load ``tools/arena.py`` deterministically (not any installed 'arena')."""
    path = os.path.join(ROOT, "tools", "arena.py")
    spec = importlib.util.spec_from_file_location("lazycat_arena", path)
    if spec is None or spec.loader is None:
        raise SetupError(f"cannot load arena module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arena = _load_arena()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


# ---------------------------------------------------------------------------
# Search controls
# ---------------------------------------------------------------------------


@dataclass
class NodeControl:
    """Duck-typed ``TimeControl`` for ``go nodes N``.

    ``arena.play_game`` only needs ``mode``, ``go_command`` and
    ``hard_timeout``; ``mode='movetime'`` disables its clock bookkeeping.
    """

    nodes: int
    move_timeout: float = 60.0
    mode: str = "movetime"

    def go_command(self, clocks) -> str:
        return f"go nodes {int(self.nodes)}"

    def hard_timeout(self, clocks, turn) -> float:
        return max(0.05, float(self.move_timeout))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight_engine(
    path: str,
    options: dict[str, str],
    *,
    model: Optional[str] = None,
    startup_timeout: float = 30.0,
) -> dict:
    """Load an engine, apply options and prove NNUE is active.

    Returns a report dict; raises :class:`SetupError` on a fatal problem.
    """
    session: Optional[UciSession] = None
    report: dict = {
        "path": path,
        "id": None,
        "options": dict(options),
        "eval": {},
        "diagnostics": [],
        "warnings": [],
        "errors": [],
    }
    try:
        session = UciSession(path, startup_timeout=startup_timeout)
        report["id"] = session.id_name
        if model is not None and not session.has_option("EvalFile"):
            raise SetupError(
                "candidate engine does not advertise EvalFile; the NNUE model "
                "cannot be loaded. Run tools/nnue_validate.py first."
            )
        for name, value in options.items():
            if not session.has_option(name):
                if name == "EvalBackend":
                    continue
                if name == "UseNNUE" and value.strip().lower() in ("false", "0"):
                    # A baseline may predate the option; NNUE is already off.
                    continue
                raise SetupError(f"engine does not advertise option {name!r}")
            session.set_option(name, value)
        ready_lines = session.isready()
        report["diagnostics"] = diagnostic_lines(ready_lines)
        failures = [
            line for line in report["diagnostics"] if diagnostic_is_failure(line)
        ]
        if failures:
            raise SetupError(
                "engine reported a load failure: " + "; ".join(failures)
            )

        fields, _ = session.eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        report["eval"] = fields
        if model is not None:
            if fields.get("enabled", 1) < 1:
                raise SetupError("engine reports UseNNUE disabled after setoption")
            if fields.get("available", 1) < 1:
                raise SetupError("engine reports nnue_available=0 after loading a model")
            if not is_direct_eval(fields):
                report["warnings"].append(
                    "engine eval is not direct (direct=0); it may be using the "
                    "legacy NNUE correction path. Verify with nnue_validate.py "
                    "before trusting SPRT results."
                )
        return report
    except UciError as exc:
        raise SetupError(f"preflight failed for {path}: {exc}") from exc
    finally:
        if session is not None:
            session.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def los_from_scores(scores: list[float]) -> Optional[float]:
    """P(true score > 50%) under a normal approximation, from game scores."""
    n = len(scores)
    if n == 0:
        return None
    mean = sum(scores) / n
    if n < 2:
        return 1.0 if mean > 0.5 else (0.0 if mean < 0.5 else 0.5)
    variance = sum((score - mean) ** 2 for score in scores) / (n - 1)
    if variance <= 0.0:
        return 1.0 if mean > 0.5 else (0.0 if mean < 0.5 else 0.5)
    standard_error = math.sqrt(variance / n)
    return normal_cdf((mean - 0.5) / standard_error)


def pair_scores(records: list) -> list[float]:
    groups: dict[int, list[float]] = {}
    for record in records:
        groups.setdefault(record.pair_index, []).append(record.a_score)
    return [sum(values) / len(values) for values in groups.values() if len(values) == 2]


def sprt_llr(
    scores: list[float],
    *,
    elo0: float,
    elo1: float,
    sigma2: float = 0.25,
) -> dict:
    """Sequential log-likelihood ratio for H0: Elo=elo0 vs H1: Elo=elo1.

    Uses a per-game normal approximation with a fixed variance.  This is a
    pragmatic gate, not a replacement for a long, properly calibrated SPRT;
    see the limitations in ``docs/reports/VALIDATION.md``.
    """
    if elo1 <= elo0:
        raise ValueError("SPRT requires elo1 > elo0")
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    p0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    p1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    llr = 0.0
    for score in scores:
        llr += (((score - p0) ** 2) - ((score - p1) ** 2)) / (2.0 * sigma2)
    return {
        "llr": llr,
        "p0": p0,
        "p1": p1,
        "sigma2": sigma2,
        "games": len(scores),
    }


def sprt_decision(
    llr: float, *, alpha: float, beta: float
) -> tuple[str, float, float]:
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    if llr >= upper:
        return "accept_h1", lower, upper
    if llr <= lower:
        return "accept_h0", lower, upper
    return "continue", lower, upper


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


@dataclass
class MatchConfig:
    baseline: str
    candidate: str
    model: Optional[str]
    pairs: int
    nodes: Optional[int]
    movetime: Optional[int]
    move_timeout: float
    timeout_extra: float
    max_plies: int
    opening_offset: int
    openings: list[str]
    candidate_options: dict[str, str] = field(default_factory=dict)
    baseline_options: dict[str, str] = field(default_factory=dict)
    candidate_sha256: str = ""
    baseline_sha256: str = ""
    pgn: Optional[str] = None
    strict: bool = False
    sprt: bool = False
    sprt_elo0: float = 0.0
    sprt_elo1: float = 5.0
    sprt_alpha: float = 0.05
    sprt_beta: float = 0.05
    startup_timeout: float = 30.0


def run_match(
    config: MatchConfig,
    *,
    candidate_preflight: Optional[dict] = None,
    baseline_preflight: Optional[dict] = None,
) -> dict:
    candidate_options = dict(config.candidate_options)
    baseline_options = dict(config.baseline_options)

    if candidate_preflight is None:
        candidate_preflight = preflight_engine(
            config.candidate,
            candidate_options,
            model=config.model,
            startup_timeout=config.startup_timeout,
        )
    if baseline_preflight is None:
        baseline_preflight = preflight_engine(
            config.baseline,
            baseline_options,
            model=None,
            startup_timeout=config.startup_timeout,
        )

    if config.nodes is not None:
        tc = NodeControl(config.nodes, config.move_timeout)
        time_description = f"nodes {config.nodes}/move (hard timeout {config.move_timeout:g}s)"
    else:
        assert config.movetime is not None
        tc = arena.TimeControl(
            mode="movetime",
            movetime=config.movetime,
            timeout_extra=config.timeout_extra,
        )
        time_description = f"{config.movetime} ms/move"

    total_games = config.pairs * 2
    pair_count = config.pairs
    openings = config.openings
    records: list = []
    stopped_by_sprt = False
    sprt_history: list[dict] = []

    a: Optional[arena.UciEngine] = None
    b: Optional[arena.UciEngine] = None
    pgn_fp = None
    fatal: Optional[Exception] = None

    try:
        a = arena.UciEngine(
            config.candidate,
            candidate_options,
            "A",
            startup_timeout=config.startup_timeout,
        )
        b = arena.UciEngine(
            config.baseline,
            baseline_options,
            "B",
            startup_timeout=config.startup_timeout,
        )
        print("=== NNUE SPRT-style match (local only) ===")
        print(f"candidate (A): {a.display_name}")
        print(f"baseline  (B): {b.display_name}")
        print(f"model:         {config.model or '(none)'}")
        print(f"time control:  {time_description}")
        print(f"games:         {total_games} ({pair_count} mirrored pairs)")
        print()

        if config.pgn:
            pgn_fp = open(config.pgn, "w", encoding="utf-8", newline="\n")

        for pair_index in range(pair_count):
            opening_text = openings[
                (pair_index + config.opening_offset) % len(openings)
            ]
            opening_moves = opening_text.split()
            pair_start = len(records)
            print(
                f"pair {pair_index + 1}/{pair_count} opening: "
                f"{opening_text or 'startpos'}",
                flush=True,
            )
            for a_white in (True, False):
                if a is None or b is None:
                    raise RuntimeError("engines not initialized")
                a.new_game()
                b.new_game()
                if a_white:
                    white, black = a, b
                    white_key, black_key = "A", "B"
                else:
                    white, black = b, a
                    white_key, black_key = "B", "A"
                record = arena.play_game(
                    game_no=len(records) + 1,
                    pair_no=pair_index + 1,
                    pair_index=pair_index,
                    white=white,
                    black=black,
                    white_key=white_key,
                    black_key=black_key,
                    a_white=a_white,
                    opening_moves=opening_moves,
                    tc=tc,
                    max_plies=config.max_plies,
                    verbose=False,
                )
                records.append(record)
                w = sum(1 for r in records if r.a_score == 1.0)
                d = sum(1 for r in records if r.a_score == 0.5)
                l = sum(1 for r in records if r.a_score == 0.0)
                print(
                    f"  game {record.game_no}/{total_games} "
                    f"{'A' if record.a_white else 'B'} as White: "
                    f"{record.result:7s} {record.plies:3d} plies "
                    f"{record.elapsed:6.1f}s  "
                    f"A=+{w} ={d} -{l}  ({record.reason})",
                    flush=True,
                )
                if pgn_fp is not None:
                    arena.write_pgn(
                        pgn_fp,
                        record,
                        event=f"nnue_sprt {a.display_name} vs {b.display_name}",
                    )

            if config.sprt and len(records) > pair_start:
                scores = [r.a_score for r in records]
                info = sprt_llr(
                    scores,
                    elo0=config.sprt_elo0,
                    elo1=config.sprt_elo1,
                )
                decision, lower, upper = sprt_decision(
                    info["llr"], alpha=config.sprt_alpha, beta=config.sprt_beta
                )
                info.update(
                    {
                        "decision": decision,
                        "lower": lower,
                        "upper": upper,
                        "pairs": pair_index + 1,
                    }
                )
                sprt_history.append(info)
                print(
                    f"  SPRT: LLR {info['llr']:+.3f} "
                    f"(bounds {lower:+.3f} .. {upper:+.3f}) -> {decision}",
                    flush=True,
                )
                if decision != "continue":
                    stopped_by_sprt = True
                    break
    except arena.EngineError as exc:
        fatal = exc
        print(f"\nfatal engine error: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        fatal = exc
        print(f"\nfatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if a is not None:
            a.quit()
        if b is not None:
            b.quit()
        if pgn_fp is not None:
            try:
                pgn_fp.close()
            except Exception:  # noqa: BLE001
                pass

    summary = summarise(
        records,
        candidate_preflight=candidate_preflight,
        baseline_preflight=baseline_preflight,
        config=config,
        time_description=time_description,
        stopped_by_sprt=stopped_by_sprt,
        sprt_history=sprt_history,
        fatal=fatal,
        engines=(a, b),
    )
    print_match_summary(summary)
    return summary


def summarise(
    records: list,
    *,
    candidate_preflight: dict,
    baseline_preflight: dict,
    config: MatchConfig,
    time_description: str,
    stopped_by_sprt: bool,
    sprt_history: list[dict],
    fatal: Optional[Exception],
    engines: tuple,
) -> dict:
    n = len(records)
    wins = sum(1 for r in records if r.a_score == 1.0)
    draws = sum(1 for r in records if r.a_score == 0.5)
    losses = sum(1 for r in records if r.a_score == 0.0)
    score_sum = sum(r.a_score for r in records)
    score = score_sum / n if n else 0.0
    elo = arena.elo_from_score(score) if n else None
    game_scores = [r.a_score for r in records]
    pairs = pair_scores(records)

    los_game = los_from_scores(game_scores)
    los_pair = los_from_scores(pairs) if len(pairs) >= 2 else None

    wilson = arena.wilson_interval(score_sum, n) if n else (0.0, 1.0)
    pair_ci = None
    if len(pairs) >= 2:
        mean = sum(pairs) / len(pairs)
        variance = sum((value - mean) ** 2 for value in pairs) / (len(pairs) - 1)
        se = math.sqrt(variance / len(pairs))
        tcrit = arena.t_critical_975(len(pairs) - 1)
        pair_ci = (
            max(0.0, mean - tcrit * se),
            min(1.0, mean + tcrit * se),
        )

    sprt_final = sprt_history[-1] if sprt_history else None
    errors = [r for r in records if r.error_engine]

    return {
        "candidate": {
            "path": config.candidate,
            "id": candidate_preflight.get("id"),
            "sha256": config.candidate_sha256,
            "preflight": candidate_preflight,
            "options": dict(candidate_preflight.get("options", {})),
        },
        "baseline": {
            "path": config.baseline,
            "id": baseline_preflight.get("id"),
            "sha256": config.baseline_sha256,
            "preflight": baseline_preflight,
            "options": dict(baseline_preflight.get("options", {})),
        },
        "model": config.model,
        "time_control": time_description,
        "nodes": config.nodes,
        "movetime_ms": config.movetime,
        "games": n,
        "pairs": len(pairs),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "score_sum": score_sum,
        "elo": elo,
        "wilson_95": list(wilson),
        "pair_ci_95": list(pair_ci) if pair_ci else None,
        "los_game": los_game,
        "los_pair": los_pair,
        "sprt": {
            "enabled": config.sprt,
            "elo0": config.sprt_elo0,
            "elo1": config.sprt_elo1,
            "alpha": config.sprt_alpha,
            "beta": config.sprt_beta,
            "stopped": stopped_by_sprt,
            "final": sprt_final,
            "history": sprt_history,
        },
        "engine_errors": len(errors),
        "engine_error_details": [
            {
                "game": r.game_no,
                "engine": r.error_engine,
                "kind": r.error_kind,
                "detail": r.error_detail,
            }
            for r in errors
        ],
        "restarts": {
            "A": engines[0].restarts if engines[0] is not None else 0,
            "B": engines[1].restarts if engines[1] is not None else 0,
        },
        "fatal": str(fatal) if fatal is not None else None,
        "records": [
            {
                "game": r.game_no,
                "pair": r.pair_no,
                "opening": r.opening,
                "a_white": r.a_white,
                "result": r.result,
                "reason": r.reason,
                "plies": r.plies,
                "elapsed_s": round(r.elapsed, 2),
                "a_score": r.a_score,
                "error_engine": r.error_engine,
                "error_kind": r.error_kind,
                "san": r.san_moves,
            }
            for r in records
        ],
        "ok": fatal is None and not errors,
    }


def print_match_summary(summary: dict) -> None:
    print("\n=== summary ===")
    print(f"candidate (A): {summary['candidate']['path']} [{summary['candidate']['id']}]")
    print(f"baseline  (B): {summary['baseline']['path']} [{summary['baseline']['id']}]")
    if summary["candidate"].get("sha256"):
        print(f"candidate sha256: {summary['candidate']['sha256']}")
    if summary["baseline"].get("sha256"):
        print(f"baseline  sha256: {summary['baseline']['sha256']}")
    print(f"model:         {summary['model'] or '(none)'}")
    print(f"time control:  {summary['time_control']}")
    for label, preflight in (
        ("candidate", summary["candidate"]["preflight"]),
        ("baseline", summary["baseline"]["preflight"]),
    ):
        for warning in preflight.get("warnings", []):
            print(f"  {label} warning: {warning}")

    n = summary["games"]
    if n == 0:
        print("no completed games")
        print("\nRESULT: FAIL")
        return
    print(f"games: {n}  complete mirrored pairs: {summary['pairs']}")
    print(
        f"A W/D/L: +{summary['wins']} ={summary['draws']} "
        f"-{summary['losses']}   score {summary['score_sum']:.2f}/{n} = "
        f"{summary['score']:.3f}"
    )
    if summary["elo"] is None:
        print("approx Elo difference (A-B): unbounded (A scored 0% or 100%)")
    else:
        print(
            f"approx Elo difference (A-B): {arena.fmt_elo(summary['elo'])}"
        )
    lo, hi = summary["wilson_95"]
    print(
        f"95% Wilson score interval (naive, game-level): "
        f"{lo:.3f} .. {hi:.3f}  -> Elo {arena.fmt_ci(lo, hi)}"
    )
    if summary["pair_ci_95"]:
        lo, hi = summary["pair_ci_95"]
        print(
            f"95% pair-level score CI ({summary['pairs']} pairs): "
            f"{lo:.3f} .. {hi:.3f}  -> Elo {arena.fmt_ci(lo, hi)}"
        )
    else:
        print("95% pair-level CI: not available (need >=2 complete pairs)")
    if summary["los_game"] is not None:
        print(f"LOS (game-level, normal approx): {summary['los_game'] * 100.0:.1f}%")
    if summary["los_pair"] is not None:
        print(f"LOS (pair-level, normal approx): {summary['los_pair'] * 100.0:.1f}%")

    sprt = summary["sprt"]
    if sprt["enabled"] and sprt["final"] is not None:
        final = sprt["final"]
        print(
            f"SPRT (H0 {sprt['elo0']:g}, H1 {sprt['elo1']:g}, "
            f"alpha {sprt['alpha']:g}, beta {sprt['beta']:g}): "
            f"LLR {final['llr']:+.3f} -> {final['decision']}"
            + (" (stopped early)" if sprt["stopped"] else " (max pairs reached)")
        )
    elif sprt["enabled"]:
        print("SPRT: not evaluated")

    if summary["engine_errors"]:
        print(f"engine errors: {summary['engine_errors']}")
        for detail in summary["engine_error_details"]:
            print(
                f"  game {detail['game']}: {detail['engine']} "
                f"[{detail['kind']}] {detail['detail']}"
            )
    print(
        f"restarts: A={summary['restarts']['A']} B={summary['restarts']['B']}"
    )
    if n < 30:
        print(
            "\nSAMPLE-SIZE CAVEAT: fewer than 30 games. Treat Elo, LOS and the "
            "confidence intervals as indicative only; the pilot gate needs "
            "hundreds of games."
        )
    print()
    print("RESULT: PASS" if summary["ok"] else "RESULT: FAIL")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local NNUE candidate vs classical baseline match "
        "(mirrored openings, fixed nodes or movetime, W/D/L + Elo + LOS)."
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help=f"classical baseline engine (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--nnue",
        default=DEFAULT_ENGINE,
        help=f"NNUE candidate engine (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument("--model", default=None, help=".nnue model for the candidate")
    parser.add_argument("--pairs", type=int, default=10, help="mirrored opening pairs")
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="total games (rounded up to complete mirrored pairs; alternative to --pairs)",
    )
    tc = parser.add_mutually_exclusive_group()
    tc.add_argument("--nodes", type=int, default=None, help="fixed nodes per move")
    tc.add_argument("--movetime", type=int, default=None, help="fixed ms per move")
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=60.0,
        help="hard wall-clock timeout per move in nodes mode (seconds)",
    )
    parser.add_argument("--timeout-extra", type=float, default=5.0)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--opening", action="append", default=[], metavar="UCI_MOVES",
                        help="override the built-in opening list (repeatable)")
    parser.add_argument("--opening-offset", type=int, default=0)
    parser.add_argument("--pgn", default=None, help="write PGN to this path")
    parser.add_argument("--json", default=None, help="write the result as JSON")
    parser.add_argument("--option-a", action="append", default=[], metavar="NAME=VALUE",
                        help="extra candidate UCI option")
    parser.add_argument("--option-b", action="append", default=[], metavar="NAME=VALUE",
                        help="extra baseline UCI option")
    parser.add_argument("--strict", action="store_true",
                        help="exit 3 if any engine error/timeout occurred")
    parser.add_argument("--sprt", action="store_true",
                        help="stop early when the approximate SPRT bounds are crossed")
    parser.add_argument("--sprt-elo0", type=float, default=0.0)
    parser.add_argument("--sprt-elo1", type=float, default=5.0)
    parser.add_argument("--sprt-alpha", type=float, default=0.05)
    parser.add_argument("--sprt-beta", type=float, default=0.05)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    return parser


def _parse_extra_options(items: Iterable[str]) -> dict[str, str]:
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

    if args.games is not None and args.pairs != 10:
        parser.error("use either --pairs or --games, not both")
    if args.games is not None:
        if args.games < 1:
            parser.error("--games must be >= 1")
        pairs = (args.games + 1) // 2
    else:
        if args.pairs < 1:
            parser.error("--pairs must be >= 1")
        pairs = args.pairs

    if args.nodes is None and args.movetime is None:
        args.movetime = 100
    if args.nodes is not None and args.nodes < 1:
        parser.error("--nodes must be >= 1")
    if args.movetime is not None and args.movetime < 1:
        parser.error("--movetime must be >= 1 ms")
    if args.sprt and args.sprt_elo1 <= args.sprt_elo0:
        parser.error("--sprt-elo1 must be greater than --sprt-elo0")

    try:
        baseline = discover_engine(args.baseline)
        candidate = discover_engine(args.nnue)
        if args.model:
            model = os.path.abspath(os.path.expanduser(args.model))
            if not os.path.isfile(model):
                raise SetupError(f"model file not found: {model}")
        else:
            model = None
        extra_a = _parse_extra_options(args.option_a)
        extra_b = _parse_extra_options(args.option_b)
        candidate_sha = sha256_file(candidate)
        baseline_sha = sha256_file(baseline)
    except (SetupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    openings = arena.validate_openings(args.opening or arena.DEFAULT_OPENINGS)

    # Extra options are applied first so load-time options such as
    # ``EvalQuant=int16`` take effect before EvalFile.  NNUE-critical options
    # are then set defaults-wise; an explicit ``--option-a UseNNUE=false``
    # still wins because setdefault is used for UseNNUE.
    candidate_options = dict(extra_a)
    candidate_options.setdefault("EvalBackend", "cpu")
    if model:
        candidate_options["EvalFile"] = model
    candidate_options.setdefault("UseNNUE", "true")
    baseline_options = dict(extra_b)
    baseline_options.setdefault("UseNNUE", "false")

    # Reuse the preflight code for a clear early failure before any game.
    try:
        candidate_preflight = preflight_engine(
            candidate,
            candidate_options,
            model=model,
            startup_timeout=args.startup_timeout,
        )
        baseline_preflight = preflight_engine(
            baseline,
            baseline_options,
            model=None,
            startup_timeout=args.startup_timeout,
        )
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = MatchConfig(
        baseline=baseline,
        candidate=candidate,
        model=model,
        pairs=pairs,
        nodes=args.nodes,
        movetime=args.movetime,
        move_timeout=args.move_timeout,
        timeout_extra=args.timeout_extra,
        max_plies=args.max_plies,
        opening_offset=args.opening_offset,
        openings=openings,
        candidate_options=candidate_options,
        baseline_options=baseline_options,
        candidate_sha256=candidate_sha,
        baseline_sha256=baseline_sha,
        pgn=args.pgn,
        strict=args.strict,
        sprt=args.sprt,
        sprt_elo0=args.sprt_elo0,
        sprt_elo1=args.sprt_elo1,
        sprt_alpha=args.sprt_alpha,
        sprt_beta=args.sprt_beta,
        startup_timeout=args.startup_timeout,
    )

    if args.pgn:
        pgn_dir = os.path.dirname(os.path.abspath(args.pgn))
        if pgn_dir:
            os.makedirs(pgn_dir, exist_ok=True)

    summary = run_match(
        config,
        candidate_preflight=candidate_preflight,
        baseline_preflight=baseline_preflight,
    )

    changed = []
    if sha256_file(candidate) != candidate_sha:
        changed.append("candidate")
    if sha256_file(baseline) != baseline_sha:
        changed.append("baseline")
    if changed:
        summary["binary_changed"] = changed
        summary["ok"] = False
        print(
            "error: binary changed during the match: " + ", ".join(changed),
            file=sys.stderr,
        )

    if args.json:
        path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"json: {path}")
    if args.pgn:
        print(f"pgn: {os.path.abspath(args.pgn)}")

    if summary.get("binary_changed"):
        return 2
    if summary["fatal"] is not None:
        return 2
    if args.strict and summary["engine_errors"]:
        return 3
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
