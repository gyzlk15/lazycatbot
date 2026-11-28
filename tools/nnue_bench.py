#!/usr/bin/env python3
"""Measure engine bench NPS with NNUE off/on (and with alternative models).

Each measurement starts a fresh engine process, applies the configuration,
waits for ``readyok`` and runs the engine's own ``bench N`` command.  Runs are
interleaved (configuration order is rotated inside every repetition) and the
median NPS over ``--repeat`` runs is reported, together with the percentage
loss relative to the classical configuration.

Typical NNUE acceptance run (plan section 5, criterion 3)::

    .venv\\Scripts\\python.exe tools/nnue_bench.py ^
        --engine build/chessbot.exe ^
        --model training/work/model/halfkp_v1.nnue ^
        --depth 10 --repeat 3 --json temp/nnue_bench.json --max-loss 10

Multiple engines / models / free-form profiles are supported::

    tools/nnue_bench.py --engine build/chessbot.exe --engine build-baseline/chessbot_baseline.exe ^
        --model model_a.nnue --model model_b.nnue
    tools/nnue_bench.py --profile "nnue256:UseNNUE=true;EvalFile=models/h256.nnue"

Exit codes: 0 = all runs succeeded (and ``--max-loss`` passed, if given),
1 = a run failed or the loss threshold was exceeded, 2 = setup error.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
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
)


# The six positions used by the engine's own ``bench`` command
# (src/uci.cpp).  Reused for the fixed-node mode so both modes cover the same
# spread of material and tactics.
BENCH_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p3/2B1P3/P1NP1N2/1PP1QPPP/R1B1K2R w KQ - 1 10",
]


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    label: str
    kind: str  # classical | nnue | profile
    options: dict[str, str] = field(default_factory=dict)
    model: Optional[str] = None


def parse_option_pairs(items: Iterable[str], separator: str = ";") -> dict[str, str]:
    options: dict[str, str] = {}
    for item in items:
        item = item.strip()
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"bad option {item!r}; expected NAME=VALUE")
        options[name.strip()] = value.strip()
    return options


def parse_profile(text: str) -> BenchConfig:
    """Parse ``label:Name=Value;Name2=Value2``."""
    label, sep, rest = text.partition(":")
    if not sep or not label.strip() or not rest.strip():
        raise ValueError(
            f"bad --profile {text!r}; expected label:Name=Value;Name2=Value2"
        )
    parts = [part for part in rest.replace(",", ";").split(";") if part.strip()]
    options = parse_option_pairs(parts)
    return BenchConfig(
        label=label.strip(),
        kind="profile",
        options=options,
        model=options.get("EvalFile"),
    )


def build_configs(
    models: Iterable[str],
    profiles: Iterable[str],
) -> list[BenchConfig]:
    configs: list[BenchConfig] = [
        BenchConfig(label="classical", kind="classical", options={"UseNNUE": "false"})
    ]
    for model in models:
        path = os.path.abspath(os.path.expanduser(model))
        configs.append(
            BenchConfig(
                label=f"nnue:{os.path.basename(path)}",
                kind="nnue",
                options={
                    "UseNNUE": "true",
                    "EvalFile": path,
                    "EvalBackend": "cpu",
                },
                model=path,
            )
        )
    for profile in profiles:
        configs.append(parse_profile(profile))

    # Duplicate labels make the report ambiguous.
    seen: dict[str, int] = {}
    for config in configs:
        seen[config.label] = seen.get(config.label, 0) + 1
    duplicates = [label for label, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate configuration label(s): {', '.join(duplicates)}")
    return configs


# ---------------------------------------------------------------------------
# Bench execution
# ---------------------------------------------------------------------------


@dataclass
class BenchRun:
    config: str
    repeat: int
    nodes: Optional[int]
    time_ms: Optional[int]
    nps: Optional[int]
    wall_s: float
    diagnostics: list[str] = field(default_factory=list)
    error: Optional[str] = None
    engine_id: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "repeat": self.repeat,
            "nodes": self.nodes,
            "time_ms": self.time_ms,
            "nps": self.nps,
            "wall_s": round(self.wall_s, 3),
            "diagnostics": list(self.diagnostics),
            "error": self.error,
        }


def apply_options(session: UciSession, options: dict[str, str]) -> list[str]:
    for name, value in options.items():
        if not session.has_option(name):
            if name == "EvalBackend":
                # Optional hint; older builds may not advertise it yet.
                continue
            raise SetupError(f"engine does not advertise option {name!r}")
        session.set_option(name, value)
    ready_lines = session.isready()
    diagnostics = diagnostic_lines(ready_lines)
    failures = [line for line in diagnostics if diagnostic_is_failure(line)]
    if failures:
        raise SetupError("engine reported a load failure: " + "; ".join(failures))
    return diagnostics


def run_bench_once(
    engine: str,
    config: BenchConfig,
    *,
    depth: int,
    base_options: dict[str, str],
    repeat: int,
    timeout: float,
    startup_timeout: float,
    trace: bool,
    fixed_nodes: Optional[int] = None,
) -> BenchRun:
    session: Optional[UciSession] = None
    try:
        session = UciSession(
            engine, startup_timeout=startup_timeout, trace=trace
        )
        merged = dict(base_options)
        merged.update(config.options)
        diagnostics = apply_options(session, merged)
        if fixed_nodes is not None:
            start = time.perf_counter()
            total_nodes = 0
            for fen in BENCH_FENS:
                result = session.search_nodes(fen, fixed_nodes, timeout=timeout)
                total_nodes += int(result["nodes"] or 0)
            wall = time.perf_counter() - start
            time_ms = int(round(wall * 1000.0))
            nps = (total_nodes * 1000) // time_ms if time_ms > 0 else None
            return BenchRun(
                config=config.label,
                repeat=repeat,
                nodes=total_nodes,
                time_ms=time_ms,
                nps=nps,
                wall_s=wall,
                diagnostics=diagnostics,
                engine_id=session.id_name,
            )
        result, wall = session.bench(depth, timeout=timeout)
        return BenchRun(
            config=config.label,
            repeat=repeat,
            nodes=result["nodes"],
            time_ms=result["time_ms"],
            nps=result["nps"],
            wall_s=wall,
            diagnostics=diagnostics,
            engine_id=session.id_name,
        )
    except (SetupError, UciError) as exc:
        return BenchRun(
            config=config.label,
            repeat=repeat,
            nodes=None,
            time_ms=None,
            nps=None,
            wall_s=0.0,
            error=str(exc),
        )
    finally:
        if session is not None:
            session.close()


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return statistics.median(present)


def run_benchmarks(
    engines: list[str],
    configs: list[BenchConfig],
    *,
    depth: int = 10,
    repeat: int = 3,
    hash_mb: int = 64,
    threads: int = 1,
    extra_options: Optional[dict[str, str]] = None,
    timeout: float = 600.0,
    startup_timeout: float = 30.0,
    trace: bool = False,
    fixed_nodes: Optional[int] = None,
) -> dict:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    if fixed_nodes is None and depth < 1:
        raise ValueError("depth must be >= 1")
    if fixed_nodes is not None and fixed_nodes < 1:
        raise ValueError("fixed_nodes must be >= 1")

    base_options: dict[str, str] = {"Hash": str(hash_mb), "Threads": str(threads)}
    if extra_options:
        base_options.update(extra_options)

    report: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "fixed-nodes" if fixed_nodes is not None else "depth",
        "depth": depth,
        "fixed_nodes": fixed_nodes,
        "repeat": repeat,
        "hash_mb": hash_mb,
        "threads": threads,
        "extra_options": dict(extra_options or {}),
        "engines": [],
        "ok": True,
    }

    for engine in engines:
        entry: dict = {
            "path": engine,
            "id": None,
            "configs": [],
            "classical_median_nps": None,
        }
        # Interleave: one round of every configuration per repetition, so
        # thermal drift and background load affect all configurations equally.
        runs_by_label: dict[str, list[BenchRun]] = {c.label: [] for c in configs}
        engine_id: Optional[str] = None
        for rep in range(1, repeat + 1):
            for config in configs:
                print(
                    f"  {os.path.basename(engine)} {config.label:28s} "
                    f"rep {rep}/{repeat}: running ...",
                    flush=True,
                )
                run = run_bench_once(
                    engine,
                    config,
                    depth=depth,
                    base_options=base_options,
                    repeat=rep,
                    timeout=timeout,
                    startup_timeout=startup_timeout,
                    trace=trace,
                    fixed_nodes=fixed_nodes,
                )
                runs_by_label[config.label].append(run)
                if run.error:
                    report["ok"] = False
                if engine_id is None and run.engine_id:
                    engine_id = run.engine_id
                print(
                    f"  {os.path.basename(engine)} {config.label:28s} "
                    f"rep {rep}/{repeat}: "
                    + (
                        f"{run.nps if run.nps is not None else '?':>9} nps "
                        f"({run.nodes} nodes, {run.time_ms} ms)"
                        if run.error is None
                        else f"ERROR: {run.error}"
                    ),
                    flush=True,
                )
        entry["id"] = engine_id

        for config in configs:
            runs = runs_by_label[config.label]
            errors = [run.error for run in runs if run.error]
            median_nps = _median(run.nps for run in runs)
            median_nodes = _median(run.nodes for run in runs)
            median_time = _median(run.time_ms for run in runs)
            entry["configs"].append(
                {
                    "label": config.label,
                    "kind": config.kind,
                    "options": dict(config.options),
                    "model": config.model,
                    "runs": [run.as_dict() for run in runs],
                    "median_nps": median_nps,
                    "median_nodes": median_nodes,
                    "median_time_ms": median_time,
                    "loss_pct": None,
                    "errors": errors,
                }
            )

        classical = next(
            (item for item in entry["configs"] if item["kind"] == "classical"),
            None,
        )
        if classical is not None and classical["median_nps"]:
            baseline_nps = classical["median_nps"]
            entry["classical_median_nps"] = baseline_nps
            for item in entry["configs"]:
                if item is classical or not item["median_nps"]:
                    continue
                item["loss_pct"] = (
                    (baseline_nps - item["median_nps"]) / baseline_nps * 100.0
                )
        report["engines"].append(entry)

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(report: dict, max_loss: Optional[float]) -> None:
    print("\n=== NNUE bench ===")
    if report.get("mode") == "fixed-nodes":
        workload = f"fixed {report.get('fixed_nodes')} nodes/move x {len(BENCH_FENS)} FENs"
    else:
        workload = f"depth {report['depth']}"
    print(
        f"{workload}  repeat {report['repeat']}  "
        f"Hash {report['hash_mb']} MB  Threads {report['threads']}"
    )
    over_threshold: list[str] = []
    for engine in report["engines"]:
        print(f"\nengine: {engine['path']}")
        if engine.get("id"):
            print(f"id:     {engine['id']}")
        print(
            f"{'configuration':30s} {'median NPS':>12s} {'nodes':>10s} "
            f"{'ms':>8s} {'loss %':>8s}  status"
        )
        print("-" * 92)
        for item in engine["configs"]:
            loss_text = (
                f"{item['loss_pct']:8.2f}"
                if item["loss_pct"] is not None
                else f"{'-':>8s}"
            )
            status = "ok" if not item["errors"] else "FAIL"
            nps = item["median_nps"]
            nps_text = f"{int(round(nps)):>12,d}" if nps is not None else f"{'-':>12s}"
            nodes = item["median_nodes"]
            nodes_text = (
                f"{int(round(nodes)):>10,d}" if nodes is not None else f"{'-':>10s}"
            )
            ms = item["median_time_ms"]
            ms_text = f"{int(round(ms)):>8,d}" if ms is not None else f"{'-':>8s}"
            print(
                f"{item['label'][:30]:30s} {nps_text} {nodes_text} {ms_text} "
                f"{loss_text}  {status}"
            )
            if item["errors"]:
                for error in item["errors"]:
                    print(f"    error: {error}")
            if (
                max_loss is not None
                and item["kind"] != "classical"
                and item["loss_pct"] is not None
                and item["loss_pct"] > max_loss
            ):
                over_threshold.append(
                    f"{os.path.basename(engine['path'])}/{item['label']}: "
                    f"{item['loss_pct']:.2f}% > {max_loss:g}%"
                )
        if engine["classical_median_nps"] is None:
            print("  note: no classical baseline; percentage loss not computed")
    print()
    if max_loss is not None:
        if over_threshold:
            print(f"THRESHOLD FAILED (max loss {max_loss:g}%):")
            for line in over_threshold:
                print(f"  - {line}")
        else:
            print(f"THRESHOLD PASSED (max loss {max_loss:g}%)")
    print("RESULT: PASS" if report.get("ok") and not over_threshold else "RESULT: FAIL")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure engine bench NPS with NNUE off/on, repeated and "
        "interleaved, and report median NPS loss."
    )
    parser.add_argument(
        "--engine",
        action="append",
        default=None,
        help=f"engine binary (repeatable; default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=".nnue model to benchmark (repeatable); adds an NNUE-on configuration",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        metavar="LABEL:NAME=VALUE;...",
        help="free-form configuration (repeatable)",
    )
    parser.add_argument("--depth", type=int, default=10, help="bench depth (default 10)")
    parser.add_argument(
        "--fixed-nodes",
        type=int,
        default=None,
        help="measure throughput at equal nodes/move instead of the "
        "engine's depth-N bench (ignores --depth)",
    )
    parser.add_argument("--repeat", type=int, default=3, help="runs per configuration")
    parser.add_argument("--hash", type=int, default=64, help="Hash size in MB")
    parser.add_argument("--threads", type=int, default=1, help="engine threads")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra UCI option applied to every configuration (repeatable)",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        default=None,
        help="exit 1 when an NNUE configuration loses more than this %% NPS",
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per bench")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--json", default=None, help="write the full report as JSON")
    parser.add_argument("--trace", action="store_true", help="dump raw UCI traffic")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    engine_paths: list[str] = []
    try:
        for explicit in args.engine or [DEFAULT_ENGINE]:
            engine_paths.append(discover_engine(explicit))
        for model in args.model:
            if not os.path.isfile(model):
                raise SetupError(f"model file not found: {model}")
        extra_options = parse_option_pairs(args.option)
        configs = build_configs(args.model, args.profile)
        for config in configs:
            if config.model and config.model not in ("", "<empty>"):
                if not os.path.isfile(os.path.abspath(config.model)):
                    raise SetupError(
                        f"model file not found for {config.label}: {config.model}"
                    )
    except (SetupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.model and not args.profile:
        print(
            "warning: no --model/--profile given; only the classical "
            "configuration will be measured",
            file=sys.stderr,
        )

    try:
        report = run_benchmarks(
            engine_paths,
            configs,
            depth=args.depth,
            repeat=args.repeat,
            hash_mb=args.hash,
            threads=args.threads,
            extra_options=extra_options,
            timeout=args.timeout,
            startup_timeout=args.startup_timeout,
            trace=args.trace,
            fixed_nodes=args.fixed_nodes,
        )
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_report(report, args.max_loss)

    if args.json:
        path = os.path.abspath(args.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"json: {path}")

    failed = not report.get("ok")
    if args.max_loss is not None:
        for engine in report["engines"]:
            for item in engine["configs"]:
                if (
                    item["kind"] != "classical"
                    and item["loss_pct"] is not None
                    and item["loss_pct"] > args.max_loss
                ):
                    failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
