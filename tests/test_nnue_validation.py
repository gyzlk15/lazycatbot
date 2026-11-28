"""Smoke tests for the NNUE validation harness.

These tests are local-only.  Engine-dependent tests skip cleanly when no
binary is present.  A tiny scripted UCI engine (written to ``temp/``) covers
the protocol/parse paths without needing a real model or build.
"""
from __future__ import annotations

import os
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import nnue_bench  # noqa: E402
import nnue_sprt  # noqa: E402
import nnue_validate  # noqa: E402
from nnue_validate import (  # noqa: E402
    BUILTIN_CASES,
    Expectation,
    FenCase,
    ReferenceEvaluator,
    engine_net_cp,
    is_direct_eval,
    is_eval_line,
    parse_bench_line,
    parse_eval_line,
    parse_fen_cases,
    parse_info_string,
    search_eval_cp,
)

BASELINE = os.path.join(ROOT, "build-baseline", "chessbot_baseline.exe")
CANDIDATE = os.path.join(ROOT, "build", "chessbot.exe")
REFERENCE = os.path.join(ROOT, "training", "reference_inference.py")

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch():
    path = os.path.join(ROOT, "temp", "nnue_validation_tests")
    os.makedirs(path, exist_ok=True)
    return path


FAKE_ENGINE = r'''#!/usr/bin/env python3
"""Scripted UCI engine for tools/nnue_validate.py tests (local only)."""
import os
import sys

VALUES = {"P": 100, "N": 300, "B": 300, "R": 500, "Q": 900, "K": 0}


def material(fen):
    placement, turn = fen.split()[0], fen.split()[1]
    score = 0
    for ch in placement:
        if ch in VALUES:
            score += VALUES[ch]
        elif ch.upper() in VALUES and ch.islower():
            score -= VALUES[ch.upper()]
    return -score if turn == "b" else score


def main():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    use_nnue = False
    evalfile = ""
    available = 1
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        head = line.split()[0]
        if head == "uci":
            print("id name FakeNNUEEngine", flush=True)
            print("option name Hash type spin default 64 min 1 max 4096", flush=True)
            print("option name Threads type spin default 1 min 1 max 16", flush=True)
            print("option name UseNNUE type check default false", flush=True)
            print("option name EvalFile type string default <empty>", flush=True)
            print("option name EvalBackend type string default cpu", flush=True)
            print("option name Clear Hash type button", flush=True)
            print("uciok", flush=True)
        elif head == "isready":
            if use_nnue and evalfile and not os.path.isfile(evalfile):
                available = 0
                print("info string NNUE: load failed: " + evalfile, flush=True)
            else:
                available = 1
            print("readyok", flush=True)
        elif head == "ucinewgame":
            pass
        elif head == "setoption":
            rest = line[len("setoption name "):] if line.startswith("setoption name ") else ""
            parts = rest.split(" value ", 1)
            name = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if name == "UseNNUE":
                use_nnue = value in ("true", "1")
            elif name == "EvalFile":
                evalfile = "" if value in ("<empty>", "") else value
        elif head == "position":
            tokens = line.split()
            if len(tokens) >= 3 and tokens[1] == "fen":
                try:
                    idx = tokens.index("moves", 2)
                except ValueError:
                    idx = len(tokens)
                fen = " ".join(tokens[2:idx])
            else:
                fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        elif head == "eval":
            classical = material(fen)
            net = material(fen)
            enabled = 1 if (use_nnue and available) else 0
            total = net if enabled else classical
            print(
                "info string classical(stm) %d neural %d total %d "
                "nnue_available %d enabled %d weight 25 clip 80 direct %d net %d"
                % (classical, net if enabled else 0, total, available, enabled,
                   1 if enabled else 0, net),
                flush=True,
            )
        elif head == "go":
            # Fixed-node bench mode: the exact move is irrelevant.
            print("info depth 1 nodes 1000 nps 1000000", flush=True)
            print("bestmove 0000", flush=True)
        elif head == "bench":
            depth = int(line.split()[1]) if len(line.split()) > 1 else 10
            nodes = 9000 if use_nnue else 10000
            print(
                "bench nodes %d time 100 ms nps %d" % (nodes, nodes * 1000 // 100),
                flush=True,
            )
        elif head == "quit":
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_fake_engine(scratch: str) -> str:
    path = os.path.join(scratch, "fake_uci_engine.py")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(FAKE_ENGINE)
    return path


def _fnv1a_64(data: bytes) -> int:
    h = 14695981039346656037
    for byte in data:
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def write_small_nnue_model(path: str, out_bias: float = 42.0, hidden: int = 32) -> str:
    """Write a valid .nnue v1 file whose dense output is ``out_bias`` cp."""
    inputs = 49152
    ft = np.zeros((inputs, hidden), dtype="<i2")
    ft_bias = np.zeros(hidden, dtype="<f4")
    fc1_w = np.zeros((hidden, 32), dtype="<f4")
    fc1_b = np.zeros(32, dtype="<f4")
    fc2_w = np.zeros((32, 32), dtype="<f4")
    fc2_b = np.zeros(32, dtype="<f4")
    out_w = np.zeros(32, dtype="<f4")
    out_b = np.asarray([out_bias], dtype="<f4")
    payload = b"".join(
        array.tobytes()
        for array in (ft, ft_bias, fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b)
    )
    header = struct.pack(
        "<4s7IfIQ16s",
        b"LCNN",
        1,
        1,
        inputs,
        hidden,
        64,
        1,
        0,
        1.0,
        0,
        _fnv1a_64(payload),
        b"\x00" * 16,
    )
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(payload)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_info_string_correction_mode():
    line = (
        "info string classical(stm) 12 neural -3 total 9 nnue_available 1 "
        "enabled 1 weight 25 clip 80 direct 0"
    )
    fields = parse_info_string(line)
    assert fields == {
        "classical": 12.0,
        "neural": -3.0,
        "total": 9.0,
        "available": 1.0,
        "enabled": 1.0,
        "weight": 25.0,
        "clip": 80.0,
        "direct": 0.0,
    }


def test_parse_info_string_direct_mode_and_aliases():
    line = (
        "info string classical(stm) -20 nnue_cp 35 total 35 nnue_available 1 "
        "enabled 1 direct 1"
    )
    fields = parse_eval_line(line)
    assert fields["classical"] == -20.0
    assert fields["net"] == 35.0
    assert fields["total"] == 35.0
    assert is_direct_eval(fields)
    assert engine_net_cp(fields) == 35


def test_parse_info_string_ignores_non_numeric_values():
    fields = parse_info_string("info string NNUE: loaded 1 file C:\\models\\x.nnue")
    # ``loaded``/``file`` are not canonical fields; no false numeric keys.
    assert fields == {}


def test_engine_net_cp_choices():
    assert engine_net_cp({"enabled": 0, "neural": 0, "total": 10}) is None
    assert engine_net_cp({"enabled": 1, "neural": 7, "total": 12, "direct": 0}) == 7
    assert engine_net_cp({"enabled": 1, "total": 12, "direct": 1}) == 12
    assert engine_net_cp({"net": 99, "total": 12, "direct": 1}) == 99
    assert engine_net_cp({"neural": 7, "total": 12}, "total") == 12
    assert engine_net_cp({"neural": 7, "total": 12}, "neural") == 7
    assert engine_net_cp({"neural": 7}, "net") is None
    with pytest.raises(ValueError):
        engine_net_cp({"net": 1}, "bogus")
    assert search_eval_cp({"total": 12, "net": 99}) == 12


def test_is_eval_line():
    assert is_eval_line("info string classical(stm) 1 neural 0 total 1")
    assert not is_eval_line("info string NNUE: loaded model.nnue")
    assert not is_eval_line("readyok")


def test_parse_bench_line():
    assert parse_bench_line("bench nodes 849356 time 407 ms nps 2086869") == {
        "nodes": 849356,
        "time_ms": 407,
        "nps": 2086869,
    }
    assert parse_bench_line("bench nodes 1000 time 5 ms") == {
        "nodes": 1000,
        "time_ms": 5,
        "nps": 200000,
    }
    assert parse_bench_line("bench nodes 1000 time 0 ms")["nps"] is None
    with pytest.raises(ValueError):
        parse_bench_line("not a bench line")


# ---------------------------------------------------------------------------
# FEN cases / expectations
# ---------------------------------------------------------------------------


def test_builtin_cases_are_valid_and_named_uniquely():
    import chess

    names = [case.name for case in BUILTIN_CASES]
    assert len(names) == len(set(names))
    for case in BUILTIN_CASES:
        board = chess.Board(case.fen)
        assert board.is_valid(), case.name
        if case.expectation.kind in ("positive", "negative"):
            assert case.expectation.threshold >= 0
        if case.expectation.kind == "zero":
            assert 0 <= case.expectation.threshold < 1000


def test_parse_fen_cases():
    text = """
    # comment
    start | rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    4k3/8/8/8/8/8/8/4K3 w - - 0 1
    """
    cases = parse_fen_cases(text, source="unit")
    assert len(cases) == 2
    assert cases[0].name == "start"
    assert cases[1].name == "unit:4"
    labelled = parse_fen_cases(
        "tac | 4k3/8/8/8/8/8/8/3QK3 w - - 0 1 | -420", source="lab"
    )
    assert labelled[0].label_cp == -420
    assert labelled[0].expectation.kind == "negative"
    with pytest.raises(ValueError):
        parse_fen_cases("not a fen at all", source="bad")


def test_fen_case_rejects_illegal_position():
    with pytest.raises(ValueError):
        FenCase("bad", "8/8/8/8/8/8/8/8 w - - 0 1")


def test_expectation_checks():
    assert Expectation("positive", 100).check(101)
    assert not Expectation("positive", 100).check(100)
    assert Expectation("negative", 100).check(-101)
    assert Expectation("zero", 30).check(-30)
    assert not Expectation("zero", 30).check(31)
    assert Expectation("none").check(None)


# ---------------------------------------------------------------------------
# Reference adapter
# ---------------------------------------------------------------------------


def test_reference_adapter_plain_function(scratch):
    path = os.path.join(scratch, "ref_plain.py")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("def evaluate_fen(fen):\n    return 42\n")
    evaluator = ReferenceEvaluator(path, None)
    assert evaluator.evaluate(START_FEN) == 42.0


def test_reference_adapter_dict_result(scratch):
    path = os.path.join(scratch, "ref_dict.py")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("def eval_fen(fen):\n    return {'cp': 7.5}\n")
    evaluator = ReferenceEvaluator(path, None)
    assert evaluator.evaluate(START_FEN) == 7.5


@pytest.mark.skipif(
    not os.path.isfile(REFERENCE), reason="training/reference_inference.py missing"
)
def test_reference_adapter_agent_c_model(scratch):
    model = write_small_nnue_model(os.path.join(scratch, "tiny.nnue"), out_bias=42.0)
    evaluator = ReferenceEvaluator(REFERENCE, model)
    assert "load_nnue" in (evaluator.describe() or "")
    assert evaluator.evaluate(START_FEN) == pytest.approx(42.0, abs=1e-3)


# ---------------------------------------------------------------------------
# End-to-end with the scripted engine
# ---------------------------------------------------------------------------


def test_fake_engine_validation(scratch):
    engine = write_fake_engine(scratch)
    model = os.path.join(scratch, "fake.nnue")
    with open(model, "wb") as handle:
        handle.write(b"fake")
    report = nnue_validate.run_validation(
        engine,
        model=model,
        cases=[],
        reference_path=None,
        tolerance=2.0,
        use_nnue=True,
        eval_timeout=5.0,
        startup_timeout=10.0,
    )
    assert report["summary"]["failed"] == 0
    assert report["summary"]["cases"] == len(BUILTIN_CASES)
    by_name = {item["name"]: item for item in report["cases"]}
    assert by_name["kqk-white-to-move"]["search_cp"] > 0
    assert by_name["kqk-white-to-move-weak"]["search_cp"] < 0
    assert by_name["startpos"]["search_cp"] == 0
    assert by_name["kqk-white-to-move"]["fields"]["enabled"] == 1


def test_fake_engine_missing_model_is_reported(scratch):
    engine = write_fake_engine(scratch)
    with pytest.raises(nnue_validate.SetupError):
        nnue_validate.run_validation(
            engine,
            model=os.path.join(scratch, "does-not-exist.nnue"),
            use_nnue=True,
            startup_timeout=10.0,
        )


def test_fake_engine_bench(scratch):
    engine = write_fake_engine(scratch)
    model = os.path.join(scratch, "fake.nnue")
    with open(model, "wb") as handle:
        handle.write(b"fake")
    configs = nnue_bench.build_configs([model], [])
    report = nnue_bench.run_benchmarks(
        [engine],
        configs,
        depth=1,
        repeat=1,
        timeout=30.0,
        startup_timeout=10.0,
    )
    assert report["ok"]
    entry = report["engines"][0]
    classical = next(c for c in entry["configs"] if c["kind"] == "classical")
    nnue = next(c for c in entry["configs"] if c["kind"] == "nnue")
    assert classical["median_nps"] == 100000
    assert nnue["median_nps"] == 90000
    assert nnue["loss_pct"] == pytest.approx(10.0)


def test_fake_engine_fixed_node_bench(scratch):
    engine = write_fake_engine(scratch)
    model = os.path.join(scratch, "fake.nnue")
    with open(model, "wb") as handle:
        handle.write(b"fake")
    configs = nnue_bench.build_configs([model], [])
    report = nnue_bench.run_benchmarks(
        [engine],
        configs,
        fixed_nodes=1000,
        repeat=1,
        timeout=30.0,
        startup_timeout=10.0,
    )
    assert report["ok"]
    assert report["mode"] == "fixed-nodes"
    for item in report["engines"][0]["configs"]:
        assert item["median_nodes"] == len(nnue_bench.BENCH_FENS) * 1000


@pytest.mark.skipif(not os.path.isfile(CANDIDATE), reason="candidate build missing")
@pytest.mark.skipif(
    not os.path.isfile(REFERENCE), reason="training/reference_inference.py missing"
)
def test_candidate_reference_scale_smoke(scratch):
    """A constant-output model must match the reference (catches 400x scale bugs)."""
    model = write_small_nnue_model(os.path.join(scratch, "scale42.nnue"), out_bias=42.0)
    report = nnue_validate.run_validation(
        CANDIDATE,
        model=model,
        cases=[],
        reference_path=REFERENCE,
        tolerance=2.0,
        use_nnue=True,
        extra_options={"EvalQuant": "int16"},
        eval_timeout=10.0,
        startup_timeout=30.0,
    )
    summary = report["summary"]
    assert summary["reference_compared"] == len(BUILTIN_CASES)
    assert summary["reference_failed"] == 0, summary
    assert summary["max_abs_reference_delta"] is not None
    assert summary["max_abs_reference_delta"] <= 2.0


# ---------------------------------------------------------------------------
# Statistics / SPRT
# ---------------------------------------------------------------------------


def test_los_from_scores():
    assert nnue_sprt.los_from_scores([]) is None
    assert nnue_sprt.los_from_scores([1.0]) == 1.0
    assert nnue_sprt.los_from_scores([0.0]) == 0.0
    assert nnue_sprt.los_from_scores([1.0, 0.0]) == pytest.approx(0.5)
    assert nnue_sprt.los_from_scores([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert nnue_sprt.los_from_scores([0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert nnue_sprt.los_from_scores([1.0, 1.0, 0.5]) > 0.5
    assert nnue_sprt.los_from_scores([0.0, 0.0, 0.5]) < 0.5


def test_pair_scores_uses_complete_pairs_only():
    records = [
        SimpleNamespace(pair_index=0, a_score=1.0),
        SimpleNamespace(pair_index=0, a_score=0.5),
        SimpleNamespace(pair_index=1, a_score=0.0),
    ]
    assert nnue_sprt.pair_scores(records) == [0.75]


def test_sprt_llr_and_decision():
    wins = nnue_sprt.sprt_llr([1.0] * 400, elo0=0.0, elo1=5.0)
    decision, lower, upper = nnue_sprt.sprt_decision(
        wins["llr"], alpha=0.05, beta=0.05
    )
    assert decision == "accept_h1"
    assert wins["llr"] >= upper
    losses = nnue_sprt.sprt_llr([0.0] * 400, elo0=0.0, elo1=5.0)
    decision, lower, upper = nnue_sprt.sprt_decision(
        losses["llr"], alpha=0.05, beta=0.05
    )
    assert decision == "accept_h0"
    assert losses["llr"] <= lower
    with pytest.raises(ValueError):
        nnue_sprt.sprt_llr([1.0], elo0=5.0, elo1=5.0)


def test_normal_cdf():
    assert nnue_sprt.normal_cdf(0.0) == pytest.approx(0.5)
    assert nnue_sprt.normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


def test_node_control_command():
    control = nnue_sprt.NodeControl(nodes=1234, move_timeout=7.0)
    assert control.go_command({}) == "go nodes 1234"
    assert control.hard_timeout(None, None) == 7.0


def test_build_configs_and_profiles():
    configs = nnue_bench.build_configs(
        ["C:/models/a.nnue"], ["extra:UseNNUE=true;EvalFile=C:/models/b.nnue"]
    )
    labels = [config.label for config in configs]
    assert labels[0] == "classical"
    assert labels[1] == "nnue:a.nnue"
    assert labels[2] == "extra"
    with pytest.raises(ValueError):
        nnue_bench.build_configs([], ["extra:UseNNUE=true", "extra:UseNNUE=false"])


# ---------------------------------------------------------------------------
# Real-engine smoke tests (local binaries only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(CANDIDATE), reason="candidate build missing")
def test_candidate_classical_validation_smoke():
    report = nnue_validate.run_validation(
        CANDIDATE,
        model=None,
        cases=[],
        reference_path=None,
        use_nnue=False,
        eval_timeout=10.0,
        startup_timeout=30.0,
    )
    assert report["summary"]["failed"] == 0, report["cases"]


@pytest.mark.skipif(not os.path.isfile(BASELINE), reason="baseline binary missing")
def test_baseline_classical_validation_smoke():
    report = nnue_validate.run_validation(
        BASELINE,
        model=None,
        cases=[],
        reference_path=None,
        use_nnue=False,
        eval_timeout=10.0,
        startup_timeout=30.0,
    )
    assert report["summary"]["cases"] == len(BUILTIN_CASES)
    assert report["summary"]["failed"] == 0, report["cases"]
    for item in report["cases"]:
        assert item["fields"], item["name"]
        assert item["sign_ok"], item["name"]


@pytest.mark.skipif(not os.path.isfile(BASELINE), reason="baseline binary missing")
def test_baseline_bench_smoke():
    configs = [
        nnue_bench.BenchConfig(
            label="classical", kind="classical", options={"UseNNUE": "false"}
        )
    ]
    report = nnue_bench.run_benchmarks(
        [BASELINE],
        configs,
        depth=6,
        repeat=1,
        timeout=120.0,
        startup_timeout=30.0,
    )
    assert report["ok"], report
    item = report["engines"][0]["configs"][0]
    assert item["median_nodes"] and item["median_nodes"] > 0
    assert item["median_nps"] and item["median_nps"] > 0


@pytest.mark.skipif(not os.path.isfile(BASELINE), reason="baseline binary missing")
def test_baseline_match_smoke():
    config = nnue_sprt.MatchConfig(
        baseline=BASELINE,
        candidate=BASELINE,
        model=None,
        pairs=1,
        nodes=300,
        movetime=None,
        move_timeout=10.0,
        timeout_extra=2.0,
        max_plies=80,
        opening_offset=0,
        openings=[""],
        candidate_options={"UseNNUE": "true"},
        baseline_options={"UseNNUE": "false"},
        startup_timeout=30.0,
    )
    summary = nnue_sprt.run_match(config)
    assert summary["fatal"] is None
    assert summary["games"] >= 1
    assert summary["wins"] + summary["draws"] + summary["losses"] == summary["games"]
    assert 0.0 <= summary["score"] <= 1.0
    assert summary["los_game"] is not None
