#!/usr/bin/env python3
"""C++ vs Python validation for the HalfKP v1 loader and inference.

Run from the repository root:
  .venv\\Scripts\\python.exe tests\\run_nnue.py [--engine build\\chessbot.exe]

Checks:
  * EvalFile loads the synthetic fixture; diagnostics report version/hidden.
  * `eval` network output matches the Python reference on 100 FENs (<=2 cp).
  * UseNNUE=false and EvalBackend=none fall back to classical.
  * A corrupt .nnue file is rejected with an `info string` warning and the
    engine keeps running with classical eval.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, HERE)

from uci_engine import Engine  # noqa: E402


def parse_eval(lines: list[str]) -> dict:
    for line in lines:
        if "classical(stm)" not in line:
            continue
        out: dict[str, object] = {}
        toks = line.split()
        for key in ("classical(stm)", "neural", "total", "weight", "clip",
                    "net_weight", "net_clip", "ft_quant", "fc1_quant",
                    "net_material", "material_cp"):
            try:
                out[key] = int(toks[toks.index(key) + 1])
            except (ValueError, IndexError):
                pass
        for key in ("nnue_available", "enabled", "direct"):
            out[key] = f"{key} 1" in line
        for key in ("backend", "simd"):
            try:
                out[key] = toks[toks.index(key) + 1]
            except (ValueError, IndexError):
                pass
        try:
            out["version"] = int(toks[toks.index("version") + 1])
        except (ValueError, IndexError):
            pass
        try:
            out["hidden"] = int(toks[toks.index("hidden") + 1])
        except (ValueError, IndexError):
            pass
        return out
    return {}


def eval_now(eng: Engine) -> dict:
    start = eng.mark()
    eng.send("eval")
    eng.wait_for("info string", timeout=5, start=start)
    return parse_eval(eng.collect(start))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(ROOT, "build", "chessbot.exe"))
    ap.add_argument("--tol", type=int, default=2)
    args = ap.parse_args()

    if not os.path.exists(os.path.join(FIXTURES, "tiny.nnue")):
        print("fixtures missing; run tests/nnue_fixture.py first", file=sys.stderr)
        return 2
    with open(os.path.join(FIXTURES, "tiny_reference.json"), encoding="utf-8") as f:
        ref = json.load(f)
    if not os.path.exists(args.engine):
        print(f"engine not found: {args.engine}", file=sys.stderr)
        return 2

    failures: list[str] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        if not ok:
            failures.append(f"{what}: {detail}".strip())

    eng = Engine(args.engine)
    try:
        eng.uci()
        eng.send("setoption name Threads value 1")

        start = eng.mark()
        eng.send(f"setoption name EvalFile value {os.path.join(FIXTURES, 'tiny.nnue')}")
        eng.wait_ready()
        lines = eng.collect(start)
        loaded = next((ln for ln in lines if "EvalFile: loaded" in ln), "")
        check(bool(loaded), "EvalFile load message", str(lines[-2:]))

        eng.send("setoption name UseNNUE value true")
        eng.send("setoption name NNUEWeight value 100")
        eng.send("setoption name EvalBackend value cpu")
        eng.wait_ready()

        eng.set_position("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        diag = eval_now(eng)
        check(diag.get("backend") == "cpu", "backend diagnostic", str(diag))
        check(diag.get("version") == 1, "version diagnostic", str(diag))
        check(diag.get("hidden") == ref["hidden"], "hidden diagnostic", str(diag))
        check(diag.get("simd") in ("avx2", "scalar"), "simd diagnostic", str(diag))

        worst = 0
        worst_fen = ""
        checked = 0
        for fen, expected in ref["fens"].items():
            eng.set_position(fen)
            diag = eval_now(eng)
            if "neural" not in diag:
                check(False, "missing neural field", fen)
                continue
            got = int(diag["neural"])
            diff = abs(got - int(expected))
            checked += 1
            if diff > worst:
                worst = diff
                worst_fen = fen
            if diff > args.tol:
                check(False, f"inference mismatch ({got} vs {expected}, diff {diff})", fen)
        check(checked == len(ref["fens"]), "checked all reference FENs",
              f"{checked}/{len(ref['fens'])}")
        print(f"inference: {checked} FENs, max |C++ - Python| = {worst} cp"
              + (f" ({worst_fen})" if worst_fen else ""))

        # Direct eval: total should equal neural when weight is 100.
        eng.set_position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        d = eval_now(eng)
        check(d.get("total") == d.get("neural"), "weight 100 total == neural", str(d))

        # UseNNUE false -> classical.
        eng.send("setoption name UseNNUE value false")
        eng.wait_ready()
        d = eval_now(eng)
        check(d.get("neural") == 0 and d.get("total") == d.get("classical(stm)"),
              "UseNNUE false -> classical", str(d))

        # Backend none -> classical even with UseNNUE true.
        eng.send("setoption name UseNNUE value true")
        eng.send("setoption name EvalBackend value none")
        eng.wait_ready()
        d = eval_now(eng)
        check(d.get("backend") == "none" and d.get("total") == d.get("classical(stm)"),
              "EvalBackend none -> classical", str(d))

        # Invalid backend value is ignored safely.
        eng.send("setoption name EvalBackend value bogus")
        eng.wait_ready()
        d = eval_now(eng)
        check(d.get("backend") == "none", "invalid backend keeps previous", str(d))
        eng.send("setoption name EvalBackend value cpu")

        # Corrupt file: warning + classical fallback + engine still alive.
        bad = os.path.join(ROOT, "temp", "nnue_bad_checksum.nnue")
        if not os.path.exists(bad):
            with open(os.path.join(FIXTURES, "tiny.nnue"), "rb") as f:
                data = bytearray(f.read())
            data[-5] ^= 0x5A
            os.makedirs(os.path.dirname(bad), exist_ok=True)
            with open(bad, "wb") as f:
                f.write(data)
        start = eng.mark()
        eng.send(f"setoption name EvalFile value {bad}")
        eng.wait_ready()
        lines = eng.collect(start)
        check(any("EvalFile: failed" in ln for ln in lines),
              "corrupt EvalFile warning", str(lines[-2:]))
        check(eng.alive(), "engine alive after failed load")
        d = eval_now(eng)
        check(d.get("total") == d.get("classical(stm)"),
              "classical fallback after failed load", str(d))

        # Odd hidden size exercises the scalar tail of the accumulator.
        start = eng.mark()
        eng.send(f"setoption name EvalFile value "
                 f"{os.path.join(FIXTURES, 'tiny_odd.nnue')}")
        eng.wait_ready()
        d = eval_now(eng)
        check(d.get("hidden") == 13, "odd fixture hidden 13", str(d))
        check(eng.alive(), "engine alive after odd fixture")

        # If Agent C's NumPy reference and exported models are present, run
        # the contract comparison: C++ `neural` vs Python cp on 100 FENs.
        ref_module = os.path.join(ROOT, "training", "reference_inference.py")
        if os.path.exists(ref_module):
            sys.path.insert(0, os.path.join(ROOT, "training"))
            try:
                from reference_inference import load_nnue
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"reference_inference unavailable: {exc}")
                load_nnue = None
            if load_nnue is not None:
                models = []
                for name in ("c-pilot-smoke", "c-final-cpu2", "c-final-cpu",
                             "c-learn4", "c-final-gpu"):
                    path = os.path.join(ROOT, "training", "work", name,
                                        "model.nnue")
                    if os.path.exists(path):
                        models.append(path)
                for model_path in models:
                    try:
                        model = load_nnue(model_path)
                    except Exception as exc:
                        check(False, f"Python model load {model_path}", str(exc))
                        continue
                    if model.hidden > 512:
                        print(f"skip non-deployable hidden {model.hidden}: "
                              f"{model_path}")
                        continue
                    start = eng.mark()
                    eng.send(f"setoption name EvalFile value {model_path}")
                    eng.wait_ready()
                    loaded = any("EvalFile: loaded" in ln
                                 for ln in eng.collect(start))
                    check(loaded, f"load real model {os.path.basename(model_path)}",
                          "")
                    eng.send("setoption name UseNNUE value true")
                    eng.send("setoption name NNUEWeight value 100")
                    eng.wait_ready()
                    worst = 0.0
                    worst_fen = ""
                    for fen in ref["fens"]:
                        expected = model.eval_fen(fen)["cp"]
                        eng.set_position(fen)
                        diag = eval_now(eng)
                        got = int(diag.get("neural", 0))
                        diff = abs(got - expected)
                        if diff > worst:
                            worst, worst_fen = diff, fen
                        if diff > args.tol:
                            check(False,
                                  f"real inference mismatch hidden "
                                  f"{model.hidden} ({got} vs {expected:.2f})",
                                  fen)
                    check(worst <= args.tol,
                          f"real model hidden {model.hidden} <= {args.tol} cp",
                          f"max {worst:.3f} cp at {worst_fen[:60]}")
                    print(f"real model hidden {model.hidden}: max diff "
                          f"{worst:.3f} cp on {len(ref['fens'])} FENs")

        # EvalQuant round-trip: int16 forces the exact v1 path; auto picks
        # int8 FT/FC1 for the pilot models.
        pilot = os.path.join(ROOT, "training", "work", "c-pilot-smoke",
                             "model.nnue")
        if os.path.exists(pilot):
            eng.send("setoption name EvalQuant value int16")
            eng.send(f"setoption name EvalFile value {pilot}")
            eng.wait_ready()
            d = eval_now(eng)
            check(d.get("ft_quant") == 16 and d.get("fc1_quant") == 32,
                  "EvalQuant int16 forces exact path", str(d))
            eng.send("setoption name EvalQuant value auto")
            eng.send(f"setoption name EvalFile value {pilot}")
            eng.wait_ready()
            d = eval_now(eng)
            check(d.get("ft_quant") == 8 and d.get("fc1_quant") == 8,
                  "EvalQuant auto selects int8 path", str(d))
            eng.send("setoption name EvalQuant value bogus")
            eng.wait_ready()
            check(eng.alive(), "invalid EvalQuant ignored safely")

        # NNUEMaterial: diagnostic material correction on a skewed position.
        eng.send("setoption name UseNNUE value true")
        eng.send("setoption name NNUEWeight value 100")
        eng.send("setoption name NNUEMaterial value 50")
        eng.set_position("4k3/8/8/8/8/8/8/QQ2K3 w - - 0 1")
        d = eval_now(eng)
        check(d.get("material_cp") == 1800, "material_cp white two queens",
              str(d))
        check(d.get("net_material") == 900, "net_material 50% of 1800",
              str(d))
        check(d.get("total") == d.get("neural", 0) + 900,
              "material applied before direct eval", str(d))
        eng.send("setoption name NNUEMaterial value 0")
        eng.wait_ready()

        # A teacher model (hidden > 512) must be rejected, not deployed.
        import struct
        teacher = os.path.join(ROOT, "temp", "nnue_teacher_header.nnue")
        with open(os.path.join(FIXTURES, "tiny.nnue"), "rb") as f:
            data = bytearray(f.read())
        struct.pack_into("<I", data, 16, 1024)
        with open(teacher, "wb") as f:
            f.write(data)
        start = eng.mark()
        eng.send(f"setoption name EvalFile value {teacher}")
        eng.wait_ready()
        lines = eng.collect(start)
        check(any("EvalFile: failed" in ln for ln in lines),
              "teacher hidden>512 rejected", str(lines[-2:]))
        check(eng.alive(), "engine alive after teacher rejection")
        d = eval_now(eng)
        check(d.get("total") == d.get("classical(stm)"),
              "classical fallback after teacher rejection", str(d))

        # Piped bench sanity (argv is not used by this engine).  Disable the
        # network first: a deliberately weak/random real model can make a
        # depth-10 tree explode, which is not what this smoke check measures.
        eng.send("setoption name UseNNUE value false")
        eng.send("setoption name EvalFile value <empty>")
        eng.wait_ready()
        start = eng.mark()
        eng.send("bench 2")
        check(eng.wait_for("bench nodes", timeout=120, start=start),
              "bench via stdin", "")
    finally:
        eng.close()

    if failures:
        print(f"\nrun_nnue: {len(failures)} failure(s)")
        for f in failures[:40]:
            print("  -", f)
        return 1
    print("run_nnue: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
