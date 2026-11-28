# LazyCatBot NNUE Pilot — Multi-Agent Plan

Status: active (2026-11-26)
Coordinator: main agent. All work stays inside `C:\Github\lazycatbot`.
All Python dependencies live in `.venv\`; never install into the system interpreter.
Tests are local-only: no Lichess/online games, no external bot matches. Data
downloads from `database.lichess.org` are allowed; gameplay and validation stay local.

Target machines (for model sizing):

- Training/optimization: i5-12400 (6C/12T), 32 GB RAM, RTX 5060 Ti 16 GB.
- Deployment A: 8 cores, 16 GB RAM, 6 GB GPU.
- Deployment B: 8 cores, 16 GB RAM, RTX 3050 8 GB.

## 1. Goal and scope

Pilot goal: prove or reject, with measurements, that a HalfKP value network plus a
correct incremental CPU inference path is stronger than the current classical
evaluation on this repository.

Strong play is the final goal. Analysis (PV, per-move scores, policy) is part of
the loop, not a replacement for search:

- value network improves leaf evaluation;
- policy / PV data improves move ordering and move prediction;
- alpha-beta search provides real multi-ply lookahead;
- SPRT matches decide whether a change is kept.

Non-goals for this pilot:

- no AlphaZero/MCTS engine;
- no GPU inference inside the alpha-beta hot path;
- no claim of a specific Elo before SPRT;
- no 3000-Elo promises.

## 2. Agents and ownership

Each agent owns the listed files. Do not edit files owned by another agent.

| Agent | Role | Owned paths |
|---|---|---|
| A | Environment, build, baseline | `.venv/`, `build-baseline/`, `temp/baseline-worktree/`, `docs/reports/BASELINE.md` |
| B | Data pipeline | `training/download_lichess_evals.py`, `training/extract_halfkp.cpp`, `training/data_stats.py`, `training/verify_lichess_pov.py`, `training/data_contract.md`, `training/tests_data/` |
| C | Training and export | `training/train_halfkp.py`, `training/export_nnue.py`, `training/reference_inference.py`, `training/model_contract.md`, `training/tests_train/` |
| D | C++ NNUE loader, accumulator, inference | `src/nnue.hpp`, `src/nnue.cpp`, `src/engine.hpp`, `src/eval.cpp`, `src/uci.cpp`, `CMakeLists.txt`, `build.ps1`, `tests/nnue_*` |
| E | Validation harness and reports | `tools/nnue_validate.py`, `tools/nnue_bench.py`, `tools/nnue_sprt.py`, `docs/reports/VALIDATION.md`, `tests/test_nnue_validation.py` |
| Coordinator | Contracts, integration, review, final SPRT | `docs/NNUE_PILOT.md`, `docs/reports/UNRESOLVED.md`, `temp/` |

Agents may add new files inside their owned directories. Any cross-cutting change
must be documented in the agent report and in `docs/reports/UNRESOLVED.md`.

## 3. Fixed contracts

### 3.1 HalfKP feature function (training and inference)

```
perspective = WHITE or BLACK
ownKing    = kingSq[perspective]
bucket     = (perspective == WHITE) ? ownKing : (ownKing ^ 56)
for every piece (color, type, sq):
    if color == perspective and type == KING: continue   # own king is the bucket
    osq = (perspective == WHITE) ? sq : (sq ^ 56)
    rel = (color == perspective) ? 0 : 1
    idx = (bucket * 12 + rel * 6 + type) * 64 + osq
inputs = 64 * 12 * 64 = 49152
```

The training extractor writes features for the side to move (`perspective = pos.side`).
The C++ inference path computes both perspectives and selects `pos.side` at eval time.

### 3.2 Data shards (produced by extract_halfkp)

All files in one directory, little-endian:

| File | Type | Shape | Meaning |
|---|---|---|---|
| `features.u16` | uint16 | [N, 32] | HalfKP indices, zero-padded; count in `counts.u8` |
| `counts.u8` | uint8 | [N] | active feature count (<= 32) |
| `targets.f32` | float32 | [N] | win probability in [0,1], side-to-move |
| `cp.f32` | float32 | [N] | optional raw cp, side-to-move |
| `moves.u16` | uint16 | [N] | policy target index, 0xFFFF if absent |
| `pv.u16` | uint16 | [N, 8] | normalized UCI moves, 0xFFFF padding |
| `groups.u32` | uint32 | [N] | group hash (game / cluster), for split |
| `meta.json` | JSON | — | filters, counts, feature version, POV check |

Policy index encoding (4352 classes):

- normal move: `from * 64 + to`
- promotion: `4096 + (promo - 2) * 64 + from`, promo: 2=N, 3=B, 4=R, 5=Q
- castling: normalize to king from-to (`e1g1`, `e1c1`, `e8g8`, `e8c8`)
- absent: `0xFFFF`

Lichess PV lines are UCI_Chess960; the extractor must normalize castling to
standard UCI before encoding.

### 3.3 `.nnue` model file v1

Little-endian. Fixed 64-byte header:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | magic `LCNN` |
| 4 | 4 | version = 1 |
| 8 | 4 | arch = 1 (HalfKP) |
| 12 | 4 | inputs = 49152 |
| 16 | 4 | hidden (128/256/384/512) |
| 20 | 4 | buckets = 64 |
| 24 | 4 | quant = 1 (FT int16, dense float32) |
| 28 | 4 | reserved = 0 |
| 32 | 4 | ft_scale (float32) |
| 36 | 4 | policy_size (0 or 4352) |
| 40 | 8 | checksum (FNV-1a 64 over payload) |
| 48 | 16 | reserved = 0 |

Payload in order:

1. `int16 ft[inputs * hidden]` row-major, input-major, quantized as `round(w / ft_scale)`
2. `float ft_bias[hidden]`
3. `float fc1_w[hidden * 32]` row-major (input-major)
4. `float fc1_b[32]`
5. `float fc2_w[32 * 32]` row-major
6. `float fc2_b[32]`
7. `float out_w[32]`
8. `float out_b[1]`
9. if `policy_size > 0`: `float policy_w[hidden * policy_size]`, `float policy_b[policy_size]`

Forward pass: `acc = sum(int16 ft rows)` in int32; `h = acc * ft_scale + ft_bias`;
`h = max(h, 0)`; `fc1 -> max(0)`; `fc2 -> max(0)`; `out`; return cp or WDL logit.
The training exporter must also write `model_card.json` with metrics and config.

### 3.4 UCI options (contract)

- `EvalFile` (string, default empty) — path to `.nnue`.
- `EvalBackend` (string: `cpu` or `none`, default `cpu`).
- `UseNNUE` (check, default false until accepted).
- Existing `NNUEWeight`/`NNUEClip` may remain but direct eval is the default when enabled.
- On any load failure: fall back to classical, print `info string` warning, never crash.

## 4. Build and environment

- Python: `C:\Github\lazycatbot\.venv\Scripts\python.exe` (Python 3.11).
- Required packages: numpy, torch (CUDA 12.8 build), zstandard, python-chess,
  orjson, pytest, tqdm.
- C++: Visual Studio 2026 at
  `C:\Program Files\Microsoft Visual Studio\18\Community`; use `vcvars64.bat`
  and CMake. Output: `build/chessbot.exe`.
- If MSVC cannot build the current tree, document the exact failure and try a
  portable MinGW toolchain inside `tools/` (project-local only).
- Release builds must not use `-march=native`; AVX2 plus scalar fallback for
  portability to weaker machines.

## 5. Acceptance criteria (pilot)

1. `build/chessbot.exe` builds and passes the existing perft/protocol tests.
2. C++ NNUE output matches the Python reference within 2 cp on 100 FENs.
3. NPS loss with NNUE enabled is <= 10% (target <= 5%) on `bench 10`.
4. Value net: calibration curve is sane; material sanity positions have correct
   signs; SPRT vs classical gives a positive result at equal nodes, and no loss
   at equal time, with LOS >= 95%.
5. Policy (if implemented): top-1/top-3 on held-out PV is clearly above random,
   and move ordering reduces nodes-to-depth; no strength regression.
6. No regression in `tests/run_all.py`; NNUE remains off by default until accepted.

## 6. Unresolved questions log

Record blockers and decisions needed from the human in `docs/reports/UNRESOLVED.md`.
Do not silently guess on licensing, hardware availability, or data source changes.

## 7. Training tiers (local GPU and A6000 90 GB)

The pilot separates teacher training from deployable student training.

| Tier | Hardware | Data | Model | Purpose |
|---|---|---|---|---|
| Pilot | RTX 5060 Ti 16 GB | 3–5M | HalfKP 256 | prove pipeline, offline metrics, first SPRT |
| Teacher | rented A6000 Blackwell 90 GB, Debian 13 | 20–100M | HalfKP 1024 (or 1536) | strong evaluator for analysis and distillation |
| Student | 5060 Ti or A6000 | 20–50M | HalfKP 256/384 | deployable CPU model, quantized |
| Scale | A6000 | 100M+ | 512/1024 | iterative self-play and larger nets |

Rules:

- The teacher is not shipped into the alpha-beta hot path. It is used for
  analysis, GPU batch evaluation, and soft targets for distillation.
- The student is what gets exported to `.nnue` for CPU play on 8-core/16 GB
  machines. It must meet the NPS budget.
- Distillation: teacher WDL/cp soft targets + hard Lichess labels + optional
  policy targets; then quantization-aware fine-tuning of the student.
- The A6000 run must be reproducible: same commit, `requirements-lock.txt`,
  config JSON, seed, data manifest, and checkpoint hashes.
- If the A6000 is not directly reachable from this session, prepare a
  self-contained bundle and runbook (`training/RUNBOOK_A6000.md`) that the human
  can copy and execute. Do not put secrets or Lichess tokens in the bundle.
- Data transfer: prefer extracted shards (a few GB) over the 20.9 GiB raw
  `.zst`; if the A6000 has bandwidth, it may download the raw DB directly and
  reuse the same `manifest.json`/pipeline.
- A 90 GB GPU removes VRAM limits, not data limits: batch size and hidden width
  can grow, but training quality still depends on label quality and validation
  discipline. Do not skip SPRT because the teacher looks strong offline.

Remote A6000 (Debian 13) rules:

- Source of truth stays the local project. Remote work happens in a single
  project directory (for example `~/lazycatbot`) with its own `.venv`; no
  system-wide `pip install`, no secrets in the bundle.
- Check `nvidia-smi` and the driver first. Blackwell needs a recent driver and
  a CUDA 12.8+ PyTorch wheel; if the wheel does not import, record the driver
  version and fall back to the newest compatible CUDA wheel.
- Transfer extracted shards (a few GB) rather than the 20.9 GiB raw `.zst` when
  possible. If the server has good bandwidth, it may download the raw DB itself
  and reuse the same pipeline/manifest. Verify SHA-256 on the remote side.
- Run training under `tmux`/`nohup`, log to a file, checkpoint every epoch, and
  support `--resume`. The SSH session may drop.
- The remote bundle must contain: training code, `requirements-remote.txt`,
  config JSON, data manifests/checksums, `setup_remote.sh`, and
  `RUNBOOK_A6000.md`. It must not contain datasets, checkpoints, SSH keys, or
  Lichess tokens.
- Remote teacher output: `weights_best.pt` + `weights_best.nnue` +
  `model_card.json` + logs. The `.nnue` is still tagged non-deployable if the
  hidden width exceeds the CPU budget; the student export is what ships.
- Distillation can run on the A6000 or locally. The final acceptance test is
  always the local CPU student in SPRT, not the teacher's offline metrics.
