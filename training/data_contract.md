# HalfKP training data contract

Status: active (2026-11-26). Owner: Agent B (data pipeline).
Everything here is implemented in the files below and covered by
`training/tests_data/` (17 tests, all passing).

| Stage | File | Output |
|---|---|---|
| verify source conventions | `training/verify_lichess_pov.py` | `training/work/pov_report.md` / `.json` |
| sample the dump | `training/download_lichess_evals.py` | `positions.txt`, `positions.txt.stats.json` |
| extract shards | `training/extract_halfkp.cpp` | `features.u16`, `counts.u8`, `targets.f32`, `cp.f32`, `moves.u16`, `pv.u16`, `groups.u32`, `meta.json` |
| audit shards | `training/data_stats.py` | text report + JSON |

## 1. Source: `lichess_db_eval.jsonl.zst` (CC0)

Official schema (database.lichess.org): one JSON object per line,

```
{"fen": "...", "evals": [{"depth": N, "knodes": K,
                          "pvs": [{"cp": C | "mate": M, "line": "e2e4 e7e5 ..."}]}]}
```

Conventions that the pipeline depends on (all three verified against the dump;
see `training/work/pov_partial.md` for the provisional run):

1. **`cp` / `mate` are White POV** in every eval and every PV.
   Confirmed externally by the database docs and the
   `Lichess/fishnet-evals` dataset card ("Always given from White's point of
   view"), and empirically by material correlation. On 1,000,000 scanned lines
   (4,000 sampled): Black-to-move subset `corr(cp, White-minus-Black material)
   = +0.22`, sign agreement 0.61. A side-to-move file would show the opposite
   sign on that subset. (Material is a noisy proxy on real positions: sharp
   tactics and odds/study start FENs keep the pooled correlation near 0.3, so
   the Black-to-move split is the decisive test.)
2. **`line` is a UCI move list with UCI_Chess960 castling**: the king moves
   onto its own rook (`e1h1`, `e1a1`, `e8h8`, `e8a8`), not `e1g1`.
   Evidence: 4,000/4,000 sampled PVs are legal under a Chess960 replay;
   1,040 tokens in 763 lines are king-to-own-rook castles, and those raw
   tokens are **not** legal moves on a standard board. The pipeline rewrites
   them to standard UCI before any policy encoding.
3. **Pick the highest-depth eval, first PV** (also the recommendation on the
   database page). The `depth` of the chosen eval is recorded in the
   downloader stats histogram; it is not stored per position in the shards.

FEN fields: the dump uses 4 fields (board, side, castling, ep) and may use 6.
The pipeline accepts 4–6.

## 2. `positions.txt` — training text format (`halfkp-text-v2`)

One position per line, UTF-8, `\n`, exactly five `|`-separated fields:

```
FEN|wdl|cp|bestmove|pv
```

| Field | Type | Meaning |
|---|---|---|
| `FEN` | string | position before the move, 4–6 fields |
| `wdl` | float, 7 decimals | **side-to-move** win probability, `sigmoid(cp/400)` |
| `cp` | integer or `nan` | **side-to-move** centipawns; `cp = (White to move ? cp_white : -cp_white)` |
| `bestmove` | UCI or empty | first PV move, standard UCI castling, `""` if unknown |
| `pv` | space-separated UCI | up to 8 normalized moves; `""` if unknown |

Rules applied by `download_lichess_evals.py`:

- choose the highest-depth eval (ties: first), then `pvs[0]`;
- drop records with `mate`, missing/NaN `cp`, `depth < --min-depth` (default
  20), `|cp| > --max-cp` (default 1500);
- convert White POV → side-to-move POV (negate for Black), then
  `wdl = sigmoid(cp_stm/400)`;
- normalize UCI_Chess960 castling (king→rook) to standard UCI (king→king) in
  both `bestmove` and every PV move; a castle after the first move is handled
  as well (the test is "from holds a king and to holds that king's rook");
- deduplicate PV tokens and cap the PV at 8 moves;
- 64-bit hash dedup of `board|stm|castling|ep` (move counters ignored) with a
  compact numpy open-addressing set — no Python string sets;
- optional deterministic sampling: `--sample-stride N` (every Nth line) and/or
  `--sample-rate R` (CRC32 of the raw line + `--seed`), applied before JSON
  parsing so unselected lines are cheap;
- resume: `<out>.resume.json` + `<out>.stats.json` are written periodically.
  Re-running the same command truncates to the checkpoint offset, reloads the
  already written FENs into the dedup table, and continues at the checkpointed
  line index. Filters are part of the checkpoint signature; a mismatch aborts.

Legacy input: a non-`.zst` input (or `--format text`) accepts both
`FEN|target` (old downloader / `quiet-labeled`, target is a win probability or
game result) and the five-field format. Missing cp is written as `nan`.

`positions.txt.stats.json` records the filter settings, rejection reasons
(`json`, `fen`, `no_evals`, `no_pvs`, `mate`, `no_cp`, `shallow`, `cp_range`),
the chosen-depth histogram, PV length histogram, castling normalization count,
dedup hits, and `input_truncated` for provisional sampling.

## 3. Shards (`extract_halfkp`)

All files little-endian in one directory:

| File | Type | Shape | Meaning |
|---|---|---|---|
| `features.u16` | uint16 | `[N, 32]` | HalfKP indices, zero padded |
| `counts.u8` | uint8 | `[N]` | active feature count (≤ 32; 2..31 observed) |
| `targets.f32` | float32 | `[N]` | side-to-move win probability |
| `cp.f32` | float32 | `[N]` | raw side-to-move cp; NaN when unknown (legacy input) |
| `moves.u16` | uint16 | `[N]` | policy index of `bestmove`, `0xFFFF` if absent |
| `pv.u16` | uint16 | `[N, 8]` | policy indices of the PV, `0xFFFF` padded |
| `groups.u32` | uint32 | `[N]` | FNV-1a 32 of `board|stm|castling|ep` |
| `meta.json` | JSON | — | filters, counts, feature version, POV statement, source stats |

Feature function (fixed contract, `docs/NNUE_PILOT.md` 3.1):

```
perspective = side to move
bucket = (perspective == WHITE) ? kingSq : (kingSq ^ 56)
for every piece except the perspective king:
    osq = (perspective == WHITE) ? sq : (sq ^ 56)
    rel = (piece color == perspective) ? 0 : 1
    idx = (bucket * 12 + rel * 6 + pieceType) * 64 + osq
inputs = 64 * 12 * 64 = 49,152
```

Policy encoding (4352 classes, `docs/NNUE_PILOT.md` 3.2):

- normal: `from * 64 + to`
- promotion: `4096 + (promo - 2) * 64 + from`, promo 2=N, 3=B, 4=R, 5=Q
  (so `a7a8q` = `4096 + 3*64 + 48` = 4336)
- castling: normalized to king from-to first (`e1g1` = 262)
- absent: `0xFFFF`

`pv.u16` holds policy indices (they fit in uint16), not packed text.

Extractor robustness: invalid/short lines, invalid FENs (including positions
the engine's stricter parser rejects, e.g. the side not to move being in
check), missing fields, non-finite targets, duplicate PV tokens, and unknown
promotions are all rejected/skipped and counted in `meta.json`.
Only the first PV move is legality-checked (pseudo-legal generation); the rest
of the PV is trusted and normalized, because Lichess lines come from
Stockfish. A failed first move makes the policy absent (`0xFFFF`) but keeps the
row.

## 4. Exact commands

All commands run from the repository root with the venv Python.

### 4.1 Build the extractor (MSVC)

The plan's source list predates `src/nnue.cpp`; `src/position.cpp` now calls
the HalfKP accumulator, so `src/nnue.cpp` must be linked. `Position` is
~250 KB, hence the 16 MB stack.

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /std:c++20 /O2 /EHsc /F 16777216 /I src ^
   training\extract_halfkp.cpp src\bitboards.cpp src\position.cpp ^
   src\eval.cpp src\nnue_weights.cpp src\nnue.cpp ^
   /Fe:training\work\extract_halfkp.exe
training\work\extract_halfkp.exe --selftest
```

A ready-made script is `temp\build_extractor.bat`. `g++` is not on PATH.

### 4.2 Verify the source conventions (any time; partial DB is fine)

```bat
.venv\Scripts\python.exe training\verify_lichess_pov.py ^
  --input training\work\lichess_db_eval.jsonl.zst ^
  --scan 1000000 --sample 4000 ^
  --check-text training\work\pilot\positions.txt
```

Exit 0 = pass (warnings for a partial download). Reports to
`training/work/pov_report.md|json` by default. Fixture-only mode:
`--fixture training/tests_data/fixture_evals.jsonl --scan 0`.

### 4.3 Full-dump rebuild (run when `lichess_db_eval.manifest.json` exists)

```bat
.venv\Scripts\python.exe training\download_lichess_evals.py ^
  --input training\work\lichess_db_eval.jsonl.zst ^
  --out training\work\pilot\positions.txt ^
  --max-positions 3000000 --min-depth 20 --max-cp 1500 ^
  --sample-stride 100 --seed 1234 --no-resume

training\work\extract_halfkp.exe training\work\pilot\positions.txt ^
  training\work\pilot\halfkp ^
  --source-stats training\work\pilot\positions.txt.stats.json

.venv\Scripts\python.exe training\data_stats.py ^
  --data training\work\pilot\halfkp ^
  --text training\work\pilot\positions.txt ^
  --holdout <external.epd> ^
  --json training\work\pilot\stats.json --md training\work\pilot\stats.md
```

`--holdout` is optional; without an external FEN/EPD/PGN set the report says
explicitly that sibling-position leakage is unmeasured (see §6.1).

Why `--sample-stride 100`: the dump is ordered (time/game based), so taking the
first N accepted lines would be a temporal slice and would also over-represent
the Stockfish version of that slice. Stride 100 scans the whole file and keeps
1% of the lines, spread uniformly; with a ~70% acceptance rate that yields
~2.9M positions. Do not stop early if the count is 2.9–3.0M: that is the
intended spread. To force a full 3.0M, first confirm the actual line count in
`positions.txt.stats.json` after the run and rerun with stride 95 if needed.
`--sample-rate R` is the alternative (hash-based, scans every line and costs a
CRC per line: ~0.0105 gives ~3.07M); it is resumable in exactly the same way.

Measured throughput on this machine (partial prefix, 17.7M lines):

| Path | Rate |
|---|---|
| full parse + selection (stride 1) | ~27,000 accepted positions/s (38k lines/s) |
| stride/rate scan (JSON skipped) | ~650,000–850,000 lines/s |
| extraction (80k positions/s) | 3M in ~38 s |

So a whole-file strided run is ~8–12 minutes; the full-parse prefix mode is
~2 minutes per 4M lines. The official run should use the whole file.

Expand to 5M only if the 3M run and a first training epoch are fast:
`--max-positions 5000000` (the shard format is unchanged).

### 4.4 Tests

```bat
.venv\Scripts\python.exe -m pytest training\tests_data -q --basetemp=temp\pytest
```

`--basetemp` is required on this machine: the default pytest temp dir under
the user profile is not accessible. The extractor tests skip (not fail) when
`training/work/extract_halfkp.exe` and `HALFKP_EXTRACTOR` are missing.

## 5. Canonical pilot: `training/work/pilot/` (stable, 2026-11-26)

Built from the prefix of the dump that was available while the 20.9 GiB
download was still running (3.18 GB at manifest time; no verified source
manifest yet). The directory was built under `training/work/pilot_build/` and
atomically renamed; it must not be rewritten in place. The SHA-256 record is
`training/work/pilot/manifest.json`.

```bat
.venv\Scripts\python.exe training\download_lichess_evals.py ^
  --input training\work\lichess_db_eval.jsonl.zst ^
  --out training\work\pilot_build\positions.txt ^
  --max-positions 3000000 --min-depth 20 --max-cp 1500 ^
  --sample-stride 8 --seed 1234 --allow-truncated --no-resume
training\work\extract_halfkp.exe training\work\pilot_build\positions.txt ^
  training\work\pilot_build\halfkp ^
  --source-stats training\work\pilot_build\positions.txt.stats.json
.venv\Scripts\python.exe training\data_stats.py ^
  --data training\work\pilot_build\halfkp ^
  --text training\work\pilot_build\positions.txt ^
  --json training\work\pilot_build\stats.json --md training\work\pilot_build\stats.md
.venv\Scripts\python.exe training\verify_lichess_pov.py ^
  --input training\work\lichess_db_eval.jsonl.zst --scan 1000000 --sample 4000 ^
  --check-text training\work\pilot_build\positions.txt ^
  --report-md training\work\pilot_build\pov_report.md ^
  --report-json training\work\pilot_build\pov_report.json
.venv\Scripts\python.exe training\tests_data\make_pilot_manifest.py ^
  --dir training\work\pilot_build --final-path training/work/pilot ^
  --commands training\work\pilot_build\build_commands.json
```

Measured: 160 s to stream 33.9M lines and write 3M positions; 37 s to extract
2,999,596 rows (80k positions/s).

| Metric | Value |
|---|---|
| text positions | 3,000,000 (from 33,858,921 lines, stride 8) |
| shard rows | **2,999,596**; 404 rejected (`bad_fen`) |
| shard sizes | features 191,974,144 B; pv 47,993,536 B; cp/targets/groups 11,998,384 B each; moves 5,999,192 B; counts 2,999,596 B |
| directory size | 620,315,396 B (0.578 GiB, 16 files) |
| active features/row | mean 21.6, min 2, max 31 |
| target (win prob) | mean 0.5064 |
| cp (side to move) | mean 11.52, std ~275 |
| material phase (opening / mid / end) | 49.8% / 28.6% / 21.6% |
| duplicate group hashes | 1,115 vs 1,047.5 expected 32-bit collisions |
| policy coverage | 2,999,407 (99.994%); 189 absent (first move not pseudo-legal) |
| PV coverage | 2,999,407 (99.994%); policy == pv[0] for all present rows |
| internal split (`groups % 10`) | 2,699,407 train / 300,189 val; overlap 0 |
| **sibling leakage (PV replay)** | 9,305 replayed siblings found in the pilot; **8,291 (89%) on the other side of the split** |
| POV | source White POV (Black-to-move corr +0.220); output side-to-move POV |
| Chess960 | 4,000/4,000 sampled PVs legal under Chess960; 1,040 king-to-rook castle tokens; all rewritten to standard UCI |
| consumer check | `train_halfkp.Shards` loads: 2,999,596 rows, policy enabled, cp/moves/pv present |

The internal `%10` split is a random-position split (see §6.1). The 89%
sibling estimate is the measured lower bound of same-game leakage: internal
validation loss will be optimistic. Use an external holdout for the real
generalization measure and for the SPRT accept/reject gate.

`manifest.json` records every file with size and SHA-256, the exact commands,
the source byte size at manifest time, the POV/Chess960 summary, and the group
semantics. To hand off to training:

```bat
.venv\Scripts\python.exe training\train_halfkp.py ^
  --data training\work\pilot\halfkp --out training\work\model --hidden 1024 ^
  --batch 16384 --epochs 8 --device cuda
```

To expand to 5M later, build a **new** directory (never rewrite `pilot/`):

```bat
.venv\Scripts\python.exe training\download_lichess_evals.py ^
  --input training\work\lichess_db_eval.jsonl.zst ^
  --out training\work\pilot5m_build\positions.txt ^
  --max-positions 5000000 --min-depth 20 --max-cp 1500 --sample-stride 5
```

When the complete dump and its `manifest.json` exist, the same pipeline can be
rebuilt over the whole file (stride ~100 for ~3M uniform positions, see §4.3).
That is a new dataset, not a mutation of `training/work/pilot/`.

## 6. Known limitations and decisions

1. **No game IDs, no true group split.** The dump has no game identifier, so
   `groups.u32` hashes only the position (`board|stm|castling|ep`).
   `training/train_halfkp.py` splits on `groups % 10 == 0`, which therefore is
   a **random position split**, not a game/position-family split: identical
   positions cannot straddle it, but sibling positions one ply apart from the
   same game can. A 32-bit hash also has ~n²/2³³ accidental collisions (≈1,048
   at 3M); collisions only merge unrelated positions into the same split
   (conservative, not leakage) and are checked against that budget by
   `data_stats.py`.
   Because a true game-level group is not derivable from this source, the
   leakage control must be **external**, and `data_stats.py --holdout FILE`
   measures exact position overlap with one:
   - game-level PGN holdout with game IDs (best; requires a PGN source),
   - the Lichess puzzle database (UNRESOLVED #4; download only when needed),
   - local self-play games (no online games).
   Until such a holdout is measured, validation losses are "random-position
   generalization", not "new-game generalization". Record results accordingly.
   Adding a real game group would require a contract change (`groups`
   semantics), so it is documented instead of silently redefined.

   Measured on the canonical pilot (`data_stats.py --text ...`, PV replay of
   50,000 rows): of 9,305 replayed sibling positions that also exist in the
   pilot, **8,291 (89%) are on the other side of the `%10` split**. That is the
   expected same-game leakage and the reason the internal validation split is
   not a credible strength gate. The number is stored in
   `pilot/stats.json` under `split.sibling_leakage` and in `manifest.json`.
2. **Depth is not stored per position.** The contract fixes the shard files,
   so depth lives only in `positions.txt.stats.json` as a histogram. Add a
   `depth.u16` shard only with coordinator approval.
3. **PV legality**: only the first move is checked (pseudo-legal). The rest of
   the PV is normalized syntactically. Lichess PVs are engine output, and
   4,000/4,000 sampled lines replay legally.
4. **Mates are dropped** (`mate` is never encoded), so the value net learns
   from cp positions only; tactical mates enter through the cp cap and PV
   policy, not as ±infinity targets.
5. **`cp.f32` may be NaN** for legacy `FEN|target` input. Trainers must mask
   non-finite cp; the value target is always finite.
6. **Value target scale** is `sigmoid(cp/400)`, a training label, not a
   calibrated win probability. Calibration is measured after training
   (`docs/NNUE_PILOT.md` acceptance 4).

## 7. Related files

- `docs/NNUE_PILOT.md` 3.1/3.2 — fixed feature and shard contracts.
- `training/train_halfkp.py` — consumes only
  `features/counts/targets/groups`; do not modify for data-format reasons.
- `docs/reports/UNRESOLVED.md` — open decisions (3M vs 5M, policy targets).
