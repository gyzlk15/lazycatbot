# Data pipeline tests

Synthetic fixtures for the HalfKP data pipeline. No network access and no
Lichess/online games: every test passes `--no-download`.

## Run

```bat
.venv\Scripts\python.exe -m pytest training\tests_data -q --basetemp=temp\pytest
```

`--basetemp=temp\pytest` is required on this machine: the default pytest temp
directory under the user profile raises `PermissionError`.

## What is covered

| Test area | Checks |
|---|---|
| downloader selection | independent reimplementation of the documented rules (highest depth, mate/cp/depth/cp-range filters, FEN validation, dedup) must match line-for-line |
| POV conversion | every Black-to-move record's cp is the negated White-POV value; `wdl == sigmoid(cp/400)` within 2e-6 |
| Chess960 | `e1h1`/`e8h8` from the fixture are rewritten to `e1g1`/`e8g8`; no `e1h1` survives; the castle *after* the first PV move is normalized too |
| promotions | `a7a8q` policy index uses `4096 + (promo-2)*64 + from` |
| legacy input | `FEN|target` becomes `nan` cp and no policy; invalid FENs are rejected; duplicate PV tokens collapsed |
| sampling / resume | stride sampling is byte-identical across runs; a hand-written checkpoint resumes to the exact same output as a full run |
| extractor | selftest; shard sizes; features/counts/targets/cp/moves/pv/groups match an independent Python reference extractor; meta.json fields; `--cp` legacy mode; invalid-input skip counts |
| data stats | all checks pass on healthy shards; a real duplicate exceeds the 32-bit collision budget and fails the check; broken shard sizes are rejected |
| POV verifier | fixture is detected as White POV; a deliberately side-to-move fixture is rejected; Chess960 tokens counted; standard-board raw replay fails on 960 castles; the pipeline text is detected as side-to-move POV |
| manifest tool | `make_pilot_manifest.py` records sizes/hashes/counts and the recorded SHA-256 matches the file |

The extractor tests need `training/work/extract_halfkp.exe` (or
`HALFKP_EXTRACTOR=<path>`). They skip with the build command when it is absent;
the downloader / verifier / stats tests still run. `test_extractor_selftest`
also fails if the exe is older than `extract_halfkp.cpp`, because a failed
build leaves the previous binary in place and would silently test stale code.

## Fixtures

Regenerate (deterministic, seed 20261126):

```bat
.venv\Scripts\python.exe training\tests_data\make_fixtures.py
```

- `fixture_evals.jsonl` — 42 Lichess-shaped records: 26 material-driven
  positions (mixed side to move), standard castling encoded as UCI_Chess960
  (both colors), promotion, a 14-move PV, a duplicated PV token, a highest-depth
  eval that is not the first one, an equal-depth tie, an exact duplicate FEN, a
  PV without a `line`, and the drop cases (mate, depth 12, cp 2000, no pvs, no
  cp, invalid FEN, empty evals).
- `fixture_legacy.txt` — old `FEN|target` lines, a v2 line with two castles
  (Black then White), a promotion with a duplicate token, and a malformed FEN.
- `fixture_resume.txt` — 24 distinct positions for the resume test.

The independent Python reference implementations in `test_pipeline.py`
(HalfKP features, policy indices, FNV group hash) are intentionally written
from the contract, not imported from the C++ or the downloader.
