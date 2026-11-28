# training/tests_train

Local-only tests for the training/export/inference contract. They use tiny
synthetic shards generated from random-chess playouts, so no dataset download
is required and nothing here is a strength test.

```powershell
# Windows (from the repo root)
.venv\Scripts\python.exe -m pytest training\tests_train -q
# if the system temp directory is not writable:
.venv\Scripts\python.exe -m pytest training\tests_train -q --basetemp training\work\pytest-tmp
```

```bash
# Debian / macOS
.venv/bin/python -m pytest training/tests_train -q
```

Coverage:

| File | What it checks |
|---|---|
| `test_contract.py` | HalfKP indices for a known FEN, own-king exclusion, material decode, policy encoding, 64-byte header fields, payload/file sizes, FNV-1a checksum, reserved bytes, dense exactness, FT quantization/clipping, cp output units, policy layout |
| `test_pipeline.py` | profile defaults, project-relative paths, group-disjoint deterministic split, CPU train → export → NumPy reload, optional cp/moves files, checkpoint resume, `--export-only`, teacher non-deployable marker, GPU-resident batch gather vs CPU (skipped without CUDA) |

Expected runtime: under a minute on CPU; the GPU gather test adds a few
seconds when CUDA is available.

Notes:

- The synthetic generator lives in `train_halfkp.make_smoke_shards` and is
  wrapped by `synthetic.py`; it writes `features.u16`, `counts.u8`,
  `targets.f32`, `cp.f32`, `moves.u16`, `pv.u16`, `groups.u32`, `meta.json`.
- Metrics printed by these tests are pipeline checks only. Do not cite them as
  evidence of playing strength.
