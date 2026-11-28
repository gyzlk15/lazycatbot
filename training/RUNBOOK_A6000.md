# RUNBOOK — teacher training on a rented A6000 (90 GB), Debian 13

This runbook trains the **teacher** HalfKP net on a remote CUDA host, without
touching the local Windows checkout. The teacher is an analysis/distillation
artifact: files with `hidden > 512` are exported as non-deployable
(`model_card.json` → `"deployable": false`). Only the **student** export
(`--profile student`, hidden 512) may be wired into the engine, and only after
the validation/SPRT gates in `docs/NNUE_PILOT.md`.

All commands assume:

```bash
export PROJECT="$HOME/lazycatbot"      # repo root on the server
export DATA="/data/lazycatbot/pilot"   # extracted shard directory
export WORK="/data/lazycatbot/teacher" # checkpoints / exports / logs
cd "$PROJECT"
source .venv/bin/activate              # after setup_remote.sh
```

## 1. What to transfer

Two options; upload the **extracted shards** unless you specifically want to
re-extract on the server.

### A. Extracted shards (recommended)

`extract_halfkp` writes one directory with (bytes per position):

| File | Type | Bytes/pos | 20M positions | 100M positions |
|---|---|---:|---:|---:|
| `features.u16` | [N,32] u16 | 64 | 1.28 GB | 6.40 GB |
| `counts.u8` | [N] u8 | 1 | 20 MB | 100 MB |
| `targets.f32` | [N] f32 | 4 | 80 MB | 400 MB |
| `groups.u32` | [N] u32 | 4 | 80 MB | 400 MB |
| `cp.f32` (optional) | [N] f32 | 4 | 80 MB | 400 MB |
| `moves.u16` (optional) | [N] u16 | 2 | 40 MB | 200 MB |
| `pv.u16` (optional) | [N,8] u16 | 16 | 320 MB | 1.60 GB |
| `meta.json` | JSON | — | — | — |

Base shards (no cp/moves/pv): **≈1.4 GiB @20M**, **≈6.8 GiB @100M**.
Full shards: **≈1.8 GiB @20M**, **≈8.8 GiB @100M**.
GPU residency (`--data-on-gpu`) uses int32 features on the GPU: ~2.6 GB @20M,
~12.8 GB @100M plus ~1 GB of labels — fine on 90 GB, not on 16 GB.

### B. Raw Lichess eval database

`lichess_db_eval.jsonl.zst` is **20.9 GiB** compressed. If you upload it, build
the extractor on the server from the repo (`./training/auto_train.py` path) or
ship `src/` in the bundle (`make_bundle.py --with-src`). Extraction is
single-threaded and I/O heavy; expect it to dominate the session.

## 2. Build and verify the bundle locally

```powershell
# local Windows checkout, inside C:\Github\lazycatbot
.venv\Scripts\python.exe training\make_bundle.py
# optional: include src/ so the C++ extractor can be rebuilt remotely
.venv\Scripts\python.exe training\make_bundle.py --with-src
```

Output (default `temp/bundles/`): `lazycatbot_nnue_bundle_<UTC>.tar.gz` and a
`.sha256` sidecar. The archive contains `MANIFEST.json` and `SHA256SUMS`; the
builder reopens the tar and verifies every payload hash. It refuses files that
contain private-key/token markers and excludes `.venv`, `training/work`,
checkpoints, `*.nnue`, `*.pt`, datasets and `temp/`.

Upload and verify:

```bash
scp temp/bundles/lazycatbot_nnue_bundle_*.tar.gz user@host:/data/lazycatbot/
ssh user@host
cd /data/lazycatbot
sha256sum -c lazycatbot_nnue_bundle_*.tar.gz.sha256
tar -xzf lazycatbot_nnue_bundle_*.tar.gz
mv lazycatbot_nnue_bundle_* lazycatbot   # bundle root becomes $PROJECT
```

Bundle contents: `training/*.py`, `training/tests_train/`, requirements,
`RUNBOOK_A6000.md`, `model_contract.md`, `docs/NNUE_PILOT.md`,
`docs/reports/UNRESOLVED.md`, `MANIFEST.json`, `SHA256SUMS`.

Verify the extracted shards (run on the machine that holds them; put the
output in `SHA256SUMS.shards`):

```bash
cd "$DATA"
ls -lh   # expect features.u16 counts.u8 targets.f32 groups.u32 [+ cp/moves/pv] meta.json
sha256sum features.u16 counts.u8 targets.f32 groups.u32 > SHA256SUMS.shards
sha256sum -c SHA256SUMS.shards
cat meta.json
```

If `meta.json` is absent (the C++ extractor writes `meta.txt`), check
`meta.txt`. The trainer reports the group count and warns that
`extract_halfkp`'s group hash is per-FEN unless the extractor clusters by game;
confirm with the coordinator before treating the split as game-disjoint.

## 3. Server setup (no system-wide pip)

```bash
cd "$PROJECT"
chmod +x training/setup_remote.sh
./training/setup_remote.sh
```

The script:

- requires `python3 >= 3.11` (Debian 13 ships 3.13), installs nothing system-wide;
- creates `.venv/` in the project root;
- `pip install -r training/requirements-remote.txt` (torch 2.11.0, cu128);
- prints `df -h`, `nproc`, `free -h`, `nvidia-smi`;
- runs a CUDA smoke test: bf16 matmul + sparse `EmbeddingBag` + `SparseAdam`.

If `python3-venv` is missing:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
```

For a cu129 image, edit the first line of `training/requirements-remote.txt`:

```text
--extra-index-url https://download.pytorch.org/whl/cu129
```

and re-run `pip install -r training/requirements-remote.txt`.

## 4. Smoke tests (do these before the long run)

CPU sanity (synthetic data, seconds):

```bash
python training/train_halfkp.py --smoke --out /tmp/lcb_smoke_cpu --device cpu
```

GPU + GPU-resident data + AMP + gradient accumulation:

```bash
python training/train_halfkp.py --smoke 8192 --out /tmp/lcb_smoke_gpu \
    --device cuda --amp bf16 --data-on-gpu --grad-accum 2 --pin-memory
```

Check that it printed `exported .../model.nnue`, no `export vs torch` warning,
and `material_sanity` has no `sign_ok: false`. These are **pipeline** checks,
not strength evidence.

Run the unit/contract tests:

```bash
python -m pytest training/tests_train -q
```

(If pytest's default temp directory is not writable, add
`--basetemp training/work/pytest-tmp`.)

## 5. Teacher run under tmux

Start from a clean output directory and keep the log on disk:

```bash
mkdir -p "$WORK"
tmux new -s lcb
cd "$PROJECT" && source .venv/bin/activate

python training/train_halfkp.py \
    --profile teacher \
    --data "$DATA" \
    --out "$WORK" \
    --name teacher \
    --data-on-gpu \
    --amp bf16 \
    --batch 65536 \
    --grad-accum 1 \
    --num-workers 2 \
    --pin-memory \
    --epochs 10 \
    --val-fraction 0.02 \
    --max-val-samples 200000 \
    --policy off \
    --log-every 50 \
    2>&1 | tee "$WORK/train.log"
```

Detach with `Ctrl-b d`; reattach with `tmux attach -t lcb`.

Notes:

- `--profile teacher` sets hidden 1536, batch 65536, lr 5e-4, epochs 10, bf16.
  Explicit flags still win; `--hidden 1024` is allowed.
- `--policy off` keeps the 4352-class head out of the teacher (value
  distillation). Remove it to train the policy head as well; it is much more
  expensive at batch 65536.
- `--data-on-gpu` uploads the shards once, then gathers batches on-device.
  Without it the memmap/prefetch path is used; start there if VRAM is tight or
  the shard set is larger than the GPU. `--num-workers` parallelises memmap
  reader threads and has no effect when the shards are GPU-resident.
- `--smoke` shrinks any flag that still equals its profile default; pass
  explicit non-default values (`--epochs 3`, `--hidden 1024`, ...) to keep them.
- The trainer exports `.nnue` + `model_card.json` on every best epoch, saves
  `checkpoint.pt` every epoch, and stops early after 3 epochs without
  improvement (`--patience 0` disables). `--export-only --resume <dir>`
  re-exports the last checkpoint without further training.
- Offline metrics (loss/Brier/ECE/per-phase/policy) do **not** prove playing
  strength; SPRT does.

### nohup fallback (no tmux)

```bash
cd "$PROJECT" && source .venv/bin/activate
nohup python training/train_halfkp.py \
    --profile teacher --data "$DATA" --out "$WORK" --name teacher \
    --data-on-gpu --amp bf16 --batch 65536 --policy off \
    > "$WORK/train.log" 2>&1 &
echo $! > "$WORK/train.pid"
tail -f "$WORK/train.log"
```

### Monitor

```bash
tail -f "$WORK/train.log"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv -l 10
ls -lh "$WORK"
```

Check the first 100 optimizer steps: `samples/s` in the log is the real
throughput; extrapolate epoch time before committing the whole session.
Hidden 1536 policy-off on a 90 GB A6000 should keep the GPU busy; if
`nvidia-smi` shows low utilization, raise `--num-workers` and check storage
latency.

## 6. Resume after interruption

`--resume` accepts the checkpoint file or its directory. Keep the same data,
batch/grad-accum and seed so the group split and shuffle order match:

```bash
python training/train_halfkp.py \
    --profile teacher --data "$DATA" --out "$WORK" --name teacher \
    --data-on-gpu --amp bf16 --batch 65536 --policy off \
    --epochs 10 --resume "$WORK" \
    2>&1 | tee -a "$WORK/train.log"
```

Resume starts at the epoch after the checkpoint's last completed epoch and
restores model/optimizer/scaler state, best value, history and early-stop
counter. It does not resume mid-epoch. If you change `--batch`,
`--grad-accum` or the scheduler, the LR position changes but training still
resumes correctly.

## 7. Export and validate

Automatic: best-epoch `.nnue` and `model_card.json` are written to `$WORK`.
Manual re-export of the last checkpoint (e.g. to regenerate the card):

```bash
python training/train_halfkp.py \
    --profile teacher --data "$DATA" --out "$WORK" --name teacher \
    --data-on-gpu --amp bf16 --batch 65536 --policy off \
    --resume "$WORK" --export-only
```

Structure/checksum/quantization check (no torch needed):

```bash
python training/export_nnue.py --check "$WORK/teacher.nnue" --json
python training/reference_inference.py --model "$WORK/teacher.nnue" --header
python training/reference_inference.py --model "$WORK/teacher.nnue" \
    --fen "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
```

Acceptance (from the plan): exported cp must match the NumPy reference within
2 cp on the same FENs; the trainer records `checks.export_vs_torch` and
`checks.material_sanity` in `model_card.json`. A teacher card with
`"deployable": false` is expected — do not ship it to the engine.

## 8. Download the artifacts

```bash
# from the local machine
mkdir -p training/work/teacher
rsync -avP user@host:/data/lazycatbot/teacher/teacher.nnue \
    user@host:/data/lazycatbot/teacher/model_card.json \
    user@host:/data/lazycatbot/teacher/metrics.json \
    training/work/teacher/
# checkpoint + optimizer state are large (~150-500 MB for hidden 1536)
rsync -avP user@host:/data/lazycatbot/teacher/checkpoint.pt \
    training/work/teacher/
```

Verify on the local machine:

```powershell
.venv\Scripts\python.exe training\export_nnue.py --check training\work\teacher\teacher.nnue
.venv\Scripts\python.exe training\reference_inference.py --model training\work\teacher\teacher.nnue --fen "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
```

## 9. Student export (the deployable one)

The teacher is not deployable. Train the student on the same shards (or on
teacher-labeled data, once that pipeline exists) with the deployable profile:

```bash
python training/train_halfkp.py \
    --profile student \
    --data "$DATA" \
    --out /data/lazycatbot/student \
    --name student \
    --data-on-gpu --amp bf16 --batch 32768 \
    --epochs 8 --policy auto \
    2>&1 | tee /data/lazycatbot/student/train.log
```

`hidden=512` keeps the engine's CPU accumulator inside the pilot budget.
Check `model_card.json` has `"deployable": true` before handing the file to
Agent D/E.

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| `CUDA out of memory` | halve `--batch`, double `--grad-accum`, or drop `--data-on-gpu` |
| `--data-on-gpu` estimate too large | use `--max-samples`, a smaller shard set, or memmap mode |
| low GPU utilization | raise `--num-workers`, ensure shards are on local NVMe, not NFS |
| fp16 instability | use `--amp bf16` (default on Blackwell) |
| sparse optimizer unsupported | `--dense-emb` (slow, high VRAM; last resort) |
| `checksum mismatch` in export | re-export; never hand-edit `.nnue` bytes |
| split group warning | confirm whether `groups.u32` is per-game; contact the coordinator |
| `export vs torch` > 2 cp | stop, keep the checkpoint, report to Agent C |
| download interrupted | `rsync --partial --append-verify` |
| disk full | keep only latest `checkpoint.pt` + best `.nnue`; delete `*.npz`/`*.pt` duplicates |

## 11. Exact command summary

```bash
# setup
tar -xzf lazycatbot_nnue_bundle_*.tar.gz && mv lazycatbot_nnue_bundle_* lazycatbot
cd lazycatbot && ./training/setup_remote.sh

# smoke
python training/train_halfkp.py --smoke --out /tmp/smoke_cpu --device cpu
python training/train_halfkp.py --smoke 8192 --out /tmp/smoke_gpu --device cuda --data-on-gpu --amp bf16

# teacher (tmux)
python training/train_halfkp.py --profile teacher --data /data/lazycatbot/pilot \
    --out /data/lazycatbot/teacher --name teacher --data-on-gpu --amp bf16 \
    --batch 65536 --num-workers 2 --pin-memory --policy off --log-every 50

# resume
python training/train_halfkp.py --profile teacher --data /data/lazycatbot/pilot \
    --out /data/lazycatbot/teacher --name teacher --data-on-gpu --amp bf16 \
    --batch 65536 --policy off --resume /data/lazycatbot/teacher

# verify / export
python training/export_nnue.py --check /data/lazycatbot/teacher/teacher.nnue --json
python training/reference_inference.py --model /data/lazycatbot/teacher/teacher.nnue \
    --fen "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
```
