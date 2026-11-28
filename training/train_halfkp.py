#!/usr/bin/env python3
"""Train the LazyCatBot HalfKP pilot network and export a `.nnue` v1 file.

Architecture (docs/NNUE_PILOT.md section 3.1/3.3)::

    49,152 (HalfKP) -> FT hidden (128/256/384/512) -> 32 -> 32 -> 1

Targets and heads
-----------------
* WDL: ``BCEWithLogits`` on ``targets.f32`` (side-to-move win probability).
* cp (optional): smooth-L1 auxiliary loss between the shared logit and
  ``cp / 400``; only active with ``--cp-weight > 0`` and ``cp.f32`` present.
* policy (optional): cross-entropy over 4,352 classes from ``moves.u16``;
  ``0xFFFF`` entries are ignored.  Enabled automatically when ``moves.u16``
  exists (``--policy auto``).

Optimisation
------------
Sparse FT rows use ``SparseAdam``; the dense head and policy layer use
``AdamW``.  CUDA runs bf16 autocast by default (fp16 with a GradScaler is
available), dense gradients are clipped, the LR schedule is warmup+cosine or
step decay, checkpoints are saved every epoch and training stops early when
validation loss stalls.

Split
-----
Validation is split by ``groups.u32`` (unique group ids shuffled with a fixed
seed), never by position index or by ``%10`` of a per-FEN hash.  Train and
validation groups are disjoint and both counts are reported.

CLI profiles set the sizing defaults (``--hidden/--batch/--epochs/--lr/--amp``);
explicit flags always win::

    --profile pilot     hidden 256,  batch 16384  (pilot/CPU-friendly)
    --profile student   hidden 512,  batch 32768  (deployable ceiling)
    --profile teacher   hidden 1536, batch 65536  (A6000-90GB, non-deployable)

Smoke test without the real dataset (CPU, seconds)::

    python3 training/train_halfkp.py --smoke --out training/work/smoke_model

Pilot run::

    python3 training/train_halfkp.py --profile pilot --data training/work/pilot \
        --out training/work/pilot_model --device cuda

Teacher run on the rented A6000 (all shards resident on the GPU)::

    python3 training/train_halfkp.py --profile teacher \
        --data /data/pilot_shards --out /data/teacher_model \
        --data-on-gpu --pin-memory --num-workers 2 --grad-accum 1

All paths are resolved against the repository root when relative, so the same
command works on Windows and Debian.  See ``training/RUNBOOK_A6000.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - environment guard
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

import export_nnue
from reference_inference import (
    CP_SCALE,
    INPUTS,
    MATERIAL,
    PHASE_INC,
    POLICY_SIZE,
    fen_to_halfkp_indices,
    load_nnue,
)

FEATURE_STRIDE = 32
IGNORE_INDEX = 0xFFFF
DENSE_WIDTH = 32
POLICY_CLASSES = POLICY_SIZE
CHECKPOINT = "checkpoint.pt"
METRICS_JSON = "metrics.json"

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The pilot `.nnue` v1 CPU inference budget is hidden <= 512.  Larger teacher
# models are exportable for distillation/labeling but must not be deployed.
DEPLOYABLE_MAX_HIDDEN = 512

PROFILES: Dict[str, Dict[str, object]] = {
    "pilot": {"hidden": 256, "batch": 16384, "epochs": 8, "lr": 1e-3, "amp": "auto"},
    "student": {"hidden": 512, "batch": 32768, "epochs": 8, "lr": 1e-3, "amp": "auto"},
    "teacher": {"hidden": 1536, "batch": 65536, "epochs": 10, "lr": 5e-4, "amp": "bf16"},
}

# Material sanity positions: (name, fen, expected sign of side-to-move cp)
SANITY_FENS: Sequence[Tuple[str, str, int]] = (
    ("startpos", START_FEN, 0),
    ("white_up_queen_wtm", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", 1),
    ("white_up_queen_btm", "4k3/8/8/8/8/8/8/3QK3 b - - 0 1", -1),
    ("white_up_rook_wtm", "4k3/8/8/8/8/8/8/3RK3 w - - 0 1", 1),
    ("black_up_rook_btm", "4k3/8/8/8/8/8/8/3RK3 b - - 0 1", -1),
    ("kings_only", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
    ("white_queen_vs_black_rook", "4k2r/8/8/8/8/8/8/3QK3 w - - 0 1", 1),
)

PHASE_BINS = (("endgame", 0, 8), ("middlegame", 9, 16), ("opening", 17, 99))


# ---------------------------------------------------------------------------
# Errors / small helpers
# ---------------------------------------------------------------------------
class TrainError(RuntimeError):
    pass


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise TrainError(
            f"PyTorch is not installed in this interpreter ({_TORCH_IMPORT_ERROR}). "
            "Use the project virtualenv (.venv/Scripts/python.exe on Windows, "
            ".venv/bin/python on Linux) and install torch there."
        )


def _resolve_path(path: str) -> str:
    """Resolve a CLI path against the repository root (Windows and Linux)."""
    if not path:
        return path
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(obj):
    """Recursively convert numpy/torch scalars for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    feats: np.ndarray  # int64 [F] flattened active features
    offsets: np.ndarray  # int64 [B]
    target: np.ndarray  # float32 [B]
    cp: Optional[np.ndarray]  # float32 [B] or None
    moves: Optional[np.ndarray]  # int64 [B] or None
    material: np.ndarray  # int64 [B, 3] = (own, opponent, phase)
    rows: np.ndarray  # int64 [B] source row indices (for diagnostics)

    @property
    def n(self) -> int:
        return int(self.target.shape[0])


class Shards:
    """Memory-mapped access to one extractor output directory."""

    def __init__(self, root: str, policy: str = "auto"):
        self.root = os.path.abspath(root)
        if not os.path.isdir(self.root):
            raise TrainError(f"data directory not found: {self.root}")
        for name in ("features.u16", "counts.u8", "targets.f32", "groups.u32"):
            if not os.path.exists(os.path.join(self.root, name)):
                raise TrainError(f"missing shard {name} in {self.root}")

        self.counts = np.memmap(
            os.path.join(self.root, "counts.u8"), dtype=np.uint8, mode="r"
        )
        n = int(len(self.counts))
        if n == 0:
            raise TrainError(f"empty shards in {self.root}")
        feat = np.memmap(
            os.path.join(self.root, "features.u16"), dtype=np.uint16, mode="r"
        )
        if feat.size != n * FEATURE_STRIDE:
            raise TrainError(
                f"features.u16 has {feat.size} entries, expected {n * FEATURE_STRIDE}"
            )
        self.feat = feat.reshape(n, FEATURE_STRIDE)
        self.targets = np.memmap(
            os.path.join(self.root, "targets.f32"), dtype=np.float32, mode="r"
        )
        self.groups = np.memmap(
            os.path.join(self.root, "groups.u32"), dtype=np.uint32, mode="r"
        )
        if len(self.targets) != n or len(self.groups) != n:
            raise TrainError("shard lengths disagree (counts/targets/groups)")

        self.cp = None
        cp_path = os.path.join(self.root, "cp.f32")
        if os.path.exists(cp_path):
            self.cp = np.memmap(cp_path, dtype=np.float32, mode="r")
            if len(self.cp) != n:
                raise TrainError("cp.f32 length disagrees with counts.u8")

        self.moves = None
        moves_path = os.path.join(self.root, "moves.u16")
        if os.path.exists(moves_path):
            self.moves = np.memmap(moves_path, dtype=np.uint16, mode="r")
            if len(self.moves) != n:
                raise TrainError("moves.u16 length disagrees with counts.u8")
            bad = int(np.count_nonzero(np.asarray(self.moves) >= POLICY_CLASSES))
            absent = int(np.count_nonzero(np.asarray(self.moves) == IGNORE_INDEX))
            if bad - absent:
                raise TrainError(
                    f"moves.u16 has {bad - absent} indices outside 0..4351/0xFFFF"
                )

        self.pv = None
        pv_path = os.path.join(self.root, "pv.u16")
        if os.path.exists(pv_path):
            pv = np.memmap(pv_path, dtype=np.uint16, mode="r")
            if pv.size % 8 or pv.size // 8 != n:
                raise TrainError("pv.u16 is not [N, 8]")
            self.pv = pv.reshape(n, 8)

        self.meta = self._read_meta()
        self.has_cp = self.cp is not None
        if policy == "on" and self.moves is None:
            print("warning: --policy on but moves.u16 is missing; disabling policy",
                  file=sys.stderr)
            self.use_policy = False
        elif policy == "off":
            self.use_policy = False
        else:
            self.use_policy = self.moves is not None

    def _read_meta(self) -> dict:
        meta: dict = {}
        meta_path = os.path.join(self.root, "meta.json")
        txt_path = os.path.join(self.root, "meta.txt")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            elif os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    meta = {
                        k: v
                        for k, v in (
                            line.strip().split("=", 1)
                            for line in f
                            if "=" in line
                        )
                    }
        except (OSError, ValueError, json.JSONDecodeError):
            meta = {}
        return meta

    def __len__(self) -> int:
        return int(len(self.counts))

    def validate(self, n: int, sample: int = 65536) -> Dict[str, int]:
        """Drop rows with impossible counts/targets; return validation stats."""
        counts = np.asarray(self.counts[:n])
        targets = np.asarray(self.targets[:n])
        bad_count = (counts == 0) | (counts > FEATURE_STRIDE)
        bad_target = ~np.isfinite(targets) | (targets < 0.0) | (targets > 1.0)
        bad = bad_count | bad_target
        # Feature indices must stay inside the input space.  Check a sample to
        # keep startup cheap on multi-million-row shards.
        step = max(1, n // max(1, sample))
        rows = np.arange(0, n, step)
        feat_sample = np.asarray(self.feat[rows])
        cnt_sample = counts[rows].astype(np.int64)
        mask = np.arange(FEATURE_STRIDE)[None, :] < cnt_sample[:, None]
        bad_idx = int(
            np.count_nonzero((feat_sample.astype(np.int64) >= INPUTS) & mask)
        )
        if bad_idx:
            raise TrainError(
                f"{bad_idx} feature indices >= {INPUTS} in sampled rows; "
                "corrupt or incompatible shard"
            )
        self.valid_mask = ~bad
        return {
            "rows": int(n),
            "dropped_bad_count": int(np.count_nonzero(bad_count)),
            "dropped_bad_target": int(np.count_nonzero(bad_target)),
            "dropped_total": int(np.count_nonzero(bad)),
            "usable": int(n - np.count_nonzero(bad)),
        }

    def batch(self, rows: np.ndarray) -> Batch:
        rows = np.asarray(rows, dtype=np.int64)
        f = np.asarray(self.feat[rows])  # [B, 32] uint16
        c = np.asarray(self.counts[rows]).astype(np.int64)
        mask = np.arange(FEATURE_STRIDE, dtype=np.int64)[None, :] < c[:, None]
        flat = f[mask].astype(np.int64)
        offsets = np.zeros(len(rows), dtype=np.int64)
        if len(rows) > 1:
            offsets[1:] = np.cumsum(c[:-1])

        target = np.asarray(self.targets[rows], dtype=np.float32)
        cp = np.asarray(self.cp[rows], dtype=np.float32) if self.cp is not None else None
        moves = (
            np.asarray(self.moves[rows], dtype=np.int64)
            if self.use_policy
            else None
        )

        rem = f.astype(np.int64) % 768
        rel = rem // 384
        pt = (rem % 384) // 64
        vals = MATERIAL[pt] * mask
        own = np.where(mask & (rel == 0), vals, 0).sum(axis=1)
        opp = np.where(mask & (rel == 1), vals, 0).sum(axis=1)
        phase = np.where(mask, PHASE_INC[pt], 0).sum(axis=1)
        material = np.stack([own, opp, phase], axis=1).astype(np.int64)

        return Batch(flat, offsets, target, cp, moves, material, rows)

    def iter_batches(self, indices, batch_size, shuffle, seed):
        yield from _iter_batches(self, indices, batch_size, shuffle, seed)

    def train_loader(self, indices, batch_size, seed, args):
        if args.prefetch <= 0:
            return self.iter_batches(indices, batch_size, True, seed)
        return Prefetcher(
            self,
            indices,
            batch_size,
            True,
            seed,
            depth=args.prefetch,
            workers=args.num_workers,
        )


def _iter_batches(
    shards: Shards,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[Batch]:
    order = np.asarray(indices, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = order[rng.permutation(len(order))]
    for start in range(0, len(order), batch_size):
        yield shards.batch(order[start : start + batch_size])


class Prefetcher:
    """Ordered parallel prefetch of memmap batches.

    ``workers`` reader threads build batches and at most ``depth`` batches are
    buffered ahead.  Results are yielded in shuffle order, so training stays
    deterministic regardless of thread count.
    """

    def __init__(
        self, shards, indices, batch_size, shuffle, seed, depth=2, workers=0
    ):
        self.shards = shards
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.depth = max(1, int(depth))
        self.workers = max(1, int(workers) if workers else 1)

    def __iter__(self):
        order = self.indices
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            order = order[rng.permutation(len(order))]
        n = len(order)
        if n == 0:
            return
        n_chunks = int(math.ceil(n / self.batch_size))
        cond = threading.Condition()
        state = {"claim": 0, "yielded": 0, "stop": False}
        results: Dict[int, Batch] = {}
        errors: List[BaseException] = []

        def worker():
            while True:
                with cond:
                    while (
                        not state["stop"]
                        and state["claim"] < n_chunks
                        and state["claim"] >= state["yielded"] + self.depth
                    ):
                        cond.wait()
                    if state["stop"] or state["claim"] >= n_chunks:
                        return
                    i = state["claim"]
                    state["claim"] += 1
                try:
                    b = self.shards.batch(
                        order[i * self.batch_size : (i + 1) * self.batch_size]
                    )
                except BaseException as exc:  # surface in the consumer
                    with cond:
                        errors.append(exc)
                        state["stop"] = True
                        cond.notify_all()
                    return
                with cond:
                    if state["stop"]:
                        return
                    results[i] = b
                    cond.notify_all()

        threads = [
            threading.Thread(target=worker, daemon=True, name=f"prefetch-{i}")
            for i in range(self.workers)
        ]
        for t in threads:
            t.start()
        try:
            for i in range(n_chunks):
                with cond:
                    while i not in results and not errors:
                        cond.wait()
                    if errors:
                        raise errors[0]
                    b = results.pop(i)
                    state["yielded"] = i + 1
                    cond.notify_all()
                yield b
        finally:
            with cond:
                state["stop"] = True
                cond.notify_all()
            for t in threads:
                t.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Device-resident batches
# ---------------------------------------------------------------------------
@dataclass
class DeviceBatch:
    feats: "torch.Tensor"
    offsets: "torch.Tensor"
    target: "torch.Tensor"
    cp: Optional["torch.Tensor"]
    moves: Optional["torch.Tensor"]
    material: "torch.Tensor"

    @property
    def n(self) -> int:
        return int(self.target.shape[0])


def _to_device_batch(batch: Batch, device, pin: bool) -> DeviceBatch:
    def move(a, dtype=None):
        t = torch.from_numpy(np.ascontiguousarray(a))
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if pin and device.type == "cuda":
            t = t.pin_memory()
        return t.to(device, non_blocking=pin)

    return DeviceBatch(
        feats=move(batch.feats, torch.int64),
        offsets=move(batch.offsets, torch.int64),
        target=move(batch.target, torch.float32),
        cp=move(batch.cp, torch.float32) if batch.cp is not None else None,
        moves=move(batch.moves, torch.int64) if batch.moves is not None else None,
        material=move(batch.material, torch.int64),
    )


class GpuShards:
    """All shards uploaded once to GPU memory, batches gathered on-device.

    For 100M positions the feature matrix is ~12.8 GB as int32 plus ~0.9 GB of
    labels, comfortably inside a 90 GB A6000 but not a 16 GB deployment GPU.
    The memmap path remains the default and the fallback.
    """

    def __init__(self, shards: Shards, device, rows: int):
        if device.type != "cuda":
            raise TrainError("--data-on-gpu requires a CUDA device")
        n = int(rows)
        if n < 2:
            raise TrainError("--data-on-gpu needs at least 2 usable rows")
        self.device = device
        self.n = n
        self.use_policy = shards.use_policy

        estimate = n * (FEATURE_STRIDE * 4 + 1 + 4)
        if shards.cp is not None:
            estimate += n * 4
        if shards.use_policy:
            estimate += n * 4
        total = torch.cuda.get_device_properties(device).total_memory
        if estimate > 0.8 * total:
            raise TrainError(
                f"--data-on-gpu needs ~{estimate / 2**30:.1f} GiB "
                f"({n:,} rows) but the GPU has {total / 2**30:.1f} GiB. "
                "Use --max-samples, a smaller shard set, or the memmap path."
            )

        self.feat = self._upload_2d(shards.feat, n, torch.int32)
        self.counts = self._upload_1d(shards.counts, n, torch.uint8)
        self.targets = self._upload_1d(shards.targets, n, torch.float32)
        self.cp = (
            self._upload_1d(shards.cp, n, torch.float32)
            if shards.cp is not None
            else None
        )
        self.moves = (
            self._upload_1d(shards.moves, n, torch.int32)
            if shards.use_policy
            else None
        )
        self._arange = torch.arange(FEATURE_STRIDE, device=device)
        self._mat = torch.tensor(MATERIAL, dtype=torch.int64, device=device)
        self._phase = torch.tensor(PHASE_INC, dtype=torch.int64, device=device)
        self.bytes_used = estimate

    # -- upload ----------------------------------------------------------
    def _upload_2d(self, arr, n: int, dtype, chunk: int = 2_000_000):
        out = torch.empty((n, FEATURE_STRIDE), dtype=dtype, device=self.device)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            block = torch.from_numpy(np.array(arr[start:end], copy=True)).to(dtype)
            out[start:end].copy_(block)
        return out

    def _upload_1d(self, arr, n: int, dtype, chunk: int = 16_000_000):
        out = torch.empty(n, dtype=dtype, device=self.device)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            block = torch.from_numpy(np.array(arr[start:end], copy=True)).to(dtype)
            out[start:end].copy_(block)
        return out

    # -- batching --------------------------------------------------------
    def batch(self, rows) -> DeviceBatch:
        rows_t = torch.as_tensor(
            np.asarray(rows, dtype=np.int64), dtype=torch.long, device=self.device
        )
        f = self.feat.index_select(0, rows_t)  # [B,32] int32
        c = self.counts.index_select(0, rows_t).to(torch.int64)
        mask = self._arange[None, :] < c[:, None]
        flat = f[mask].to(torch.int64)
        offsets = torch.zeros(len(rows_t), dtype=torch.int64, device=self.device)
        if len(rows_t) > 1:
            offsets[1:] = torch.cumsum(c, dim=0)[:-1]

        target = self.targets.index_select(0, rows_t)
        cp = self.cp.index_select(0, rows_t) if self.cp is not None else None
        moves = (
            self.moves.index_select(0, rows_t).to(torch.int64)
            if self.moves is not None
            else None
        )

        rem = f % 768
        rel = rem // 384
        pt = (rem % 384) // 64
        vals = self._mat[pt] * mask
        own = torch.where(mask & (rel == 0), vals, torch.zeros_like(vals)).sum(1)
        opp = torch.where(mask & (rel == 1), vals, torch.zeros_like(vals)).sum(1)
        phase = torch.where(mask, self._phase[pt], torch.zeros_like(vals)).sum(1)
        material = torch.stack([own, opp, phase], dim=1)
        return DeviceBatch(flat, offsets, target, cp, moves, material)

    def iter_batches(self, indices, batch_size, shuffle, seed):
        order = np.asarray(indices, dtype=np.int64)
        if shuffle:
            rng = np.random.default_rng(seed)
            order = order[rng.permutation(len(order))]
        for start in range(0, len(order), batch_size):
            yield self.batch(order[start : start + batch_size])

    def train_loader(self, indices, batch_size, seed, args):
        return self.iter_batches(indices, batch_size, True, seed)


# ---------------------------------------------------------------------------
# Group-aware split
# ---------------------------------------------------------------------------
def group_split(
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
    max_samples: int = 0,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Split position indices by group id, never by position hash.

    Groups are sorted by ``np.unique`` (deterministic), shuffled with
    ``default_rng(seed)`` and the first ``round(val_fraction * n_groups)``
    groups go to validation.  Every position of a group stays on one side.
    """
    groups = np.asarray(groups)
    n = len(groups)
    if not 0.0 < val_fraction < 1.0:
        raise TrainError(f"val_fraction must be in (0,1), got {val_fraction}")
    limit = n if max_samples <= 0 else min(n, int(max_samples))
    if limit < 2:
        raise TrainError("need at least 2 samples to split")

    if valid_mask is not None:
        keep = np.flatnonzero(np.asarray(valid_mask)[:limit])
    else:
        keep = np.arange(limit, dtype=np.int64)
    if len(keep) < 2:
        raise TrainError("fewer than 2 usable samples after validation")

    g = groups[keep]
    uniq, inverse = np.unique(g, return_inverse=True)
    if len(uniq) < 2:
        raise TrainError(
            "all samples share one group; cannot make a group-disjoint split"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val_groups = int(round(len(uniq) * val_fraction))
    n_val_groups = min(max(n_val_groups, 1), len(uniq) - 1)
    is_val_group = np.zeros(len(uniq), dtype=bool)
    is_val_group[perm[:n_val_groups]] = True
    mask = is_val_group[inverse]

    train = keep[~mask]
    val = keep[mask]
    if len(train) == 0 or len(val) == 0:  # pragma: no cover - guarded above
        raise TrainError("degenerate group split produced an empty side")

    info = {
        "method": "unique_groups_shuffle",
        "seed": int(seed),
        "val_fraction_requested": float(val_fraction),
        "val_fraction_actual": float(len(val) / (len(train) + len(val))),
        "samples_considered": int(limit),
        "train_samples": int(len(train)),
        "val_samples": int(len(val)),
        "total_groups": int(len(uniq)),
        "train_groups": int(len(uniq) - n_val_groups),
        "val_groups": int(n_val_groups),
        "group_overlap": 0,
        "group_uniqueness_note": (
            "groups.u32 from extract_halfkp is an FNV-1a hash of the FEN; if the "
            "extractor does not cluster by game, this becomes a random position "
            "split. Verify meta.json/grouping before trusting it as a game split."
        ),
    }
    return train.astype(np.int64), val.astype(np.int64), info


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class HalfKPNet(nn.Module):
    def __init__(
        self,
        hidden: int,
        policy_size: int = 0,
        dropout: float = 0.0,
        sparse: bool = True,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden
        self.policy_size = policy_size
        self.emb = nn.EmbeddingBag(INPUTS, hidden, mode="sum", sparse=sparse)
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.fc1 = nn.Linear(hidden, DENSE_WIDTH)
        self.fc2 = nn.Linear(DENSE_WIDTH, DENSE_WIDTH)
        self.out = nn.Linear(DENSE_WIDTH, 1)
        self.policy = nn.Linear(hidden, policy_size) if policy_size > 0 else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

        nn.init.normal_(self.emb.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.ft_bias)
        nn.init.zeros_(self.out.bias)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
        if self.policy is not None:
            nn.init.normal_(self.policy.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.policy.bias)

    def hidden(self, feats, offsets):
        h = self.emb(feats, offsets) + self.ft_bias
        return torch.relu(h)

    def value(self, h):
        if self.drop is not None:
            h = self.drop(h)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.out(h).squeeze(1)

    def policy_logits(self, h):
        return self.policy(h) if self.policy is not None else None

    def forward(self, feats, offsets):
        return self.value(self.hidden(feats, offsets))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: HalfKPNet,
    data_source,
    indices: np.ndarray,
    device,
    batch_size: int,
    max_policy_samples: int = 8192,
    calibration_bins: int = 10,
) -> Dict[str, object]:
    """Compute value metrics, calibration, per-phase bins and policy accuracy.

    ``data_source`` is a :class:`Shards` (memmap) or :class:`GpuShards`; both
    expose ``iter_batches`` and ``use_policy``.
    """
    model.eval()
    n = 0
    bce_sum = 0.0
    brier_sum = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    correct = 0
    decisive = 0

    bins_count = np.zeros(calibration_bins)
    bins_pred = np.zeros(calibration_bins)
    bins_true = np.zeros(calibration_bins)

    phase_stats: Dict[str, List[float]] = {}
    pol_n = 0
    pol_top1 = 0
    pol_top3 = 0

    for raw in data_source.iter_batches(indices, batch_size, False, 0):
        batch = (
            raw
            if isinstance(raw, DeviceBatch)
            else _to_device_batch(raw, device, pin=False)
        )
        target = batch.target
        h = model.hidden(batch.feats, batch.offsets)
        logits = model.value(h)

        loss_per = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        t = target.detach().cpu().numpy().astype(np.float64)
        material = batch.material.detach().cpu().numpy()

        b = len(t)
        n += b
        bce_sum += float(loss_per.sum())
        brier_sum += float(((p - t) ** 2).sum())
        pred_sum += float(p.sum())
        target_sum += float(t.sum())
        dmask = t != 0.5
        if dmask.any():
            correct += int(np.count_nonzero((p[dmask] > 0.5) == (t[dmask] > 0.5)))
            decisive += int(np.count_nonzero(dmask))

        bin_idx = np.minimum(
            (p * calibration_bins).astype(np.int64), calibration_bins - 1
        )
        bins_count += np.bincount(bin_idx, minlength=calibration_bins)
        bins_pred += np.bincount(bin_idx, weights=p, minlength=calibration_bins)
        bins_true += np.bincount(bin_idx, weights=t, minlength=calibration_bins)

        phase = material[:, 2]
        for name, lo, hi in PHASE_BINS:
            sel = (phase >= lo) & (phase <= hi)
            if not sel.any():
                continue
            st = phase_stats.setdefault(name, [0.0, 0.0, 0.0, 0.0, 0.0])
            st[0] += float(sel.sum())
            st[1] += float(loss_per[sel].sum())
            st[2] += float(((p[sel] - t[sel]) ** 2).sum())
            dsel = sel & dmask
            if dsel.any():
                st[3] += float(
                    np.count_nonzero((p[dsel] > 0.5) == (t[dsel] > 0.5))
                )
                st[4] += float(dsel.sum())

        if (
            data_source.use_policy
            and model.policy is not None
            and pol_n < max_policy_samples
            and batch.moves is not None
        ):
            valid = batch.moves != IGNORE_INDEX
            if bool(valid.any()):
                plogits = model.policy_logits(h[valid])
                k = min(3, plogits.shape[1])
                topk = torch.topk(plogits, k=k, dim=1).indices.cpu().numpy()
                mv = batch.moves[valid].detach().cpu().numpy()
                pol_top1 += int(np.count_nonzero(topk[:, 0] == mv))
                pol_top3 += int(
                    np.count_nonzero(np.any(topk[:, :k] == mv[:, None], axis=1))
                )
                pol_n += int(len(mv))

    if n == 0:
        raise TrainError("empty validation set")

    calib: List[dict] = []
    ece = 0.0
    total = max(float(bins_count.sum()), 1.0)
    for i in range(calibration_bins):
        c = float(bins_count[i])
        if c == 0:
            calib.append(
                {
                    "bin": f"{i / calibration_bins:.1f}-{(i + 1) / calibration_bins:.1f}",
                    "count": 0,
                    "mean_pred": None,
                    "mean_target": None,
                }
            )
        else:
            mp = float(bins_pred[i] / c)
            mt = float(bins_true[i] / c)
            ece += abs(mp - mt) * c / total
            calib.append(
                {
                    "bin": f"{i / calibration_bins:.1f}-{(i + 1) / calibration_bins:.1f}",
                    "count": int(c),
                    "mean_pred": mp,
                    "mean_target": mt,
                }
            )

    phases = {}
    for name, st in phase_stats.items():
        cnt = max(st[0], 1.0)
        phases[name] = {
            "n": int(st[0]),
            "loss": st[1] / cnt,
            "brier": st[2] / cnt,
            "acc_decisive": (st[3] / st[4]) if st[4] else None,
        }

    result: Dict[str, object] = {
        "n": n,
        "loss": bce_sum / n,
        "brier": brier_sum / n,
        "logloss": bce_sum / n,
        "acc_decisive": (correct / decisive) if decisive else None,
        "mean_pred": pred_sum / n,
        "mean_target": target_sum / n,
        "ece": ece,
        "calibration": calib,
        "per_phase": phases,
    }
    if data_source.use_policy:
        result["policy"] = {
            "n": pol_n,
            "top1": (pol_top1 / pol_n) if pol_n else None,
            "top3": (pol_top3 / pol_n) if pol_n else None,
        }
    return result


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------
def _lr_scale(step: int, total_steps: int, warmup: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(optimizers, lr: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = lr


def _clip_sparse_grads(params, max_norm: float) -> None:
    if max_norm <= 0:
        return
    with torch.no_grad():
        for p in params:
            g = p.grad
            if g is None or not g.is_sparse:
                continue
            values = g._values()
            norms = values.norm(dim=1, keepdim=True).clamp_min(1e-12)
            values.mul_((max_norm / norms).clamp(max=1.0))


def _make_optimizers(model: HalfKPNet, args, lr: float):
    emb_params = [model.emb.weight]
    dense_decay = []
    dense_no_decay = []
    for name, p in model.named_parameters():
        if name == "emb.weight":
            continue
        if p.ndim >= 2:
            dense_decay.append(p)
        else:
            dense_no_decay.append(p)

    opt_sparse = None
    if model.emb.sparse:
        opt_sparse = torch.optim.SparseAdam(emb_params, lr=lr)
        dense_groups = [
            {"params": dense_decay, "weight_decay": args.weight_decay},
            {"params": dense_no_decay, "weight_decay": 0.0},
        ]
    else:
        dense_groups = [
            {"params": emb_params + dense_decay, "weight_decay": args.weight_decay},
            {"params": dense_no_decay, "weight_decay": 0.0},
        ]
    opt_dense = torch.optim.AdamW(dense_groups, lr=lr)
    return opt_sparse, opt_dense


def _setup_amp(device, mode: str):
    if device.type != "cuda" or mode == "off":
        return None, None
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mode == "bf16":
        return torch.bfloat16, None
    if mode == "fp16":
        return torch.float16, torch.amp.GradScaler("cuda")
    raise TrainError(f"unknown amp mode {mode}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _state_arrays(model: HalfKPNet):
    sd = model.state_dict()
    arrays = {
        "ft": sd["emb.weight"].detach().cpu().numpy().astype(np.float32),
        "ft_bias": sd["ft_bias"].detach().cpu().numpy().astype(np.float32),
        # Torch stores Linear weights as [out, in]; the file is input-major.
        "fc1_w": sd["fc1.weight"].detach().cpu().numpy().T.astype(np.float32),
        "fc1_b": sd["fc1.bias"].detach().cpu().numpy().astype(np.float32),
        "fc2_w": sd["fc2.weight"].detach().cpu().numpy().T.astype(np.float32),
        "fc2_b": sd["fc2.bias"].detach().cpu().numpy().astype(np.float32),
        "out_w": sd["out.weight"].detach().cpu().numpy().reshape(-1).astype(np.float32),
        "out_b": sd["out.bias"].detach().cpu().numpy().reshape(-1).astype(np.float32),
    }
    policy_w = policy_b = None
    if model.policy is not None:
        policy_w = (
            sd["policy.weight"].detach().cpu().numpy().T.astype(np.float32)
        )  # [hidden, policy_size]
        policy_b = sd["policy.bias"].detach().cpu().numpy().astype(np.float32)
    return arrays, policy_w, policy_b


@torch.no_grad()
def _torch_cp(model: HalfKPNet, batch: Batch, device) -> np.ndarray:
    feats = torch.from_numpy(batch.feats).to(device)
    offsets = torch.from_numpy(batch.offsets).to(device)
    return (model(feats, offsets) * CP_SCALE).cpu().numpy().astype(np.float64)


def export_check(
    model: HalfKPNet,
    path: str,
    shards: Shards,
    indices: np.ndarray,
    device,
    n: int = 64,
    verify_checksum: bool = False,
) -> Dict[str, object]:
    """Compare the exported file's cp against the in-memory torch model.

    ``verify_checksum`` defaults to False because the exporter has already
    verified the freshly written file once in ``write_nnue(verify=True)``; the
    FNV pass over a 100+ MB teacher payload is not free.
    """
    batch = shards.batch(np.asarray(indices[:n], dtype=np.int64))
    torch_cp = _torch_cp(model, batch, device)
    ref = load_nnue(path, verify_checksum=verify_checksum)
    rows = []
    diffs = []
    for i in range(batch.n):
        off = int(batch.offsets[i])
        cnt = (
            int(batch.offsets[i + 1]) - off
            if i + 1 < batch.n
            else len(batch.feats) - off
        )
        ref_cp = ref.forward_indices(batch.feats[off : off + cnt])
        diffs.append(abs(ref_cp - float(torch_cp[i])))
        rows.append((off, cnt))
    diffs_arr = np.asarray(diffs, dtype=np.float64)

    result: Dict[str, object] = {
        "samples": int(batch.n),
        "max_abs_cp_diff": float(diffs_arr.max()) if diffs_arr.size else 0.0,
        "mean_abs_cp_diff": float(diffs_arr.mean()) if diffs_arr.size else 0.0,
        "tolerance_cp": 2.0,
        "ok": bool(diffs_arr.size == 0 or diffs_arr.max() <= 2.0),
    }

    if model.policy is not None and ref.policy_w is not None and rows:
        feats_t = torch.from_numpy(batch.feats).to(device)
        offsets_t = torch.from_numpy(batch.offsets).to(device)
        torch_plog = model.policy_logits(
            model.hidden(feats_t, offsets_t)
        ).detach().cpu().numpy()
        pol_diffs = []
        for i, (off, cnt) in enumerate(rows[: min(4, len(rows))]):
            ref_plog = ref.policy_logits(batch.feats[off : off + cnt])
            if ref_plog is not None:
                pol_diffs.append(float(np.max(np.abs(ref_plog - torch_plog[i]))))
        if pol_diffs:
            result["max_abs_policy_logit_diff"] = float(max(pol_diffs))
            result["policy_topk_check"] = bool(
                all(
                    np.array_equal(
                        np.argsort(-ref.policy_logits(batch.feats[off : off + cnt]))[:3],
                        np.argsort(-torch_plog[i])[:3],
                    )
                    for i, (off, cnt) in enumerate(rows[: min(4, len(rows))])
                )
            )
    return result


def material_sanity(path: str, verify_checksum: bool = False) -> List[dict]:
    ref = load_nnue(path, verify_checksum=verify_checksum)
    out = []
    for name, fen, expected in SANITY_FENS:
        try:
            r = ref.eval_fen(fen)
            cp = r["cp"]
            sign = 0 if abs(cp) < 25.0 else (1 if cp > 0 else -1)
            # An untrained/early model may be flat; only a *wrong* sign is a
            # real material-sanity failure.  Near-zero is reported as inconclusive.
            inconclusive = expected != 0 and abs(cp) < 25.0
            out.append(
                {
                    "name": name,
                    "fen": fen,
                    "cp": cp,
                    "wdl": r["wdl"],
                    "expected_sign": expected,
                    "actual_sign": sign,
                    "inconclusive": bool(inconclusive),
                    "sign_ok": bool(expected == 0 or inconclusive or sign == expected),
                    "n_features": r["n_features"],
                }
            )
        except ValueError as exc:
            out.append({"name": name, "fen": fen, "error": str(exc)})
    return out


def export_model(
    model: HalfKPNet,
    args,
    out_dir: str,
    name: str,
    card_extra: dict,
    shards: Optional[Shards] = None,
    val_indices: Optional[np.ndarray] = None,
    device=None,
) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    arrays, policy_w, policy_b = _state_arrays(model)
    nnue_path = os.path.join(out_dir, f"{name}.nnue")
    info = export_nnue.write_nnue(
        nnue_path,
        hidden=model.hidden_size,
        ft=arrays["ft"],
        ft_bias=arrays["ft_bias"],
        fc1_w=arrays["fc1_w"],
        fc1_b=arrays["fc1_b"],
        fc2_w=arrays["fc2_w"],
        fc2_b=arrays["fc2_b"],
        out_w=arrays["out_w"],
        out_b=arrays["out_b"],
        policy_w=policy_w,
        policy_b=policy_b,
        cp_scale=CP_SCALE,
        verify=True,
        verify_samples=24,
        seed=args.seed,
    )

    checks: Dict[str, object] = {"roundtrip": info.pop("roundtrip", None)}
    if shards is not None and val_indices is not None and len(val_indices):
        checks["export_vs_torch"] = export_check(
            model, nnue_path, shards, val_indices, device
        )
    checks["material_sanity"] = material_sanity(nnue_path)

    torch.save(model.state_dict(), os.path.join(out_dir, f"{name}.pt"))
    if model.hidden_size <= DEPLOYABLE_MAX_HIDDEN:
        # Legacy convenience artifact for the deployable sizes only; skip the
        # extra 100+ MB copy for teacher models.
        np.savez_compressed(
            os.path.join(out_dir, f"{name}.npz"),
            ft=np.rint(arrays["ft"] / info["ft_scale"]).clip(-32767, 32767).astype(np.int16),
            ft_scale=np.float32(info["ft_scale"]),
            ft_bias=arrays["ft_bias"],
            fc1_w=arrays["fc1_w"],
            fc1_b=arrays["fc1_b"],
            fc2_w=arrays["fc2_w"],
            fc2_b=arrays["fc2_b"],
            out_w=arrays["out_w"] * np.float32(CP_SCALE),
            out_b=arrays["out_b"] * np.float32(CP_SCALE),
            inputs=np.int32(INPUTS),
            hidden=np.int32(model.hidden_size),
        )

    role = str(getattr(args, "profile", "pilot") or "pilot")
    deployable = model.hidden_size <= DEPLOYABLE_MAX_HIDDEN
    reason = None
    if not deployable:
        reason = (
            f"hidden={model.hidden_size} exceeds the pilot CPU inference budget "
            f"(<= {DEPLOYABLE_MAX_HIDDEN}); this is a teacher/analysis artifact, "
            "not deployable in the engine. Export a student (<=512) for play."
        )
        print(
            f"WARNING: hidden={model.hidden_size} > {DEPLOYABLE_MAX_HIDDEN}; "
            "model_card.json marks this file non-deployable (teacher only)",
            flush=True,
        )

    card = {
        "format": "nnue-v1-halfkp",
        "created_utc": _now(),
        "model_file": os.path.basename(nnue_path),
        "model_path": nnue_path,
        "hidden": model.hidden_size,
        "role": role,
        "deployable": deployable,
        "deployability": {
            "deployable": deployable,
            "role": role,
            "max_deployable_hidden": DEPLOYABLE_MAX_HIDDEN,
            "reason": reason,
            "student_export_is_deployable": True,
        },
        "export": info,
        "output": {
            "units": "centipawns_side_to_move",
            "cp_scale": CP_SCALE,
            "wdl": "sigmoid(cp / cp_scale)",
            "note": "out_w/out_b are pre-multiplied by cp_scale at export time",
        },
        "checks": checks,
        "environment": {
            "torch": getattr(torch, "__version__", None),
            "cuda": torch.version.cuda if torch is not None else None,
            "device": str(device) if device is not None else None,
        },
    }
    card.update(card_extra)
    export_nnue.write_model_card(out_dir, _jsonable(card))
    return {"nnue": nnue_path, "info": info, "checks": checks, "card": card}


# ---------------------------------------------------------------------------
# Synthetic smoke shards (no real dataset required)
# ---------------------------------------------------------------------------
def make_smoke_shards(
    out_dir: str,
    n: int = 2048,
    seed: int = 7,
    synthetic: bool = True,
) -> str:
    """Write tiny but structurally valid shards for a CPU smoke test.

    Positions come from short random-chess playouts, features come from
    ``reference_inference.fen_to_halfkp_indices`` (the same contract the
    extractor implements), and labels are a noisy material heuristic.  Groups
    are game ids so the group split is exercised.  ``moves.u16`` is a random
    legal move; ``cp.f32`` is the noisy label before the sigmoid.
    """
    try:
        import chess
    except ImportError as exc:  # pragma: no cover
        raise TrainError(f"python-chess is required for --smoke: {exc}") from exc

    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
              chess.QUEEN: 9, chess.KING: 0}

    feats_rows: List[np.ndarray] = []
    counts: List[int] = []
    targets: List[float] = []
    cps: List[float] = []
    moves: List[int] = []
    pvs: List[np.ndarray] = []
    groups: List[int] = []

    def emit(board, gid: int) -> bool:
        if board.is_game_over():
            return False
        idx = fen_to_halfkp_indices(board.fen(), "w" if board.turn else "b")
        if idx.size == 0 or idx.size > FEATURE_STRIDE:
            return False

        white_mat = sum(
            len(board.pieces(pt, chess.WHITE)) * val for pt, val in values.items()
        )
        black_mat = sum(
            len(board.pieces(pt, chess.BLACK)) * val for pt, val in values.items()
        )
        stm = 1 if board.turn == chess.WHITE else -1
        cp = float(stm * (white_mat - black_mat) * 90.0 + rng.normal(0.0, 30.0))
        cp = float(np.clip(cp, -1500.0, 1500.0))
        target = 1.0 / (1.0 + math.exp(-cp / CP_SCALE))

        move = rng.choice(list(board.legal_moves))
        if move.promotion:
            mv = 4096 + (int(move.promotion) - 2) * 64 + move.from_square
        else:
            mv = move.from_square * 64 + move.to_square

        pv = np.full(8, IGNORE_INDEX, dtype=np.uint16)
        pv[0] = mv

        padded = np.zeros(FEATURE_STRIDE, dtype=np.uint16)
        padded[: idx.size] = idx.astype(np.uint16)
        feats_rows.append(padded)
        counts.append(int(idx.size))
        targets.append(float(np.clip(target, 0.0, 1.0)))
        cps.append(cp)
        moves.append(int(mv))
        pvs.append(pv)
        groups.append(gid)
        return True

    game = 0
    attempts = 0
    while len(targets) < n:
        attempts += 1
        if attempts > max(100, n * 20):
            raise TrainError("could not generate enough smoke positions")
        board = chess.Board()
        plies = int(rng.integers(10, 80))
        sampled = 0
        for ply in range(plies):
            if board.is_game_over():
                break
            # A few positions per playout share the group id, which exercises
            # the group-aware split with non-singleton groups.  Sampling starts
            # after the opening so material imbalances actually occur.
            if ply >= 8 and sampled < 3 and rng.random() < 0.45:
                if emit(board.copy(), game):
                    sampled += 1
                    if len(targets) >= n:
                        break
            board.push(rng.choice(list(board.legal_moves)))
        if sampled == 0:
            emit(board, game)
        game += 1

    np.asarray(feats_rows, dtype=np.uint16).tofile(os.path.join(out_dir, "features.u16"))
    np.asarray(counts, dtype=np.uint8).tofile(os.path.join(out_dir, "counts.u8"))
    np.asarray(targets, dtype=np.float32).tofile(os.path.join(out_dir, "targets.f32"))
    np.asarray(cps, dtype=np.float32).tofile(os.path.join(out_dir, "cp.f32"))
    np.asarray(moves, dtype=np.uint16).tofile(os.path.join(out_dir, "moves.u16"))
    np.asarray(pvs, dtype=np.uint16).tofile(os.path.join(out_dir, "pv.u16"))
    np.asarray(groups, dtype=np.uint32).tofile(os.path.join(out_dir, "groups.u32"))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "format": "halfkp-v1",
                "inputs": INPUTS,
                "max_active": FEATURE_STRIDE,
                "count": len(targets),
                "target": "win_probability",
                "synthetic": bool(synthetic),
                "seed": int(seed),
                "note": "synthetic smoke shards; not training data",
            },
            f,
            indent=2,
        )
    return out_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train HalfKP value/policy net and export .nnue v1"
    )
    ap.add_argument(
        "--profile", choices=sorted(PROFILES), default="pilot",
        help="sizing defaults: pilot (256/16k), student (512/32k), "
             "teacher (1536/64k, non-deployable)",
    )
    ap.add_argument("--data", default="", help="shard directory (extract_halfkp output)")
    ap.add_argument("--out", default="training/work/model", help="output directory")
    ap.add_argument("--name", default="model", help="output basename (model.nnue)")
    ap.add_argument("--hidden", type=int, default=None,
                    help="FT width; profile default (128/256/384/512/1024/1536)")
    ap.add_argument("--batch", type=int, default=None, help="micro-batch size")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--amp", choices=["auto", "off", "bf16", "fp16"], default=None)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation micro-batches per optimizer step")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="parallel memmap reader threads (0=prefetch only)")
    ap.add_argument("--pin-memory", action="store_true",
                    help="pin host batch memory for non_blocking H2D copies")
    ap.add_argument("--data-on-gpu", action="store_true",
                    help="upload all shards to GPU memory and batch on-device")
    ap.add_argument("--min-lr-ratio", type=float, default=0.05)
    ap.add_argument("--lr-schedule", choices=["cosine", "step", "constant"], default="cosine")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--step-gamma", type=float, default=0.5)
    ap.add_argument("--step-every", type=int, default=3, help="epochs per step decay")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="dense grad-norm clip, 0=off")
    ap.add_argument("--sparse-grad-clip", type=float, default=0.0,
                    help="per-row SparseAdam grad clip, 0=off")
    ap.add_argument("--cp-weight", type=float, default=0.0,
                    help="weight of the auxiliary cp smooth-L1 loss (0=off)")
    ap.add_argument("--cp-beta", type=float, default=0.25,
                    help="Huber beta in logit units (cp/400 units)")
    ap.add_argument("--policy", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--policy-weight", type=float, default=0.2)
    ap.add_argument("--max-samples", type=int, default=0, help="use the first N rows (0=all)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--max-val-samples", type=int, default=0,
                    help="cap validation rows for per-epoch metrics (0=all)")
    ap.add_argument("--policy-eval-samples", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--prefetch", type=int, default=2, help="batch buffer depth (0=off)")
    ap.add_argument("--dense-emb", action="store_true",
                    help="dense EmbeddingBag + Adam (slow fallback, full VRAM)")
    ap.add_argument("--resume", default="", help="checkpoint.pt or its directory")
    ap.add_argument("--export-only", action="store_true",
                    help="with --resume: re-export the checkpoint without training")
    ap.add_argument("--patience", type=int, default=3, help="early-stop epochs (0=off)")
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--smoke", nargs="?", type=int, const=2048, default=0,
                    help="train on synthetic shards (optional row count)")
    return ap


def _apply_profile(args) -> None:
    """Fill unset sizing flags from ``--profile``; explicit flags win."""
    profile = PROFILES.get(args.profile)
    if profile is None:
        raise TrainError(f"unknown profile {args.profile!r}")
    for key, value in profile.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.hidden is None or args.hidden <= 0 or args.hidden % 32:
        raise TrainError(f"--hidden must be a positive multiple of 32, got {args.hidden}")
    if args.hidden not in (128, 256, 384, 512, 1024, 1536):
        print(
            f"note: hidden={args.hidden} is outside the pilot set "
            "(128/256/384/512/1024/1536)",
            flush=True,
        )
    if args.hidden > DEPLOYABLE_MAX_HIDDEN and not args.smoke:
        print(
            f"note: hidden={args.hidden} > {DEPLOYABLE_MAX_HIDDEN}: the exported "
            ".nnue will be marked non-deployable (teacher only)",
            flush=True,
        )
    if args.grad_accum < 1:
        raise TrainError("--grad-accum must be >= 1")
    if args.num_workers < 0:
        raise TrainError("--num-workers must be >= 0")


def _resolve_device(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise TrainError("--device cuda but torch.cuda.is_available() is False")
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def train(args) -> int:
    _require_torch()
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args)
    amp_dtype, scaler = _setup_amp(device, args.amp)
    print(
        f"profile={args.profile} device={device} torch={torch.__version__} "
        f"amp={amp_dtype or 'off'} seed={args.seed}",
        flush=True,
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"vram={props.total_memory / 2**30:.1f}GiB",
            flush=True,
        )

    out_dir = _resolve_path(args.out)
    args.out = out_dir
    data_dir = _resolve_path(args.data)
    if args.smoke:
        data_dir = os.path.join(out_dir, "smoke_shards")
        print(
            f"SMOKE: generating {args.smoke} synthetic positions in {data_dir}",
            flush=True,
        )
        make_smoke_shards(data_dir, n=int(args.smoke), seed=args.seed)
        prof = PROFILES[args.profile]
        if args.hidden == prof["hidden"]:
            args.hidden = 128
        if args.batch == prof["batch"]:
            args.batch = 256
        if args.epochs == prof["epochs"]:
            args.epochs = 2
        if args.lr == prof["lr"]:
            args.lr = 1e-3
        if args.max_samples <= 0:
            args.max_samples = int(args.smoke)
        print(
            f"SMOKE: tiny run hidden={args.hidden} batch={args.batch} "
            f"epochs={args.epochs} (not a strength test)",
            flush=True,
        )
    if not data_dir:
        raise TrainError("--data is required (or use --smoke)")
    args.data = data_dir

    shards = Shards(data_dir, policy=args.policy)
    if not shards.use_policy and args.policy == "auto":
        print("note: moves.u16 absent; training value-only", flush=True)
    if args.cp_weight > 0 and not shards.has_cp:
        print("warning: --cp-weight > 0 but cp.f32 absent; disabling cp loss",
              file=sys.stderr)
        args.cp_weight = 0.0

    n_total = len(shards)
    limit = n_total if args.max_samples <= 0 else min(n_total, int(args.max_samples))
    stats = shards.validate(limit)
    print(f"rows={stats['rows']} usable={stats['usable']} "
          f"dropped={stats['dropped_total']}", flush=True)

    train_idx, val_idx, split = group_split(
        shards.groups, args.val_fraction, args.seed,
        max_samples=limit, valid_mask=getattr(shards, "valid_mask", None),
    )
    if args.max_val_samples > 0 and len(val_idx) > args.max_val_samples:
        val_idx = val_idx[: args.max_val_samples]
        split["val_samples_evaluated"] = int(len(val_idx))
    print(
        f"split: train={len(train_idx)} ({split['train_groups']} groups) "
        f"val={len(val_idx)} ({split['val_groups']} groups) "
        f"val_frac={split['val_fraction_actual']:.4f} overlap=0",
        flush=True,
    )

    # Data source: memmap/prefetch by default, optional full GPU residency.
    data_source = shards
    if args.data_on_gpu:
        gpu_shards = GpuShards(shards, device, limit)
        data_source = gpu_shards
        print(
            f"data-on-gpu: {gpu_shards.n:,} rows resident, "
            f"~{gpu_shards.bytes_used / 2**30:.2f} GiB",
            flush=True,
        )

    policy_size = POLICY_CLASSES if shards.use_policy else 0
    model = HalfKPNet(
        args.hidden,
        policy_size=policy_size,
        dropout=args.dropout,
        sparse=not args.dense_emb,
        init_std=0.01,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: hidden={args.hidden} policy={policy_size} params={n_params:,}",
          flush=True)

    opt_sparse, opt_dense = _make_optimizers(model, args, args.lr)
    optimizers = [o for o in (opt_sparse, opt_dense) if o is not None]
    accum = max(1, int(args.grad_accum))
    micro_per_epoch = max(1, math.ceil(len(train_idx) / args.batch))
    steps_per_epoch = max(1, math.ceil(micro_per_epoch / accum))
    total_steps = steps_per_epoch * args.epochs
    # Keep short runs (smoke/tests) out of a schedule that never leaves warmup.
    warmup = min(args.warmup_steps, max(1, total_steps // 20))
    eff_batch = args.batch * accum
    print(
        f"steps: micro/epoch={micro_per_epoch} opt/epoch={steps_per_epoch} "
        f"effective_batch={eff_batch} warmup={warmup}",
        flush=True,
    )

    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    history: List[dict] = []
    stale_epochs = 0

    resume_path = _resolve_path(args.resume)
    if resume_path and os.path.isdir(resume_path):
        resume_path = os.path.join(resume_path, CHECKPOINT)
    if resume_path:
        if not os.path.exists(resume_path):
            raise TrainError(f"checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if opt_sparse is not None and ckpt.get("opt_sparse") is not None:
            opt_sparse.load_state_dict(ckpt["opt_sparse"])
        if opt_dense is not None and ckpt.get("opt_dense") is not None:
            opt_dense.load_state_dict(ckpt["opt_dense"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        history = list(ckpt.get("history", []))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        print(f"resumed {resume_path} at epoch {start_epoch}", flush=True)

    bce_loss = nn.BCEWithLogitsLoss()
    os.makedirs(out_dir, exist_ok=True)

    if args.export_only:
        if not resume_path:
            raise TrainError("--export-only requires --resume <checkpoint|dir>")
        val = evaluate(
            model,
            data_source,
            val_idx,
            device,
            args.batch,
            max_policy_samples=args.policy_eval_samples,
        )
        card_extra = {
            "config": vars(args),
            "architecture": {
                "inputs": INPUTS,
                "feature": "HalfKP",
                "buckets": 64,
                "hidden": args.hidden,
                "dense": [DENSE_WIDTH, DENSE_WIDTH, 1],
                "policy_size": policy_size,
                "sparse_embedding": not args.dense_emb,
            },
            "training": {
                "epochs_run": int(ckpt.get("epoch", -1)) + 1,
                "best_val": best_val,
                "split": split,
                "data_dir": os.path.abspath(data_dir),
                "data_meta": shards.meta,
                "rows_validation": stats,
                "reexported": True,
            },
            "metrics": {"val_reexport": val},
            "calibration": val["calibration"],
            "per_phase": val["per_phase"],
        }
        if policy_size:
            card_extra["policy"] = {
                "classes": POLICY_CLASSES,
                "encoding": "normal from*64+to; promo 4096+(promo-2)*64+from",
                "val": val.get("policy"),
            }
        exported = export_model(
            model, args, out_dir, args.name, card_extra,
            shards=shards, val_indices=val_idx, device=device,
        )
        print(f"re-exported {exported['nnue']}", flush=True)
        return 0

    stop_early = False
    global_step = start_epoch * steps_per_epoch
    t_train = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()
        if args.lr_schedule == "step":
            lr = args.lr * (args.step_gamma ** (epoch // max(1, args.step_every)))
            _set_lr(optimizers, lr)
        elif args.lr_schedule == "constant":
            lr = args.lr
            _set_lr(optimizers, lr)
        else:
            lr = args.lr
            _set_lr(optimizers, lr)

        running = 0.0
        seen = 0
        pol_running = 0.0
        pol_seen = 0
        cp_running = 0.0
        accum_count = 0
        loader = data_source.train_loader(
            train_idx, args.batch, args.seed + epoch * 7919, args
        )

        for step, raw in enumerate(loader):
            batch = (
                raw
                if isinstance(raw, DeviceBatch)
                else _to_device_batch(raw, device, pin=args.pin_memory)
            )

            if args.lr_schedule == "cosine":
                lr = args.lr * _lr_scale(
                    global_step, total_steps, warmup, args.min_lr_ratio
                )
                _set_lr(optimizers, lr)

            if accum_count == 0:
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                h = model.hidden(batch.feats, batch.offsets)
                logits = model.value(h)
                loss = bce_loss(logits, batch.target)

                cp_loss = torch.zeros((), device=device)
                if args.cp_weight > 0 and batch.cp is not None:
                    finite = torch.isfinite(batch.cp)
                    if bool(finite.any()):
                        cp_loss = F.smooth_l1_loss(
                            logits[finite],
                            batch.cp[finite] / CP_SCALE,
                            beta=args.cp_beta,
                        )
                        loss = loss + args.cp_weight * cp_loss

                pol_loss = torch.zeros((), device=device)
                if policy_size > 0 and batch.moves is not None:
                    valid = batch.moves != IGNORE_INDEX
                    if bool(valid.any()):
                        plogits = model.policy_logits(h[valid])
                        pol_loss = F.cross_entropy(plogits, batch.moves[valid])
                        loss = loss + args.policy_weight * pol_loss

            if scaler is not None:
                scaler.scale(loss / accum).backward()
            else:
                (loss / accum).backward()

            b = batch.n
            running += float(loss.item()) * b
            seen += b
            if policy_size > 0 and batch.moves is not None:
                valid = batch.moves != IGNORE_INDEX
                n_valid = int(valid.sum().item())
                if n_valid:
                    pol_running += float(pol_loss.item()) * n_valid
                    pol_seen += n_valid
            if args.cp_weight > 0:
                cp_running += float(cp_loss.item()) * b

            accum_count += 1
            is_last_micro = (step + 1) == micro_per_epoch
            if accum_count < accum and not is_last_micro:
                continue

            if scaler is not None:
                if opt_sparse is not None:
                    scaler.unscale_(opt_sparse)
                scaler.unscale_(opt_dense)
            if args.sparse_grad_clip > 0:
                _clip_sparse_grads([model.emb.weight], args.sparse_grad_clip)
            if args.grad_clip > 0:
                clip_params = [
                    p
                    for p in model.parameters()
                    if not (model.emb.sparse and p is model.emb.weight)
                ]
                torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            if scaler is not None:
                if opt_sparse is not None:
                    scaler.step(opt_sparse)
                scaler.step(opt_dense)
                scaler.update()
            else:
                if opt_sparse is not None:
                    opt_sparse.step()
                opt_dense.step()

            accum_count = 0
            global_step += 1
            if args.log_every and step % args.log_every == 0:
                msg = (
                    f"epoch {epoch} step {step}/{micro_per_epoch} "
                    f"loss {running / max(seen, 1):.5f} lr {lr:.2e}"
                )
                if pol_seen:
                    msg += f" policy_ce {pol_running / pol_seen:.4f}"
                if args.cp_weight > 0:
                    msg += f" cp_sl1 {cp_running / max(seen, 1):.4f}"
                if seen:
                    msg += f" samples/s {seen / max(1e-9, time.time() - t_epoch):.0f}"
                msg += f" {time.time() - t_train:.0f}s"
                print(msg, flush=True)

        val = evaluate(
            model,
            data_source,
            val_idx,
            device,
            args.batch,
            max_policy_samples=args.policy_eval_samples,
        )
        val["epoch"] = epoch
        val["train_loss"] = running / max(seen, 1)
        val["lr_end"] = lr
        val["train_policy_ce"] = (pol_running / pol_seen) if pol_seen else None
        val["train_cp_sl1"] = (cp_running / max(seen, 1)) if args.cp_weight > 0 else None
        history.append(
            {
                "epoch": epoch,
                "train_loss": val["train_loss"],
                "val_loss": val["loss"],
                "val_brier": val["brier"],
                "val_ece": val["ece"],
                "val_acc_decisive": val["acc_decisive"],
                "val_policy_top1": val.get("policy", {}).get("top1"),
                "val_policy_top3": val.get("policy", {}).get("top3"),
            }
        )
        print(
            f"epoch {epoch} val_loss {val['loss']:.5f} brier {val['brier']:.5f} "
            f"ece {val['ece']:.4f} acc {val['acc_decisive']} "
            f"policy {val.get('policy')} ({time.time() - t_train:.0f}s)",
            flush=True,
        )

        improved = val["loss"] < best_val - args.min_delta
        if improved:
            best_val = float(val["loss"])
            best_epoch = epoch
            stale_epochs = 0
            card_extra = {
                "config": vars(args),
                "architecture": {
                    "inputs": INPUTS,
                    "feature": "HalfKP",
                    "buckets": 64,
                    "hidden": args.hidden,
                    "dense": [DENSE_WIDTH, DENSE_WIDTH, 1],
                    "policy_size": policy_size,
                    "sparse_embedding": not args.dense_emb,
                },
                "training": {
                    "epochs_planned": args.epochs,
                    "epochs_run": epoch + 1,
                    "best_epoch": best_epoch,
                    "history": history,
                    "split": split,
                    "data_dir": os.path.abspath(data_dir),
                    "data_meta": shards.meta,
                    "rows_validation": stats,
                    "loss": "bce_wdl"
                    + ("+cp_smooth_l1" if args.cp_weight > 0 else "")
                    + ("+policy_ce" if policy_size else ""),
                    "best_val": best_val,
                    "seed": args.seed,
                    "amp": str(amp_dtype) if amp_dtype else "off",
                    "device": str(device),
                    "grad_accum": accum,
                    "effective_batch": eff_batch,
                    "warmup_steps_effective": warmup,
                    "data_on_gpu": bool(args.data_on_gpu),
                    "num_workers": args.num_workers,
                    "pin_memory": bool(args.pin_memory),
                },
                "metrics": {
                    "val_best": val,
                    "final_epoch": history[-1],
                },
                "calibration": val["calibration"],
                "per_phase": val["per_phase"],
            }
            if policy_size:
                card_extra["policy"] = {
                    "classes": POLICY_CLASSES,
                    "encoding": "normal from*64+to; promo 4096+(promo-2)*64+from",
                    "weight": args.policy_weight,
                    "val": val.get("policy"),
                }
            exported = export_model(
                model, args, out_dir, args.name, card_extra,
                shards=shards, val_indices=val_idx, device=device,
            )
            print(f"exported {exported['nnue']}", flush=True)
            sanity = exported["checks"]["material_sanity"]
            bad = [s["name"] for s in sanity if not s.get("sign_ok", True)]
            if bad:
                print(f"warning: material sanity sign mismatches: {bad}", flush=True)
            ec = exported["checks"].get("export_vs_torch")
            if ec and not ec["ok"]:
                print(
                    f"warning: export vs torch max diff {ec['max_abs_cp_diff']:.3f} cp",
                    flush=True,
                )

        if not improved:
            stale_epochs += 1

        ckpt = {
            "format": "halfkp-train-v1",
            "model": model.state_dict(),
            "opt_sparse": opt_sparse.state_dict() if opt_sparse is not None else None,
            "opt_dense": opt_dense.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "config": vars(args),
            "split": split,
        }
        torch.save(ckpt, os.path.join(out_dir, CHECKPOINT))

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"early stop: no improvement for {stale_epochs} epochs", flush=True)
            stop_early = True
            break

    metrics = {
        "format": "halfkp-train-v1",
        "created_utc": _now(),
        "profile": args.profile,
        "config": vars(args),
        "split": split,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stopped": stop_early,
        "history": history,
        "elapsed_seconds": time.time() - t_start,
        "output_dir": os.path.abspath(out_dir),
        "model_file": os.path.join(os.path.abspath(out_dir), f"{args.name}.nnue"),
        "deployable": args.hidden <= DEPLOYABLE_MAX_HIDDEN,
        "note": "Offline metrics only; they do not prove playing strength.",
    }
    with open(os.path.join(out_dir, METRICS_JSON), "w", encoding="utf-8") as f:
        json.dump(_jsonable(metrics), f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"done: best_val={best_val:.5f} epoch={best_epoch} "
        f"time={time.time() - t_start:.0f}s out={os.path.abspath(out_dir)}",
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _apply_profile(args)
        return train(args)
    except TrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
