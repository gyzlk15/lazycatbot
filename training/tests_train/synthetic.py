"""Tiny synthetic shard generator used by the training tests.

The shards are structurally identical to ``extract_halfkp`` output
(features/counts/targets/cp/moves/pv/groups/meta) but contain only a few
hundred random-playout positions.  They are for smoke tests only.
"""
from __future__ import annotations

from train_halfkp import make_smoke_shards


def write_synthetic_shards(
    out_dir: str, n: int = 640, seed: int = 11
) -> str:
    """Write synthetic HalfKP shards into ``out_dir`` and return the path."""
    return make_smoke_shards(out_dir, n=n, seed=seed)


def write_value_only_shards(
    out_dir: str, n: int = 640, seed: int = 11
) -> str:
    """Synthetic shards without ``cp.f32``/``moves.u16``/``pv.u16`` (optional files)."""
    import os

    write_synthetic_shards(out_dir, n=n, seed=seed)
    for name in ("cp.f32", "moves.u16", "pv.u16"):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.remove(path)
    return out_dir


__all__ = ["write_synthetic_shards", "write_value_only_shards"]
