#!/usr/bin/env python3
"""Statistics and sanity checks for extracted HalfKP shards.

Reads the shard directory produced by ``extract_halfkp`` plus, when present,
``meta.json`` and the downloader's ``<text>.stats.json`` sidecar (depth
histogram, filter counts).

Checks
  * shard sizes agree, feature indices are in range, feature counts are sane;
  * target (win probability) and cp distributions, WDL-vs-cp calibration;
  * material/phase distribution decoded from the HalfKP features;
  * duplicate rate (group hash), duplicate-target disagreement;
  * policy and PV coverage, policy <-> pv[0] consistency;
  * split check: ``groups % 10 == 0`` is the validation set, but ``groups`` is
    a per-position hash, so this is a random *position* split, not a game
    split. Exact duplicate positions cannot straddle it; sibling positions
    from the same game can. Use ``--holdout`` with an external game/puzzle/
    self-play set to measure that leakage.

Usage:
  python training/data_stats.py --data training/work/pilot/halfkp
  python training/data_stats.py --data ... --text training/work/pilot/positions.txt \
      --holdout <external.epd> \
      --json training/work/pilot/stats.json --md training/work/pilot/stats.md

Exit status: 0 when all hard checks pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

INPUTS = 64 * 12 * 64
POLICY_SIZE = 4352
POLICY_ABSENT = 0xFFFF
PV_LEN = 8
VALUES = np.array([100, 320, 330, 500, 950, 0], dtype=np.int32)  # P,N,B,R,Q,K
PIECE_CHARS = "pnbrqk"


# --------------------------------------------------------------------- loading


def load_shards(root: str, max_rows: int = 0):
    paths = {
        "features": os.path.join(root, "features.u16"),
        "counts": os.path.join(root, "counts.u8"),
        "targets": os.path.join(root, "targets.f32"),
        "cp": os.path.join(root, "cp.f32"),
        "moves": os.path.join(root, "moves.u16"),
        "pv": os.path.join(root, "pv.u16"),
        "groups": os.path.join(root, "groups.u32"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("missing shard(s): " + ", ".join(missing))
    counts = np.memmap(paths["counts"], dtype=np.uint8, mode="r")
    n = len(counts)
    if max_rows and n > max_rows:
        n = max_rows
    arrays = {"n": n, "counts": counts[:n]}
    for name in ("features", "targets", "cp", "moves", "pv", "groups"):
        dtype = np.float32 if name in ("targets", "cp") else (
            np.uint16 if name in ("features", "moves", "pv") else np.uint32
        )
        arr = np.memmap(paths[name], dtype=dtype, mode="r")
        want = n * 32 if name == "features" else (n * PV_LEN if name == "pv" else n)
        if arr.size != want:
            raise ValueError(
                f"{name} has {arr.size} elements, expected {want} "
                f"(shard lengths disagree; corrupt or partial shard?)"
            )
        arrays[name] = arr[:want].reshape(n, 32) if name == "features" else (
            arr[:want].reshape(n, PV_LEN) if name == "pv" else arr[:want]
        )
    return arrays, paths


def read_meta(root: str):
    path = os.path.join(root, "meta.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), path
    except (OSError, json.JSONDecodeError):
        return {}, path


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------- helpers


def decode_policy(idx: int) -> str:
    if idx == POLICY_ABSENT:
        return "none"
    if idx >= 4096:
        k = idx - 4096
        promo = 2 + k // 64
        frm = k % 64
        return f"{sq_name(frm)}{chr(ord('n') + promo - 2)}"
    frm, to = divmod(int(idx), 64)
    return f"{sq_name(frm)}{sq_name(to)}"


def sq_name(sq: int) -> str:
    return f"{chr(97 + sq % 8)}{sq // 8 + 1}"


def hist(values: np.ndarray, bins) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def fnv1a32(text: str) -> int:
    """Same group hash as extract_halfkp (FNV-1a over board|stm|castling|ep)."""
    h = 2166136261
    for b in text.encode("utf-8", "replace"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def read_holdout_fens(path: str):
    """Yield normalized 4-field position keys from a FEN/EPD/text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head = line.split("|", 1)[0].strip()
            parts = head.split()
            if len(parts) >= 4 and parts[1] in ("w", "b"):
                yield " ".join(parts[:4])


def sibling_leakage(groups: np.ndarray, text_path: str, sample_rows: int = 50_000,
                    pv_moves: int = 8) -> dict:
    """Estimate same-game (sibling-position) leakage that the %10 split misses.

    For a deterministic sample of pilot rows, replay the PV one ply at a time,
    hash each resulting position with the same group function, and look it up
    among the shard groups. A hit with a different ``hash % 10`` means a
    sibling position of a training row lives in the validation split.
    """
    try:
        import chess
    except ImportError:
        return {"available": False, "reason": "python-chess not installed"}
    try:
        with open(text_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh
            n_text = sum(1 for _ in lines)
        stride = max(1, n_text // max(sample_rows, 1))
        uniq = np.unique(np.asarray(groups, dtype=np.uint64))
        total = present = same_split = other_split = rows = 0
        with open(text_path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i % stride:
                    continue
                parts = line.rstrip("\n").split("|")
                if len(parts) < 5:
                    continue
                rows += 1
                fen = parts[0]
                try:
                    board = chess.Board(fen, chess960=True)
                except ValueError:
                    continue
                src = fnv1a32(" ".join(fen.split()[:4]))
                for tok in parts[4].split()[:pv_moves]:
                    try:
                        board.push_uci(tok)
                    except ValueError:
                        break
                    h = fnv1a32(" ".join(board.fen().split()[:4]))
                    total += 1
                    pos = int(np.searchsorted(uniq, np.uint64(h)))
                    if pos < len(uniq) and int(uniq[pos]) == h:
                        present += 1
                        if h % 10 == src % 10:
                            same_split += 1
                        else:
                            other_split += 1
        return {
            "available": True,
            "sampled_rows": rows,
            "sample_stride": stride,
            "sibling_positions_checked": total,
            "siblings_found_in_pilot": present,
            "same_split": same_split,
            "different_split": other_split,
            "leakage_rate_of_found": (float(other_split) / present) if present else None,
            "note": "PV-replay sibling estimate; different_split siblings are "
                    "same-game positions that sit on the other side of the "
                    "%10 position-hash split (a lower bound on game leakage)",
        }
    except OSError as exc:
        return {"available": False, "reason": str(exc)}


def pct(a, b) -> float | None:
    return float(a) / float(b) if b else None


# ----------------------------------------------------------------------- checks


def compute_stats(arrays, meta, text_path: str, top: int = 8,
                  holdout_path: str = "", sibling_sample: int = 50_000) -> dict:
    n = arrays["n"]
    counts = arrays["counts"]
    feats = arrays["features"]
    targets = arrays["targets"]
    cp = arrays["cp"]
    moves = arrays["moves"]
    pv = arrays["pv"]
    groups = arrays["groups"]

    out: dict = {"count": int(n), "shard_dir": meta.get("source_dir", ""),
                 "meta": meta}
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # ---- features
    cnt = counts.astype(np.int32)
    out["feature_count"] = {
        "min": int(cnt.min()) if n else 0,
        "max": int(cnt.max()) if n else 0,
        "mean": float(cnt.mean()) if n else 0.0,
        "zero_rows": int((cnt == 0).sum()),
        "over_32": int((cnt > 32).sum()),
        "histogram": {
            str(k): int(v) for k, v in zip(*np.unique(cnt, return_counts=True))
        },
    }
    max_idx = int(feats.max()) if n else 0
    out["feature_index_max"] = max_idx
    check("feature indices < 49152", max_idx < INPUTS, f"max={max_idx}")
    check("counts <= 32", out["feature_count"]["max"] <= 32,
          f"max={out['feature_count']['max']}")
    check("counts > 0", out["feature_count"]["zero_rows"] == 0,
          f"{out['feature_count']['zero_rows']} empty rows")

    # duplicate feature inside a row (padding is 0, a valid index, so mask)
    dup_rows = 0
    active_mask = np.arange(32, dtype=np.int32)[None, :] < cnt[:, None]
    chunk = 262144
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        block = np.where(active_mask[s:e], feats[s:e], np.int32(-1))
        block = np.sort(block, axis=1)
        dup = (block[:, 1:] == block[:, :-1]) & (block[:, 1:] >= 0)
        dup_rows += int(dup.any(axis=1).sum())
    out["duplicate_features_in_row"] = dup_rows
    check("no duplicate features within a row", dup_rows == 0, f"{dup_rows} rows")

    # ---- targets
    tv = np.asarray(targets, dtype=np.float64)
    finite_t = np.isfinite(tv)
    out["target"] = {
        "finite": int(finite_t.sum()),
        "nan": int((~finite_t).sum()),
        "mean": float(tv[finite_t].mean()) if finite_t.any() else None,
        "std": float(tv[finite_t].std()) if finite_t.any() else None,
        "min": float(tv[finite_t].min()) if finite_t.any() else None,
        "max": float(tv[finite_t].max()) if finite_t.any() else None,
        "outside_0_1": int(((tv < 0) | (tv > 1)).sum()),
        "histogram": hist(tv[finite_t], np.linspace(0.0, 1.0, 21))
        if finite_t.any() else None,
    }
    check("targets in [0,1]", out["target"]["outside_0_1"] == 0)

    # ---- cp
    cv = np.asarray(cp, dtype=np.float64)
    fin = np.isfinite(cv)
    out["cp"] = {
        "finite": int(fin.sum()),
        "missing": int((~fin).sum()),
        "mean": float(cv[fin].mean()) if fin.any() else None,
        "std": float(cv[fin].std()) if fin.any() else None,
        "q": {str(q): float(np.quantile(cv[fin], q / 100))
              for q in (1, 5, 25, 50, 75, 95, 99)} if fin.any() else None,
        "histogram": hist(cv[fin], np.linspace(-1600, 1600, 33))
        if fin.any() else None,
        "abs_gt_1500": int((np.abs(cv[fin]) > 1500).sum()) if fin.any() else 0,
    }
    check("|cp| <= 1500", out["cp"]["abs_gt_1500"] == 0)
    check("targets finite", out["target"]["nan"] == 0)

    # ---- WDL vs cp calibration (targets are sigmoid(cp/400))
    calib = []
    if fin.any():
        edges = np.array([-2000, -800, -400, -200, -100, -50, 0, 50, 100, 200,
                          400, 800, 2000], dtype=np.float64)
        idx = np.digitize(cv[fin], edges) - 1
        tfin = tv[fin]
        for b in range(len(edges) - 1):
            m = idx == b
            if m.any():
                calib.append({
                    "cp_bin": [float(edges[b]), float(edges[b + 1])],
                    "n": int(m.sum()),
                    "mean_target": float(tfin[m].mean()),
                })
    out["calibration"] = calib
    monotone = all(
        calib[i]["mean_target"] <= calib[i + 1]["mean_target"] + 1e-3
        for i in range(len(calib) - 1)
    )
    check("mean target monotone in cp", monotone, f"{len(calib)} bins")

    # ---- material / phase decoded from features
    own_mat = np.zeros(n, dtype=np.int32)
    opp_mat = np.zeros(n, dtype=np.int32)
    piece_hist = {c: 0 for c in PIECE_CHARS}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idx = feats[s:e].astype(np.int32)
        q = idx // 64
        rel = (q % 12) // 6
        typ = q % 6
        val = VALUES[typ]
        mask = active_mask[s:e]
        own_mat[s:e] = (val * (rel == 0) * mask).sum(axis=1)
        opp_mat[s:e] = (val * (rel == 1) * mask).sum(axis=1)
        for t, ch in enumerate(PIECE_CHARS):
            piece_hist[ch] += int(((typ == t) & mask).sum())
    total = own_mat + opp_mat
    out["material"] = {
        "own_mean": float(own_mat.mean()) if n else 0.0,
        "opp_mean": float(opp_mat.mean()) if n else 0.0,
        "total_nonpawn_mean": float(total.mean()) if n else 0.0,
        "total_histogram": hist(total.astype(np.float64), np.linspace(0, 8000, 33))
        if n else None,
        "piece_counts": piece_hist,
        "phase_buckets": {
            "opening_ge_6200": int((total >= 6200).sum()),
            "middlegame_3000_6200": int(((total >= 3000) & (total < 6200)).sum()),
            "endgame_lt_3000": int((total < 3000).sum()),
        },
    }
    if n:
        out["material"]["phase_fraction"] = {
            k: float(v) / n for k, v in out["material"]["phase_buckets"].items()
        }

    # ---- duplicates / groups
    uniq, mult = np.unique(np.asarray(groups), return_counts=True)
    observed_dups = int(n - len(uniq))
    # groups.u32 is a 32-bit hash, so ~n^2 / 2^33 rows collide by construction.
    expected_collisions = n * (n - 1) / 2.0 / (2.0 ** 32)
    tolerance = 5.0 * math.sqrt(max(expected_collisions, 0.0))
    out["groups"] = {
        "unique": int(len(uniq)),
        "duplicate_rows": observed_dups,
        "duplicate_rate": float(1.0 - len(uniq) / n) if n else 0.0,
        "expected_hash_collisions": float(expected_collisions),
        "excess_duplicates": float(observed_dups - expected_collisions),
        "max_multiplicity": int(mult.max()) if len(mult) else 0,
        "multiplicity_histogram": {
            str(k): int(v) for k, v in zip(*np.unique(mult, return_counts=True))
        },
    }
    check(
        "no duplicate groups beyond 32-bit hash collisions",
        observed_dups <= expected_collisions + tolerance,
        f"observed={observed_dups} expected_collisions={expected_collisions:.1f} "
        f"tolerance={tolerance:.1f}",
    )
    # Same-group target disagreement: with a 32-bit group hash most of these
    # are unrelated positions that collided. Real duplicate positions must
    # still have one label; report the worst spread for review.
    if len(uniq) < n:
        order = np.argsort(np.asarray(groups), kind="stable")
        g_sorted = np.asarray(groups)[order]
        t_sorted = tv[order]
        starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
        ends = np.r_[starts[1:], n]
        disagree = 0
        worst = 0.0
        for a, b in zip(starts, ends):
            if b - a > 1:
                spread = float(t_sorted[a:b].max() - t_sorted[a:b].min())
                if spread > 1e-6:
                    disagree += 1
                    worst = max(worst, spread)
        out["groups"]["groups_with_target_disagreement"] = disagree
        out["groups"]["worst_target_spread"] = worst

    # ---- policy / pv coverage
    mv = np.asarray(moves, dtype=np.int32)
    pvv = np.asarray(pv, dtype=np.int32)
    present = mv != POLICY_ABSENT
    promo = (mv >= 4096) & present
    top_idx, top_cnt = np.unique(mv[present], return_counts=True)
    order = np.argsort(-top_cnt)[:top]
    out["policy"] = {
        "present": int(present.sum()),
        "coverage": pct(present.sum(), n),
        "promotions": int(promo.sum()),
        "normal": int((present & ~promo).sum()),
        "out_of_range": int((mv[present] >= POLICY_SIZE).sum()),
        "top": [
            {"index": int(top_idx[i]), "move": decode_policy(int(top_idx[i])),
             "count": int(top_cnt[i])}
            for i in order
        ],
        "distinct_moves": int(len(top_idx)),
    }
    check("policy indices < 4352", out["policy"]["out_of_range"] == 0)
    pv_present = pvv[:, 0] != POLICY_ABSENT
    pv_len = (pvv != POLICY_ABSENT).sum(axis=1)
    out["pv"] = {
        "present": int(pv_present.sum()),
        "coverage": pct(pv_present.sum(), n),
        "length_histogram": {
            str(k): int(v) for k, v in zip(*np.unique(pv_len, return_counts=True))
        },
        "full_8": int((pv_len == 8).sum()),
        "out_of_range": int(((pvv >= POLICY_SIZE) & (pvv != POLICY_ABSENT)).sum()),
    }
    check("pv indices < 4352", out["pv"]["out_of_range"] == 0)
    both = present & pv_present
    if both.any():
        same = int((mv[both] == pvv[both, 0]).sum())
        out["policy"]["matches_pv0"] = same
        out["policy"]["matches_pv0_fraction"] = pct(same, int(both.sum()))
        check("policy == pv[0] when both present", same == int(both.sum()),
              f"{same}/{int(both.sum())}")

    # ---- split check: the rule in train_halfkp.py is a pure function of the
    # per-position hash, i.e. a random *position* split, not a game split.
    val_mask = (np.asarray(groups) % 10) == 0
    train_groups = set(np.unique(np.asarray(groups)[~val_mask]).tolist())
    val_groups = set(np.unique(np.asarray(groups)[val_mask]).tolist())
    overlap = train_groups & val_groups
    out["split"] = {
        "rule": "groups % 10 == 0 -> validation (training/train_halfkp.py)",
        "semantics": "random position split keyed by the groups.u32 hash; "
                     "there are no game IDs in the Lichess eval dump, so this "
                     "is NOT a game/position-family split",
        "train_rows": int((~val_mask).sum()),
        "val_rows": int(val_mask.sum()),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "group_overlap": len(overlap),
        "leakage_warning": (
            "exact duplicate positions cannot straddle the split, but sibling "
            "positions one ply apart (same game) can. Measure that with an "
            "external game-level holdout (PGN with game ids), self-play games, "
            "or the Lichess puzzle set; this check does not detect it."
        ),
    }
    check("position hash not in both train and val (split is a pure function "
          "of the hash)",
          len(overlap) == 0, f"{len(overlap)} overlapping hashes")
    if n:
        # Same-position rows always share a group, so they cannot straddle the
        # split. Report near-duplicate risk explicitly as a limitation.
        out["split"]["duplicate_groups"] = int((mult > 1).sum())
        out["split"]["near_duplicate_check"] = (
            "not computable from shards: groups hash board|stm|castling|ep, "
            "so sibling positions (one ply apart) may land in different "
            "splits; use an external game/puzzle/self-play holdout"
        )

    # ---- internal sibling leakage estimate (needs the text file + chess)
    if text_path and sibling_sample > 0:
        out["split"]["sibling_leakage"] = sibling_leakage(
            np.asarray(groups, dtype=np.uint64), text_path, sibling_sample
        )

    # ---- optional external holdout: how many holdout positions already
    # appear in the pilot? Informational (a shared position is label leakage
    # for that position, not a defect of the shards).
    if holdout_path:
        holdout_keys = list(read_holdout_fens(holdout_path))
        hv = np.fromiter((fnv1a32(k) for k in holdout_keys), dtype=np.uint64,
                         count=len(holdout_keys))
        uniq_u64 = uniq.astype(np.uint64)
        matched = 0
        if len(hv):
            pos = np.searchsorted(uniq_u64, hv)
            pos = np.clip(pos, 0, len(uniq_u64) - 1)
            matched = int((uniq_u64[pos] == hv).sum())
        out["holdout"] = {
            "path": holdout_path,
            "positions": len(holdout_keys),
            "overlapping_positions": matched,
            "overlap_rate": pct(matched, len(holdout_keys)),
            "leakage_detected": matched > 0,
            "note": "exact position overlap between the pilot and the external "
                    "holdout; run a game-level holdout as well to measure "
                    "sibling-position leakage",
        }

    # ---- optional text-side stats (depth / pov / format)
    if text_path:
        text_stats_path = text_path + ".stats.json"
        ts = read_json(text_stats_path)
        out["text"] = {"path": text_path, "stats_file": text_stats_path,
                       "available": ts is not None}
        if ts:
            out["text"]["filters"] = ts.get("filters")
            out["text"]["reject"] = ts.get("reject")
            out["text"]["depth_histogram"] = ts.get("depth_hist")
            out["text"]["pv_len_histogram"] = ts.get("pv_len_hist")
            out["text"]["castling_960_normalized"] = ts.get("castling_960_normalized")
            out["text"]["dedup_hits"] = ts.get("dedup_hits")
            out["text"]["pov"] = ts.get("pov")

    out["checks"] = checks
    out["ok"] = all(c["ok"] for c in checks)
    return out


# --------------------------------------------------------------------- reporting


def render_markdown(st: dict) -> str:
    L = []
    n = st["count"]
    L.append("# HalfKP shard statistics")
    L.append("")
    L.append(f"- shard dir: `{st['shard_dir']}`")
    L.append(f"- positions: **{n:,}**")
    if st["meta"]:
        meta_brief = {
            k: st["meta"][k]
            for k in ("format", "feature_version", "inputs", "policy_size")
            if k in st["meta"]
        }
        L.append(f"- meta.json: {json.dumps(meta_brief)}")
    L.append("")
    L.append("## Checks")
    L.append("")
    for c in st["checks"]:
        L.append(f"- [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}"
                 + (f" ({c['detail']})" if c["detail"] else ""))
    L.append(f"- overall: **{'PASS' if st['ok'] else 'FAIL'}**")
    L.append("")
    fc = st["feature_count"]
    L.append("## Features")
    L.append("")
    L.append(f"- active per row: min {fc['min']}, max {fc['max']}, "
             f"mean {fc['mean']:.2f}; empty rows {fc['zero_rows']}; "
             f"max index {st['feature_index_max']}")
    h = fc["histogram"]
    keys = sorted(h, key=lambda k: int(k))
    L.append("- counts histogram: " + ", ".join(f"{k}:{h[k]}" for k in keys))
    L.append("")
    L.append("## Targets and cp")
    L.append("")
    t = st["target"]
    L.append(f"- win probability: mean {fmt(t['mean'])}, std {fmt(t['std'])}, "
             f"min {fmt(t['min'])}, max {fmt(t['max'])}, NaN {t['nan']}")
    c = st["cp"]
    L.append(f"- cp: finite {c['finite']:,}, missing {c['missing']:,}, "
             f"mean {fmt(c['mean'])}, std {fmt(c['std'])}")
    if c["q"]:
        L.append("- cp quantiles (1/5/25/50/75/95/99): "
                 + " / ".join(f"{c['q'][str(q)]:.0f}" for q in (1, 5, 25, 50, 75, 95, 99)))
    L.append("")
    L.append("### WDL calibration (mean target by cp bin)")
    L.append("")
    if st["calibration"]:
        L.append("| cp bin | n | mean target |")
        L.append("|---|---:|---:|")
        for row in st["calibration"]:
            L.append(f"| {row['cp_bin'][0]:.0f}..{row['cp_bin'][1]:.0f} | "
                     f"{row['n']:,} | {row['mean_target']:.4f} |")
    else:
        L.append("_no finite cp_")
    L.append("")
    m = st["material"]
    L.append("## Material / phase")
    L.append("")
    L.append(f"- non-pawn material total (both sides, cp): mean "
             f"{m['total_nonpawn_mean']:.0f}")
    if "phase_fraction" in m:
        for k, v in m["phase_fraction"].items():
            L.append(f"  - {k}: {100 * v:.1f}%")
    L.append("- piece counts (own+opponent appearances): "
             + ", ".join(f"{k}={v:,}" for k, v in m["piece_counts"].items()))
    L.append("")
    g = st["groups"]
    L.append("## Duplicates / groups")
    L.append("")
    L.append(f"- unique groups {g['unique']:,}; rows sharing a group "
             f"{g['duplicate_rows']:,}; expected 32-bit hash collisions "
             f"{g['expected_hash_collisions']:.1f}; excess "
             f"{g['excess_duplicates']:,}; max multiplicity "
             f"{g['max_multiplicity']}")
    L.append("- `groups.u32` is a 32-bit hash of board|stm|castling|ep, so a "
             "small collision rate is expected. Collisions only merge two "
             "unrelated positions into the same split (conservative, not "
             "leakage); the check fails only when duplicates exceed that "
             "budget.")
    if "groups_with_target_disagreement" in g:
        L.append(f"- groups with disagreeing targets: "
                 f"{g['groups_with_target_disagreement']} "
                 f"(worst spread {g['worst_target_spread']:.6f}); almost all "
                 f"are hash collisions, which is why the split rule stays "
                 f"conservative")
    L.append("")
    pol = st["policy"]
    L.append("## Policy")
    L.append("")
    L.append(f"- present {pol['present']:,} ({fmt_pct(pol['coverage'])}), "
             f"normal {pol['normal']:,}, promotions {pol['promotions']:,}, "
             f"distinct {pol['distinct_moves']:,}")
    if "matches_pv0_fraction" in pol:
        L.append(f"- moves == pv[0]: {fmt_pct(pol['matches_pv0_fraction'])}")
    if pol["top"]:
        L.append("- top moves: " + ", ".join(
            f"{x['move']} ({x['count']:,})" for x in pol["top"]))
    L.append("")
    p = st["pv"]
    L.append("## PV")
    L.append("")
    L.append(f"- present {p['present']:,} ({fmt_pct(p['coverage'])}), full 8 "
             f"moves {p['full_8']:,}")
    L.append("- length histogram: " + ", ".join(
        f"{k}:{v}" for k, v in sorted(p["length_histogram"].items(), key=lambda kv: int(kv[0]))))
    L.append("")
    s = st["split"]
    L.append("## Split check (random-position hash split, not a game split)")
    L.append("")
    L.append(f"- rule: {s['rule']}")
    L.append(f"- semantics: {s['semantics']}")
    L.append(f"- train rows {s['train_rows']:,} / val rows {s['val_rows']:,}; "
             f"hashes {s['train_groups']:,} / {s['val_groups']:,}; "
             f"overlap {s['group_overlap']}")
    L.append(f"- leakage: {s['leakage_warning']}")
    if "near_duplicate_check" in s:
        L.append(f"- near-duplicate check: {s['near_duplicate_check']}")
    sl = s.get("sibling_leakage")
    if sl:
        L.append("")
        if not sl.get("available"):
            L.append(f"- sibling leakage estimate: not run ({sl.get('reason')})")
        else:
            L.append("### Sibling-position leakage estimate (PV replay)")
            L.append("")
            L.append(f"- sampled rows {sl['sampled_rows']:,} "
                     f"(stride {sl['sample_stride']}), "
                     f"{sl['sibling_positions_checked']:,} sibling positions "
                     f"replayed")
            L.append(f"- siblings that also exist in the pilot: "
                     f"{sl['siblings_found_in_pilot']:,}; same split "
                     f"{sl['same_split']:,}; **different split "
                     f"{sl['different_split']:,}** "
                     f"({fmt_pct(sl['leakage_rate_of_found'])} of found)")
            L.append(f"- {sl['note']}")
    L.append("")
    if st.get("holdout"):
        h = st["holdout"]
        L.append("## External holdout overlap")
        L.append("")
        L.append(f"- file `{h['path']}`: {h['positions']:,} positions, "
                 f"{h['overlapping_positions']:,} also present in the pilot "
                 f"({fmt_pct(h['overlap_rate'])})")
        L.append(f"- {h['note']}")
        if h["leakage_detected"]:
            L.append("- **warning**: exact positions are shared with the "
                     "holdout; exclude them from one side of the validation")
        L.append("")
    if st.get("text", {}).get("available"):
        tx = st["text"]
        L.append("## Depth / source filters (from text stats sidecar)")
        L.append("")
        L.append(f"- filters: {json.dumps(tx.get('filters'))}")
        if tx.get("depth_histogram"):
            hist_items = sorted(tx["depth_histogram"].items(), key=lambda kv: int(kv[0]))
            L.append("- depth histogram: " + ", ".join(f"{k}:{v}" for k, v in hist_items))
        L.append(f"- castling_960_normalized: {tx.get('castling_960_normalized')}")
        L.append(f"- source POV: {tx.get('pov')}")
        L.append("")
    return "\n".join(L)


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="training/work/halfkp", help="shard directory")
    ap.add_argument("--text", default="", help="positions text file (optional)")
    ap.add_argument("--holdout", default="",
                    help="external FEN/EPD/text holdout to measure exact "
                         "position overlap (e.g. data/quiet-labeled.epd)")
    ap.add_argument("--sibling-sample", type=int, default=50_000,
                    help="PV-replay rows used to estimate same-game leakage "
                         "(0 disables; needs --text)")
    ap.add_argument("--json", default="", help="write JSON stats here")
    ap.add_argument("--md", default="", help="write Markdown report here")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        arrays, _ = load_shards(args.data, args.max_rows)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    meta, _ = read_meta(args.data)
    meta["source_dir"] = args.data
    st = compute_stats(arrays, meta, args.text, args.top, args.holdout,
                       args.sibling_sample)
    md = render_markdown(st)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(st, fh, indent=2, default=float)
    if args.md:
        os.makedirs(os.path.dirname(args.md) or ".", exist_ok=True)
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write(md + "\n")
    if not args.quiet:
        print(md)
    for c in st["checks"]:
        if not c["ok"]:
            print(f"FAIL: {c['name']} {c['detail']}", file=sys.stderr)
    return 0 if st["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
