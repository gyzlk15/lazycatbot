#!/usr/bin/env python3
"""Write a SHA-256 manifest for a pilot dataset directory.

The manifest is the handoff record for the training/validation agents: exact
files, sizes, hashes, counts, source completeness, the commands that produced
the data, and the POV/Chess960 verification summary.

Usage (from the repository root):
  python training/tests_data/make_pilot_manifest.py \
      --dir training/work/pilot_build \
      --final-path training/work/pilot \
      --source training/work/lichess_db_eval.jsonl.zst \
      --commands build_commands.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

EXPECTED_DB_BYTES = 22_499_667_845
MANIFEST_NAME = "manifest.json"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="built pilot directory")
    ap.add_argument("--final-path", default="", help="path recorded in the manifest")
    ap.add_argument("--source", default="training/work/lichess_db_eval.jsonl.zst")
    ap.add_argument("--source-manifest",
                    default="training/work/lichess_db_eval.manifest.json")
    ap.add_argument("--commands", default="", help="JSON file with build commands")
    ap.add_argument("--name", default="lichess-eval-halfkp-pilot")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"missing directory {root}", file=__import__("sys").stderr)
        return 1

    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if name == MANIFEST_NAME:
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            files.append({
                "path": rel,
                "bytes": os.path.getsize(full),
                "sha256": sha256_file(full),
            })
    files.sort(key=lambda f: f["path"])

    meta = read_json(os.path.join(root, "halfkp", "meta.json")) or {}
    stats = read_json(os.path.join(root, "positions.txt.stats.json")) or {}
    pov = read_json(os.path.join(root, "pov_report.json")) or {}
    commands = read_json(args.commands) if args.commands else None
    if args.commands and commands is None:
        print(f"warning: cannot read commands file {args.commands}")
    source_manifest = read_json(args.source_manifest) or {}
    source_bytes = None
    if os.path.exists(args.source):
        source_bytes = os.path.getsize(args.source)
    complete = False
    if source_bytes is not None:
        expected = int(source_manifest.get("compressed_size", 0)) or EXPECTED_DB_BYTES
        complete = source_bytes == expected and bool(source_manifest)

    shard_counts = {}
    for name in ("features", "counts", "targets", "cp", "moves", "pv", "groups"):
        path = os.path.join(root, "halfkp", name + (".u16" if name in
                           ("features", "moves", "pv") else ".u8" if name == "counts"
                           else ".f32" if name in ("targets", "cp") else ".u32"))
        if os.path.exists(path):
            shard_counts[name] = os.path.getsize(path)
    manifest_counts = dict(shard_counts)

    manifest = {
        "dataset": args.name,
        "status": "stable",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "path": args.final_path or args.dir,
        "build_dir": os.path.relpath(root, os.getcwd()).replace(os.sep, "/"),
        "source": {
            "path": args.source,
            "line_count_used": stats.get("lines"),
            "bytes_at_manifest": source_bytes,
            "expected_bytes": EXPECTED_DB_BYTES,
            "complete_at_manifest": complete,
            "verified_manifest_present": bool(source_manifest),
            "note": "the full download was still running when this pilot was "
                    "built; source bytes were read at manifest time, after the "
                    "sample, so the file is larger than when the sample ended",
        },
        "counts": {
            "positions_written": stats.get("written"),
            "shard_rows": meta.get("count"),
            "skipped_by_extractor": meta.get("skipped"),
            "skip_reasons": meta.get("skip_reasons"),
            "policy_present": meta.get("policy_present"),
            "pv_present": meta.get("pv_present"),
            "cp_present": meta.get("cp_present"),
        },
        "filters": stats.get("filters"),
        "commands": commands,
        "pov_chess960": {
            "source_pov_verdict": (pov.get("pov") or {}).get("verdict"),
            "black_to_move": (pov.get("pov") or {}).get("black_to_move"),
            "pv_lines_checked": (pov.get("pv_chess960") or {}).get("lines_checked"),
            "pv_legal_under_chess960": (pov.get("pv_chess960") or {}).get("legal_960"),
            "castling_960_tokens": (pov.get("pv_chess960") or {}).get("castling_960_tokens"),
            "standard_board_failures": (pov.get("pv_chess960") or {}).get("standard_fail_lines"),
            "output_text_pov": ((pov.get("text_check") or {}).get("pov") or {}).get("verdict"),
            "report": "pov_report.md/json",
        },
        "group_semantics": {
            "hash": "FNV-1a 32 over board|stm|castling|ep",
            "game_id_available": False,
            "split_rule": "groups % 10 == 0 -> validation (train_halfkp.py)",
            "meaning": "random-position split, NOT a game/position-family split",
            "exact_duplicates_cannot_straddle": True,
            "sibling_positions_can_straddle": True,
            "sibling_leakage_estimate": ((read_json(os.path.join(root, "stats.json"))
                                          or {}).get("split") or {}).get(
                "sibling_leakage"),
            "external_holdout_required": "game-level PGN, self-play games, or "
                                         "the Lichess puzzle set; measure with "
                                         "data_stats.py --holdout",
        },
        "files": files,
        "shard_bytes": manifest_counts,
        "total_bytes": sum(f["bytes"] for f in files),
        "notes": [
            "No game IDs exist in lichess_db_eval; the %10 split is by position "
            "hash and leaks same-game sibling positions (see "
            "group_semantics.sibling_leakage_estimate). Treat internal "
            "validation as random-position generalization until an external "
            "game-level holdout is measured.",
            "Feature layout and policy encoding are fixed by "
            "docs/NNUE_PILOT.md 3.1/3.2 and documented in "
            "training/data_contract.md.",
        ],
    }

    out = os.path.join(root, MANIFEST_NAME)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    print(f"manifest: {out} ({len(files)} files, {manifest['total_bytes']/2**30:.3f} GiB)")
    print(f"  rows={manifest['counts']['shard_rows']} "
          f"source_complete={complete} pov={(pov.get('pov') or {}).get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
