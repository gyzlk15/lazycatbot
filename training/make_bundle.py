#!/usr/bin/env python3
"""Build a portable, secret-free LazyCatBot NNUE training bundle.

The bundle contains training/export code, requirements, the runbook and the
format/plan documents.  It never contains `.venv`, `training/work`, checkpoints,
model files, datasets, SSH keys or tokens.

Usage (from anywhere)::

    python3 training/make_bundle.py
    python3 training/make_bundle.py --out temp/bundles --name lcb_nnue
    python3 training/make_bundle.py --with-src   # also ship src/ for extraction
    python3 training/make_bundle.py --dry-run

Output: ``<out>/<name>_<UTC timestamp>.tar.gz`` plus ``.sha256``.  Inside the
archive, ``MANIFEST.json`` and ``SHA256SUMS`` describe every file.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Small, explicit allow-list of source patterns.  Everything else (data,
# weights, logs, venvs, archives) is excluded even if it matches nothing here.
INCLUDE_PATTERNS = (
    "training/*.py",
    "training/*.md",
    "training/*.txt",
    "training/*.sh",
    "training/tests_train/*.py",
    "training/tests_train/*.md",
    "docs/NNUE_PILOT.md",
    "docs/reports/UNRESOLVED.md",
    "src/*.cpp",
    "src/*.hpp",
)

EXCLUDED_DIR_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ssh",
    ".aws",
}
EXCLUDED_PATH_PREFIXES = (
    "training/work/",
    "temp/",
    "build",
    "dist/",
    "data/",
)
EXCLUDED_SUFFIXES = {
    ".zst",
    ".u16",
    ".u8",
    ".f32",
    ".u32",
    ".i16",
    ".log",
    ".nnue",
    ".pt",
    ".pth",
    ".npz",
    ".bin",
    ".ckpt",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".zip",
    ".sqlite",
    ".db",
}
SECRET_NAME_PATTERNS = (
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    ".env*",
    "*.token",
    "secrets*",
    "credentials*",
)
SECRET_MARKERS = (
    b"BEGIN " b"OPENSSH PRIVATE KEY",
    b"BEGIN " b"RSA PRIVATE KEY",
    b"BEGIN " b"EC PRIVATE KEY",
    b"BEGIN " b"PRIVATE KEY",
    b"gh" b"p_",
    b"AK" b"IA",
    b"xo" b"xb-",
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".sh", ".json", ".yml", ".yaml", ".toml",
                 ".hpp", ".cpp", ".c", ".h"}


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _excluded(rel: Path) -> bool:
    posix = rel.as_posix()
    if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
        return True
    if any(posix == p.rstrip("/") or posix.startswith(p) for p in EXCLUDED_PATH_PREFIXES):
        return True
    if rel.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    for pattern in SECRET_NAME_PATTERNS:
        if fnmatch.fnmatch(rel.name.lower(), pattern):
            return True
    return False


def _scan_secrets(rel: Path, data: bytes) -> None:
    if rel.suffix.lower() not in TEXT_SUFFIXES or len(data) > 2_000_000:
        return
    for marker in SECRET_MARKERS:
        if marker in data:
            raise SystemExit(
                f"refusing to bundle {rel}: contains secret marker "
                f"{marker.decode('ascii', 'replace')!r}"
            )


def _collect(include_src: bool = True) -> List[Tuple[Path, bytes]]:
    files: List[Tuple[Path, bytes]] = []
    seen = set()
    patterns = list(INCLUDE_PATTERNS)
    if not include_src:
        patterns = [p for p in patterns if not p.startswith("src/")]
    for pattern in patterns:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(PROJECT_ROOT)
            if _excluded(rel):
                continue
            key = rel.as_posix()
            if key in seen:
                continue
            seen.add(key)
            data = path.read_bytes()
            _scan_secrets(rel, data)
            files.append((rel, data))
    if not files:
        raise SystemExit("no files matched the bundle include patterns")
    return files


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(files: List[Tuple[Path, bytes]]) -> Tuple[bytes, bytes]:
    entries = [
        {"path": rel.as_posix(), "size": len(data), "sha256": _sha256(data)}
        for rel, data in files
    ]
    meta = {
        "format": "lazycatbot-nnue-bundle-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_root": str(PROJECT_ROOT),
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "excludes": [
            ".venv", "training/work", "temp", "checkpoints", "*.nnue", "*.pt",
            "*.npz", "datasets (*.zst/*.u16/*.u8/*.f32)", "SSH keys", "tokens",
        ],
        "files": entries,
    }
    manifest = json.dumps(meta, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    sums = "".join(f"{e['sha256']}  {e['path']}\n" for e in entries).encode("utf-8")
    return manifest, sums


def build_bundle(out_dir: Path, name: str, include_src: bool = False, dry_run: bool = False) -> Path:
    files = _collect(include_src=include_src)
    manifest, sums = _manifest(files)
    total = sum(len(d) for _, d in files) + len(manifest) + len(sums)

    print(f"project: {PROJECT_ROOT}")
    print(f"files:   {len(files)}")
    for rel, data in files:
        print(f"  {len(data):>9,}  {rel.as_posix()}")
    print(f"payload: {total:,} bytes before compression")
    if dry_run:
        print("dry run: no archive written")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = out_dir / f"{name}_{stamp}.tar.gz"
    root_name = f"{name}_{stamp}"

    def add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes, mode: int = 0o644):
        info = tarfile.TarInfo(name=arcname)
        info.size = len(data)
        info.mtime = int(time.time())
        info.mode = mode
        tf.addfile(info, io.BytesIO(data))

    with tarfile.open(archive, "w:gz", compresslevel=9) as tf:
        for rel, data in files:
            add_bytes(tf, f"{root_name}/{rel.as_posix()}", data)
        add_bytes(tf, f"{root_name}/MANIFEST.json", manifest)
        add_bytes(tf, f"{root_name}/SHA256SUMS", sums)

    digest = _sha256(archive.read_bytes())
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")

    # Verify the archive can be reopened and every payload hash matches.
    verified = 0
    with tarfile.open(archive, "r:gz") as tf:
        names = set(tf.getnames())
        for rel, data in files:
            arc = f"{root_name}/{rel.as_posix()}"
            if arc not in names:
                raise SystemExit(f"verification failed: {arc} missing")
            member = tf.extractfile(arc)
            if member is None or _sha256(member.read()) != _sha256(data):
                raise SystemExit(f"verification failed: {arc} hash mismatch")
            verified += 1
    if verified != len(files):
        raise SystemExit("verification failed: file count mismatch")

    print(f"\narchive:  {archive}")
    print(f"size:     {archive.stat().st_size:,} bytes")
    print(f"sha256:   {digest}")
    print(f"verified: {verified} files round-trip")
    print(f"checksum: {sidecar}")
    return archive


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Package the NNUE training code for a remote host.")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "temp" / "bundles"))
    ap.add_argument("--name", default="lazycatbot_nnue_bundle")
    ap.add_argument("--with-src", action="store_true",
                    help="also include src/ (needed only to rebuild the C++ extractor)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        build_bundle(Path(args.out).resolve(), args.name, args.with_src, args.dry_run)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
