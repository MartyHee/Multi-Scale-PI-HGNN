#!/usr/bin/env python
"""Restore selected members of a frozen artifact archive on demand.

The 2026-11-25 first-batch archive is the immutable source of record; the
24.7 GB extraction is a convenience copy.  This helper restores only what a
given analysis needs, verifies every restored file against the archive's
``SHA256SUMS``, and refuses unsafe paths.

Examples
  # manifests + summary only (small)
  python archive_source.py --archive remote_artifacts/fe_v2_fv2_first_20261125_01_artifacts.tar.gz \
      --out outputs/restore_meta

  # one run, without the duplicate combined predictions and without checkpoints
  python archive_source.py --archive ... --out outputs/restore_one \
      --runs topology1/t1_mlp/mlp_seed20261117

  # full analysis copy (24.7 GB; checkpoints optional +0.2 GB)
  python archive_source.py --archive ... --out outputs/restore_all --all-runs --include-checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

TOP_LEVEL = ("manifests/", "summary/", "configs/", "processed/", "code/", "SHA256SUMS")


def safe_name(name: str) -> str | None:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        return None
    return str(p).lstrip("./")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="substrings of run paths under batch/ to include")
    ap.add_argument("--all-runs", action="store_true")
    ap.add_argument("--include-checkpoints", action="store_true")
    ap.add_argument("--include-combined", action="store_true",
                    help="also restore the duplicate run-root val_predictions.json")
    ap.add_argument("--verify", action="store_true", default=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f"[restore] output directory is not empty: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(a.archive)

    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for m in tar.getmembers():
            name = safe_name(m.name)
            if name is None:
                print(f"[restore] refusing unsafe member: {m.name}", file=sys.stderr)
                return 3
            if not m.isfile():
                continue
            if name == "SHA256SUMS" or any(name.startswith(t) for t in TOP_LEVEL):
                members.append((m, name)); continue
            if not name.startswith("batch/"):
                continue
            if not (a.all_runs or (a.runs and any(r.strip("/") in name for r in a.runs))):
                continue
            if not a.include_combined and name.endswith("/val_predictions.json") \
                    and name.count("/") == 4:
                continue
            if not a.include_checkpoints and "/checkpoints/" in name:
                continue
            members.append((m, name))

        sums = {}
        sums_member = next((m for m, n in members if n == "SHA256SUMS"), None)
        if sums_member is not None:
            sums_bytes = tar.extractfile(sums_member).read()
            for line in sums_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                digest, rel = line.split(None, 1)
                sums[safe_name(rel.strip())] = digest

        restored = 0
        verified = 0
        total_bytes = 0
        for m, name in members:
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            fh = tar.extractfile(m)
            data = fh.read()
            dest.write_bytes(data)
            restored += 1
            total_bytes += len(data)
            if a.verify and name in sums:
                if sha_bytes(data) != sums[name]:
                    print(f"[restore] SHA256 mismatch: {name}", file=sys.stderr)
                    return 4
                verified += 1

    manifest = {
        "schema": "qt-fe-v2-archive-restore-v1",
        "archive": str(archive), "archive_sha256": sha_file(archive),
        "out": str(out), "n_restored": restored, "n_verified": verified,
        "bytes_restored": total_bytes, "include_checkpoints": a.include_checkpoints,
        "include_combined": a.include_combined,
        "runs": a.runs, "all_runs": a.all_runs,
    }
    (out / "restore_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_restored", "n_verified",
                                               "bytes_restored", "out")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
