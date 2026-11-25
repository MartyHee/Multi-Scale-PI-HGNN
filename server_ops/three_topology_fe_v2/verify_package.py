#!/usr/bin/env python
"""Byte-level package identity check (never decodes test labels).

Default: verify the index, manifest, registries and 9,216 cell coverage.
--full: hash every file in ``dataset_manifest.json:files`` once.  This reads the
whole 19.8 GB package and must be run at most once per upload, not per job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.package_root)
    man = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    rec = {"package_root": str(root), "status": man.get("status"),
           "samples": man.get("samples"), "ok": True, "problems": []}
    if man.get("status") != "ready":
        rec["problems"].append("manifest status is not ready")
        rec["ok"] = False
    if sha256(root / "samples.jsonl") != man["samples_sha256"]:
        rec["problems"].append("samples.jsonl hash mismatch")
        rec["ok"] = False
    rows = [json.loads(line) for line in
            (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cells = {(r["topology_token"], r["design_id"], r["field_id"]) for r in rows}
    rec["index_rows"] = len(rows)
    rec["unique_cells"] = len(cells)
    if len(rows) != 9216 or len(cells) != 9216:
        rec["problems"].append(f"index coverage rows={len(rows)} cells={len(cells)}")
        rec["ok"] = False
    for topo in ("topology1", "topology2", "topology3"):
        sub = [r for r in rows if r["topology_token"] == topo]
        rec.setdefault("by_topology", {})[topo] = {
            "samples": len(sub), "designs": len({r["design_id"] for r in sub}),
            "fields": len({r["field_id"] for r in sub}),
        }
        if (len(sub), rec["by_topology"][topo]["designs"],
                rec["by_topology"][topo]["fields"]) != (3072, 48, 64):
            rec["problems"].append(f"{topo} count mismatch")
            rec["ok"] = False
    for key in ("fields_registry", "designs_registry", "provisional_holdout"):
        if not (root / man[key]).is_file():
            rec["problems"].append(f"missing {key}: {man[key]}")
            rec["ok"] = False

    if a.full:
        t0 = time.time()
        files = man.get("files") or []
        rec["full_file_count"] = len(files)
        bad = []
        total = 0
        for i, f in enumerate(files, 1):
            p = root / f["path"]
            if not p.is_file():
                bad.append({"path": f["path"], "reason": "missing"})
                continue
            total += p.stat().st_size
            if p.stat().st_size != f.get("bytes") or sha256(p) != f.get("sha256"):
                bad.append({"path": f["path"], "reason": "size_or_hash"})
            if i % 500 == 0:
                print(f"[verify] {i}/{len(files)} files, {total/1e9:.2f} GB", flush=True)
        rec["full_bad"] = bad[:50]
        rec["full_bad_count"] = len(bad)
        rec["full_bytes"] = total
        rec["full_seconds"] = round(time.time() - t0, 1)
        if bad:
            rec["ok"] = False
            rec["problems"].append(f"{len(bad)} full-package files failed")

    if a.out:
        op = Path(a.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
