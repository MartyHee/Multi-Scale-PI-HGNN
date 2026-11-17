#!/usr/bin/env python
"""Fingerprint the frozen inputs of a batch and gate runs on them.

Modes
  --hashes   print a JSON object with every identity the preflight must bind:
             code commit, data hashes, split / stats / train-config hashes and
             the batch-manifest hash.
  --check    recompute the same object and compare with PREFLIGHT_PASS.json.
             Any difference -> non-zero exit with a readable diff (a stale PASS
             must never authorise a run after inputs/config/code changed).

The split/stats identity is resolved from the frozen batch manifest, so a
job YAML only needs --batch-id + --job/--yaml path at the shell layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def native(s):
    """Map an MSYS path (/d/x) to a Windows path (D:/x) on Windows; else as-is."""
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


HERE = Path(__file__).resolve().parent
KEYS = ["samples_jsonl_sha256", "dataset_manifest_sha256", "split_sha256",
        "stats_sha256", "train_config_sha256", "batch_manifest_sha256",
        "git_commit"]


def sha(p: Path) -> str:
    """Raw byte hash for data/split/stats (matches engine identity)."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_text(p: Path) -> str:
    """Hash Git-tracked text with CRLF normalised to LF."""
    data = p.read_bytes().replace(bytes([13, 10]), bytes([10]))
    return hashlib.sha256(data).hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def manifest(batch_id: str, root: Path) -> dict:
    p = root / "remote_jobs" / "three_topology_v2" / "manifests" / f"{batch_id}.json"
    if not p.is_file():
        raise SystemExit(f"[GATE] frozen batch manifest missing: {p}\n"
                         f"Generate and commit it before preflight: "
                         f"python server_ops/three_topology_v2/_batch_manifest.py "
                         f"--job <job.yaml> --write")
    man = json.loads(p.read_text(encoding="utf-8"))
    if man.get("schema") != "qt-three-topology-batch-manifest-v2":
        raise SystemExit(f"[GATE] batch manifest schema {man.get('schema')!r} is not v2; "
                         f"regenerate with --write")
    return man


def hashes(root: Path, data_root: Path, batch_id: str) -> dict:
    man = manifest(batch_id, root)
    idx_dir = data_root / "data" / "processed" / "three_topology_graph_v1"
    ident = man.get("input_identity") or {}
    split_path = root / ident["split_source"]
    stats_path = root / ident["stats_source"]
    out = {
        "batch_id": batch_id,
        "samples_jsonl_sha256": sha(idx_dir / "samples.jsonl"),
        "dataset_manifest_sha256": sha(idx_dir / "dataset_manifest.json"),
        "split_sha256": sha(split_path),
        "stats_sha256": sha(stats_path),
        "train_config_sha256": sha_text(root / man["train_config"]),
        "batch_manifest_sha256": sha_text(root / "remote_jobs" / "three_topology_v2"
                                          / "manifests" / f"{batch_id}.json"),
        "git_commit": git_commit(root),
    }
    # the manifest itself records these hashes at freeze time
    if man.get("train_config_sha256") != out["train_config_sha256"]:
        out["manifest_config_mismatch"] = True
    if ident.get("split_sha256") != out["split_sha256"] or             ident.get("stats_sha256") != out["stats_sha256"]:
        out["manifest_input_mismatch"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashes", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--project-root", default=str(HERE.parents[1]))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pass-file", default=None)
    a = ap.parse_args()
    root = Path(native(a.project_root))
    data_root = Path(native(a.data_root))
    cur = hashes(root, data_root, a.batch_id)
    if a.hashes:
        print(json.dumps(cur, ensure_ascii=False, indent=2))
        return 0 if not (cur.get("manifest_config_mismatch") or cur.get("manifest_input_mismatch")) else 6
    if a.check:
        pf = Path(a.pass_file) if a.pass_file else (
            root / "outputs" / "three_topology_v2" / "preflight" / a.batch_id
            / "PREFLIGHT_PASS.json")
        if not pf.is_file():
            print(f"[GATE] no preflight PASS: {pf}", file=sys.stderr)
            return 4
        rec = json.loads(pf.read_text(encoding="utf-8"))
        if rec.get("result") != "PASS":
            print(f"[GATE] preflight record is not PASS: {rec.get('result')}", file=sys.stderr)
            return 4
        if rec.get("batch_id") != a.batch_id:
            print(f"[GATE] preflight batch id {rec.get('batch_id')} != {a.batch_id}", file=sys.stderr)
            return 4
        diffs = {}
        for k in KEYS:
            if rec.get(k) != cur[k]:
                diffs[k] = {"preflight": rec.get(k), "now": cur[k]}
        if cur.get("manifest_config_mismatch"):
            diffs["manifest_config_mismatch"] = True
        if cur.get("manifest_input_mismatch"):
            diffs["manifest_input_mismatch"] = True
        if diffs:
            print("[GATE] inputs/config/code changed since preflight:", file=sys.stderr)
            print(json.dumps(diffs, ensure_ascii=False, indent=2), file=sys.stderr)
            print("[GATE] rerun: CUDA_VISIBLE_DEVICES=<gpu> bash "
                  "server_ops/three_topology_v2/preflight.sh <job.yaml>", file=sys.stderr)
            return 4
        print(f"[GATE] PASS ({pf})")
        return 0
    raise SystemExit("choose --hashes or --check")


if __name__ == "__main__":
    raise SystemExit(main())
