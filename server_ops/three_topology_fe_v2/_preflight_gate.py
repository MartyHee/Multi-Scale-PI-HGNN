#!/usr/bin/env python
"""Fingerprint the frozen inputs of an FE v2 batch and gate formal runs.

The PASS record binds: frozen manifest hash, current job-YAML/effective-param
identity, code hash, train-config hash, and the input identity
(samples/manifest/schema/protocol/ledger/split-manifest/stats).  Any drift
invalidates the old PASS.

Modes
  --hashes     print the current identity as JSON
  --check      recompute and compare with PREFLIGHT_PASS.json (exit 4 on drift)
  --write      write a PASS record (used by preflight.sh after its checks)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


class GateError(RuntimeError):
    code = 4


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


HERE = Path(__file__).resolve().parent
MODEL_ROOT = Path(native(os.environ.get("PROJECT_ROOT") or HERE.parents[1]))
sys.path.insert(0, str(HERE))


def sha_raw(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def preflight_root() -> Path:
    default = MODEL_ROOT / "outputs" / "three_topology_fe_v2" / "preflight"
    return Path(native(os.environ.get("THREE_TOPO_PREFLIGHT_ROOT") or default))


def _task_tuple(t: dict):
    return (t["worker"], t["topology"], t["run"], t["model"], str(t["seed"]),
            t["device"], t["kind"], t["job_file"], t["job_file_sha256"],
            t["epochs"], t["max_batches"], t["resume_from"],
            t["model_config_sha256"])


def compute_hashes(project_root: Path, batch_id: str) -> dict:
    import _batch_manifest as bm  # noqa: PLC0415
    frozen = bm.read_manifest(batch_id)
    fresh = bm.scan(batch_id)
    diffs = {}
    for key in ("train_config", "train_config_sha256", "training_effective",
                "allow_truncation", "input_identity", "code_sha256", "n_tasks"):
        if frozen.get(key) != fresh.get(key):
            diffs[key] = True
    if {_task_tuple(t) for t in frozen["tasks"]} != {_task_tuple(t) for t in fresh["tasks"]}:
        diffs["tasks"] = True
    input_identity = fresh["input_identity"]
    cur = {
        "batch_id": batch_id,
        "manifest_sha256": sha_raw(bm.manifest_path(batch_id)),
        "code_sha256": fresh["code_sha256"],
        "git_commit": fresh["git_commit"],
        "train_config_sha256": fresh["train_config_sha256"],
        "input_identity_sha256": hash_json(input_identity),
        "samples_jsonl_sha256": input_identity["samples_jsonl_sha256"],
        "dataset_manifest_sha256": input_identity["dataset_manifest_sha256"],
        "feature_schema_sha256": input_identity["feature_schema_sha256"],
        "protocol_sha256": input_identity["protocol_sha256"],
        "role_ledger_sha256": input_identity["role_ledger_sha256"],
        "split_manifest_sha256": input_identity["split_manifest_sha256"],
        "stats_sha256": input_identity["stats_sha256"],
        "n_tasks": fresh["n_tasks"],
        "task_identity_sha256": hash_json(sorted(_task_tuple(t) for t in fresh["tasks"])),
        "manifest_current_ok": not diffs,
        "manifest_diffs": sorted(diffs),
    }
    return cur


def pass_path(batch_id: str) -> Path:
    return preflight_root() / batch_id / "PREFLIGHT_PASS.json"


def check_pass(project_root: Path, batch_id: str, file: Path | None = None) -> dict:
    path = Path(file) if file else pass_path(batch_id)
    if not path.is_file():
        raise GateError(f"preflight PASS missing: {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("result") != "PASS" or not rec.get("formal_ready"):
        raise GateError(f"preflight record is not a formal-ready PASS: {path}")
    if rec.get("batch_id") != batch_id:
        raise GateError(f"preflight batch {rec.get('batch_id')!r} != {batch_id!r}")
    cur = compute_hashes(project_root, batch_id)
    if not cur["manifest_current_ok"]:
        raise GateError(f"frozen manifest is stale relative to current jobs/files: "
                        f"{cur['manifest_diffs']}")
    old = rec.get("hashes") or {}
    diffs = {k: {"preflight": old.get(k), "now": v}
             for k, v in cur.items() if old.get(k) != v}
    if diffs:
        raise GateError("inputs/config/code changed since preflight: "
                        + json.dumps(diffs, ensure_ascii=False))
    return rec


def write_pass(project_root: Path, batch_id: str, checked, device: str,
               visible: int) -> dict:
    cur = compute_hashes(project_root, batch_id)
    if not cur["manifest_current_ok"]:
        raise GateError(f"cannot write PASS: frozen manifest stale: {cur['manifest_diffs']}")
    uniq, seen = [], set()
    for x in checked:
        key = tuple(x)
        if key not in seen:
            seen.add(key)
            uniq.append(list(key))
    rec = {
        "result": "PASS", "formal_ready": True, "batch_id": batch_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device, "visible_device_count": int(visible),
        "checked": uniq,
        "hashes": cur,
    }
    path = pass_path(batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return rec


def main() -> int:
    try:
        sys.stdout.reconfigure(newline="\n")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--hashes", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--project-root", default=str(MODEL_ROOT))
    ap.add_argument("--pass-file", default=None)
    ap.add_argument("--checked", action="append", default=[])
    ap.add_argument("--device", default="")
    ap.add_argument("--visible", type=int, default=0)
    a = ap.parse_args()
    root = Path(native(a.project_root))
    try:
        if a.hashes:
            cur = compute_hashes(root, a.batch_id)
            print(json.dumps(cur, ensure_ascii=False, indent=2))
            return 0 if cur["manifest_current_ok"] else 6
        if a.check:
            pf = Path(native(a.pass_file)) if a.pass_file else None
            rec = check_pass(root, a.batch_id, pf)
            print(f"[GATE] PASS ({pf or pass_path(a.batch_id)})")
            print(json.dumps({"checked": rec.get("checked")}, ensure_ascii=False))
            return 0
        if a.write:
            checked = []
            for item in a.checked:
                parts = item.split("/", 1)
                if len(parts) != 2:
                    raise GateError(f"--checked expects topology/model, got {item!r}")
                checked.append(tuple(parts))
            rec = write_pass(root, a.batch_id, checked, a.device, a.visible)
            print(f"[GATE] wrote {pass_path(a.batch_id)}")
            print(json.dumps({"checked": rec["checked"], "hashes": rec["hashes"]},
                             ensure_ascii=False, indent=2))
            return 0
    except GateError as exc:
        print(f"[GATE] {exc}", file=sys.stderr)
        return 4
    raise SystemExit("choose --hashes, --check or --write")


if __name__ == "__main__":
    raise SystemExit(main())
