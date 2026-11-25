#!/usr/bin/env python
"""Validate an FE v2 resume job against the frozen batch manifest.

A resume job is the only YAML allowed outside ``remote_jobs/three_topology_fe_v2``.
It cannot introduce a new model/seed, cannot point at another task's checkpoint,
and cannot reuse the original worker/attempt directory.  The target attempt is
``original_attempt + 1`` and must not exist; the original directory stays
read-only.

Exit 0 only if every task passes.  Writes a JSON record and a TSV map used by
run_job.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCHEMA = "qt-fe-v2-batch-manifest-v1"


def native(s):
    p = str(s)
    m = re.match(r"^/([A-Za-z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (os.name == "nt" and m) else p


def sha_bytes(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hash_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def die(msg: str) -> None:
    print(f"[RESUME CHECK ERROR] {msg}", file=sys.stderr)
    sys.exit(2)


def load_job_module():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _job_parse  # noqa: PLC0415
    return _job_parse


def run_suffix(model: str, seed: str) -> str:
    return model if model == "ridge" else f"{model}_seed{seed}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", ""))
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--map-out", default=None)
    a = ap.parse_args()
    root = Path(native(a.project_root or Path(__file__).resolve().parents[2]))
    package = Path(native(a.package_root))
    batch = a.batch_id
    jp = load_job_module()
    spec = jp.load(Path(native(a.job)))
    if spec["batch_id"] != batch:
        die(f"job batch_id {spec['batch_id']!r} != requested {batch!r}")

    man_path = Path(native(os.environ.get("THREE_TOPO_MANIFEST_ROOT")
                           or (root / "remote_jobs" / "three_topology_fe_v2" / "manifests"))) / f"{batch}.json"
    if not man_path.is_file():
        die(f"frozen manifest missing: {man_path}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("schema") != SCHEMA:
        die(f"manifest schema {man.get('schema')!r} is not {SCHEMA!r}")
    if spec["train_config"] != man["train_config"]:
        die(f"train_config differs from frozen manifest: {spec['train_config']!r}")
    tc_path = root / man["train_config"]
    import yaml
    cfg = yaml.safe_load(tc_path.read_text(encoding="utf-8"))
    tr = cfg["training"]

    formal_root = Path(native(os.environ.get("THREE_TOPO_FORMAL_OUTPUT_ROOT")
                              or (root / "outputs" / "three_topology_fe_v2" / "formal"))) / batch
    prot = json.loads((Path(native(os.environ.get("THREE_TOPO_PROCESSED_ROOT")
                                  or (root / "processed" / "three_topology_fe_v2_9216")))
                       / "splits" / "protocol_v1.json").read_text(encoding="utf-8"))

    orig_by_key = {}
    for t in man["tasks"]:
        if t["kind"] != "gradient":
            continue
        orig_by_key.setdefault((t["topology"], t["model"], str(t["seed"])), []).append(t)

    rows, problems, maps = [], [], []
    if not spec["tasks"]:
        die("resume job has no tasks")
    for i, t in enumerate(spec["tasks"]):
        tag = f"tasks[{i}]"
        if t["device"] != "gpu":
            die(f"{tag}: resume requires device=gpu")
        if t["resume_from"] in (None, "-"):
            die(f"{tag}: non-resume task in a resume YAML")
        if t["model"] == "ridge" or t["model"] == "hist_gb":
            die(f"{tag}: ML tasks do not support epoch resume; rerun in a new attempt")
        key = (t["topology"], t["model"], str(t["seed"]))
        if key not in orig_by_key or len(orig_by_key[key]) != 1:
            die(f"{tag}: ({t['topology']}, {t['model']}, seed {t['seed']}) does not "
                f"identify exactly one original task in batch {batch}")
        orig = orig_by_key[key][0]
        if t["worker"] == orig["worker"]:
            die(f"{tag}: resume needs a NEW worker directory; {t['worker']!r} is the original")
        ckpt = Path(native(t["resume_from"]))
        if not ckpt.is_absolute():
            ckpt = (root / ckpt).resolve()
        else:
            ckpt = ckpt.resolve()
        try:
            rel = ckpt.relative_to(formal_root.resolve())
        except ValueError:
            die(f"{tag}: resume_from is outside {formal_root}: {ckpt}")
        parts = rel.parts
        if (len(parts) != 6 or parts[0] != orig["topology"] or parts[1] != orig["worker"]
                or parts[2] != orig["run"] or not re.match(r"^attempt_\d+$", parts[3])
                or parts[4] != "checkpoints" or parts[5] not in ("best.pt", "last.pt")):
            die(f"{tag}: resume_from must be <topology>/<orig_worker>/<orig_run>/"
                f"attempt_NN/checkpoints/(best|last).pt, got {rel.as_posix()}")
        orig_attempt = int(parts[3].split("_")[1])
        target_attempt = orig_attempt + 1
        target_dir = formal_root / t["topology"] / t["worker"] / orig["run"] / f"attempt_{target_attempt:02d}"
        if target_dir.exists():
            die(f"{tag}: target attempt already exists: {target_dir}")
        if not ckpt.is_file():
            die(f"{tag}: checkpoint not found: {ckpt}")

        try:
            import torch
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            die(f"{tag}: cannot load checkpoint {ckpt}: {exc!r}")
        ident = ck.get("identity") or {}

        # Recompute the train/val identity with the same rule as the entry.
        sys.path.insert(0, str(root))
        from src.data.three_topology_fe_v2 import index as index_mod, protocol as protocol_mod
        idx = index_mod.load_index(package)
        train_ids = protocol_mod.ids_for(idx, prot, t["topology"], "A", "train")
        val_ids = []
        for p, role in (("A", "val"), ("B", "val"), ("C", "val")):
            val_ids.extend(protocol_mod.ids_for(idx, prot, t["topology"], p, role))
        expected = {
            "model_name": t["model"], "seed": int(t["seed"]),
            "split_id": "protocol_v1", "fold_id": t["topology"],
            "outer_fold": None, "inner_fold": None,
            "stats_sha256": man["input_identity"]["stats_sha256"][t["topology"]],
            "samples_jsonl_sha256": man["input_identity"]["samples_jsonl_sha256"],
            "dataset_manifest_sha256": man["input_identity"]["dataset_manifest_sha256"],
            "train_sample_list_sha256": hash_json(sorted(train_ids)),
            "val_sample_list_sha256": hash_json(sorted(val_ids)),
            "batch_size": int(tr["batch_size"]),
            "lr": float(tr["lr"]),
            "weight_decay": float(tr.get("weight_decay", 0.0)),
            "max_train_batches_per_epoch": tr.get("max_train_batches_per_epoch"),
            "model_config_sha256": orig["model_config_sha256"],
            "protocol_id": "protocol_v1",
            "protocol_sha256": man["input_identity"]["protocol_sha256"],
            "role_ledger_sha256": man["input_identity"]["role_ledger_sha256"],
            "split_manifest_sha256": man["input_identity"]["split_manifest_sha256"],
            "feature_schema_sha256": man["input_identity"]["feature_schema_sha256"],
            "read_path": man["input_identity"]["read_path"],
            "design_cache_sha256": man["input_identity"]["design_cache_sha256"][t["topology"]],
            "design_cache_meta_sha256": man["input_identity"]["design_cache_meta_sha256"][t["topology"]],
            "shared_geometry_sha256": man["input_identity"]["shared_geometry_sha256"][t["topology"]],
            "field_loads_sha256": man["input_identity"]["field_loads_sha256"][t["topology"]],
            "topology": t["topology"],
        }
        diffs = {k: {"checkpoint": ident.get(k), "expected": v}
                 for k, v in expected.items() if ident.get(k) != v}
        if diffs:
            problems.append({"task": tag, "diffs": diffs})
            continue
        maps.append({"worker": t["worker"], "topology": t["topology"], "run": orig["run"],
                     "model": t["model"], "seed": str(t["seed"]),
                     "orig_worker": orig["worker"], "orig_run": orig["run"],
                     "orig_attempt": orig_attempt, "target_attempt": target_attempt,
                     "checkpoint": str(ckpt), "checkpoint_sha256": sha_bytes(ckpt)})
        rows.append({"task": tag, "worker": t["worker"],
                     "original": f"{orig['worker']}/{orig['run']}/attempt_{orig_attempt:02d}",
                     "target_attempt": target_attempt,
                     "checkpoint": str(ckpt), "checkpoint_sha256": sha_bytes(ckpt),
                     "identity_ok": True})

    if problems:
        print(json.dumps({"ok": False, "problems": problems}, ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 1

    rec = {"schema": "qt-fe-v2-resume-check-v1", "batch_id": batch, "job": str(a.job),
           "manifest_sha256": sha_bytes(man_path), "ok": True,
           "n_resume_tasks": len(rows), "tasks": rows}
    if a.out:
        op = Path(native(a.out)); op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    if a.map_out:
        mp = Path(native(a.map_out)); mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8", newline="\n") as f:
            for m in maps:
                f.write("\t".join([m["worker"], m["topology"], m["run"], m["model"],
                                   m["seed"], m["orig_worker"], m["orig_run"],
                                   str(m["orig_attempt"]), str(m["target_attempt"]),
                                   m["checkpoint"], m["checkpoint_sha256"]]) + "\n")
    print(json.dumps({"ok": True, "n_resume_tasks": len(rows),
                      "original_runs": [r["original"] for r in rows]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
