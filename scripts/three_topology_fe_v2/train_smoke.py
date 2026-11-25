#!/usr/bin/env python
"""Capped end-to-end smoke for MLP and HGT-global on each template.

One common training core per template, three validation slices, one common
selection metric.  Test cells are never loaded.  This is not a formal run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, schema, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.evaluation import MultiSliceEvaluator, SliceBundle  # noqa: E402
from src.train.three_topology_v2 import engine, registry  # noqa: E402

TEMPLATES = ["topology1", "topology2", "topology3"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIG_ROOT / "smoke_pipeline.yaml"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default="mlp,hgt_global")
    ap.add_argument("--templates", default=",".join(TEMPLATES))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "smoke_train"))
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_candidate.json").read_text(encoding="utf-8"))
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    templates = [x.strip() for x in a.templates.split(",") if x.strip()]
    out = Path(a.out) / a.device.replace(":", "_")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "qt-fe-v2-smoke-train-v1", "device_requested": a.device,
               "models": models, "templates": templates, "runs": {}}
    dev = engine.resolve_device(a.device)
    for t in templates:
        st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / f"{t}.json")
        train_ids = protocol.ids_for(idx, prot, t, "A", "train")[: int(cfg["smoke"]["train_samples_per_template"])]
        val_slices = {
            "design_val": protocol.ids_for(idx, prot, t, "A", "val")[: int(cfg["smoke"]["val_samples_per_slice"])],
            "load_val": protocol.ids_for(idx, prot, t, "B", "val")[: int(cfg["smoke"]["val_samples_per_slice"])],
            "joint_val": protocol.ids_for(idx, prot, t, "C", "val")[: int(cfg["smoke"]["val_samples_per_slice"])],
        }
        if any(not v for v in val_slices.values()):
            raise SystemExit(f"{t}: empty validation slice")
        tr = FeV2Dataset(idx, train_ids, st, dtype=torch.float32, normalize=True)
        val_ds = {name: FeV2Dataset(idx, ids, st, dtype=torch.float32, normalize=True)
                  for name, ids in val_slices.items()}
        bundle = SliceBundle(val_ds.values())
        for model_name in models:
            seed = int(cfg["training"]["seeds"][0])
            local = json.loads(json.dumps(cfg))
            local["training"]["seed"] = seed
            local["training"]["device"] = a.device
            local["models"][model_name] = schema.model_cfg_with_dims(model_name, cfg["models"][model_name])
            engine.seed_everything(seed)
            model = registry.build_gradient_model(model_name, local["models"][model_name])
            identity = engine.build_identity(
                model_name, seed, st, local, train_ids,
                [s for ids in val_slices.values() for s in ids],
                {"samples_jsonl": idx.samples_sha256,
                 "dataset_manifest": index_mod.sha256_file(idx.root / "dataset_manifest.json")},
                "fe_v2_smoke", t)
            evaluator = MultiSliceEvaluator(list(val_ds.items()), st, max_batches=1)
            run_dir = out / t / model_name / f"seed_{seed}"
            log = engine.train_fold(model_name, model, tr, bundle, st, local, run_dir, identity,
                                    val_evaluator=evaluator)
            log["slice_selection"] = evaluator.last["slice_selection"] if evaluator.last else None
            log["slice_metrics"] = evaluator.last["slice_metrics"] if evaluator.last else None
            log["smoke_scope"] = {"train_ids": train_ids, "val_ids": val_slices,
                                  "capped_max_batches": 2, "epochs": 1,
                                  "note": "not a performance comparison; test cells not loaded"}
            (run_dir / "run_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
            summary["runs"][f"{t}/{model_name}"] = {
                "run_dir": str(run_dir), "selection_metric": log.get("final_val", {}).get("selection_metric"),
                "slice_selection": evaluator.last["slice_selection"] if evaluator.last else None,
                "completed_epochs": log.get("completed_epochs"),
                "best_epoch_zero_based": log.get("best", {}).get("epoch_zero_based"),
                "pass": log.get("pass", {}).get("all"),
                "best_sha256": log.get("best", {}).get("checkpoint_sha256"),
            }
            print(t, model_name, summary["runs"][f"{t}/{model_name}"], flush=True)
    summary["all_ok"] = bool(summary["runs"]) and all(
        v["pass"] is True and v["selection_metric"] is not None for v in summary["runs"].values())
    (out / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({"all_ok": summary["all_ok"]}, ensure_ascii=False))
    return 0 if summary["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
