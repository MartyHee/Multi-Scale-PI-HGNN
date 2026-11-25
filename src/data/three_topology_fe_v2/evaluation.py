"""Three-slice validation evaluator for the common checkpoint selection rule."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.train.three_topology_v2 import engine


class SliceBundle:
    def __init__(self, datasets: Iterable):
        self.datasets = list(datasets)
        self.sample_ids = [sid for ds in self.datasets for sid in ds.sample_ids]

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)


class MultiSliceEvaluator:
    def __init__(self, slices: List[Tuple[str, Any]], stats: Dict[str, Any],
                 max_batches: int | None = None):
        self.slices = slices
        self.stats = stats
        self.max_batches = max_batches
        self.last = None
        self.last_per_slice: Dict[str, Dict[str, Any]] = {}

    def __call__(self, model, batch_size: int, device, collect_predictions: bool = False):
        results = []
        self.last_per_slice = {}
        for name, ds in self.slices:
            r = engine.validate(model, ds, self.stats, batch_size, device,
                                max_batches=self.max_batches,
                                collect_predictions=collect_predictions)
            results.append((name, r))
            self.last_per_slice[name] = r
        vals = [r["selection_metric"] for _, r in results
                if r["selection_metric"] is not None]
        selection = float(sum(vals) / len(vals)) if vals else None
        ew = [r["entity_weighted_norm_mse"] for _, r in results
              if r["entity_weighted_norm_mse"] is not None]
        out = {
            "selection_metric": selection,
            "entity_weighted_norm_mse": float(sum(ew) / len(ew)) if ew else None,
            "n_batches": sum(r["n_batches"] for _, r in results),
            "slice_selection": {name: r["selection_metric"] for name, r in results},
            "slice_metrics": {name: r["metrics"] for name, r in results},
            "slice_entity_weighted_norm_mse": {
                name: r["entity_weighted_norm_mse"] for name, r in results},
            "metrics": {"slice_metrics": {name: r["metrics"] for name, r in results}},
            "exported": [x for _, r in results for x in r.get("exported", [])],
        }
        self.last = out
        return out

    def write_slice_artifacts(self, out_dir, run_label: str = "best") -> Dict[str, Any]:
        """Write per-slice metrics and predictions from the last evaluation only.

        The last evaluation must be the final one after loading the best
        checkpoint; the run log records the checkpoint hash so the provenance
        is auditable.  Writing is explicit so a stale evaluation cannot leak in.
        """
        import json
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        records = {}
        for name, r in self.last_per_slice.items():
            pred_path = out / f"val_predictions_{name}.json"
            pred_path.write_text(json.dumps(r.get("exported", []), ensure_ascii=False),
                                 encoding="utf-8")
            records[name] = {
                "checkpoint_source": run_label,
                "n_batches": r["n_batches"],
                "selection_metric": r["selection_metric"],
                "entity_weighted_norm_mse": r["entity_weighted_norm_mse"],
                "metrics": r["metrics"],
                "predictions_file": pred_path.name,
                "n_predictions": len(r.get("exported", [])),
            }
        agg = {
            "checkpoint_source": run_label,
            "selection_metric_equal_mean": (
                float(sum(v["selection_metric"] for v in records.values()
                          if v["selection_metric"] is not None)
                      / max(1, sum(1 for v in records.values()
                                   if v["selection_metric"] is not None)))),
            "slices": records,
        }
        (out / "slice_metrics.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        return agg
