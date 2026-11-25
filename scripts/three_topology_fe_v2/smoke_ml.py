#!/usr/bin/env python
"""Small Ridge/HistGB fit/predict/serialization check on the common graph view."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.three_topology_fe_v2 import index as index_mod, paths, protocol, stats as stats_mod  # noqa: E402
from src.data.three_topology_fe_v2.dataset import FeV2Dataset  # noqa: E402
from src.data.three_topology_fe_v2.ml_adapter import collect  # noqa: E402

MODELS = ["ridge", "hist_gb"]


def fit(name, cfg, X, Y):
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(cfg.get("alpha", 1.0))).fit(X, Y)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return [HistGradientBoostingRegressor(
        max_iter=int(cfg.get("max_iter", 10)),
        learning_rate=float(cfg.get("learning_rate", 0.1)),
        max_depth=cfg.get("max_depth"),
        early_stopping=False,
        random_state=int(cfg.get("seed", 20261125)),
    ).fit(X, Y[:, c]) for c in range(Y.shape[1])]


def predict(name, model, X):
    if name == "ridge":
        return model.predict(X)
    return np.stack([m.predict(X) for m in model], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIG_ROOT / "smoke_pipeline.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT / "smoke_ml"))
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    idx = index_mod.load_index()
    prot = json.loads((paths.PROCESSED_ROOT / "splits" / "protocol_candidate.json").read_text(encoding="utf-8"))
    st = stats_mod.load_stats(paths.PROCESSED_ROOT / "stats" / "topology1.json")
    train_ids = [s for s, v in prot["roles"].items()
                 if idx.by_id[s]["topology_token"] == "topology1" and v["A"] == "train"][: int(cfg["smoke"]["ml_train_samples"])]
    val_ids = [s for s, v in prot["roles"].items()
               if idx.by_id[s]["topology_token"] == "topology1" and v["A"] == "val"][: int(cfg["smoke"]["ml_val_samples"])]
    tr = FeV2Dataset(idx, train_ids, st, normalize=True)
    va = FeV2Dataset(idx, val_ids, st, normalize=True)
    Xn, Yn, Xb, Yb = collect(tr, train_ids)
    Vn, Vyn, Vb, Vyb = collect(va, val_ids)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "qt-fe-v2-ml-smoke-v1", "train_samples": len(train_ids),
              "val_samples": len(val_ids), "input_dims": {"node": int(Xn.shape[1]),
              "beam": int(Xb.shape[1])}, "models": {}}
    inv_d = st["arrays"]["y_disp"]; inv_f = st["arrays"]["y_force"]
    for name in MODELS:
        mc = dict(cfg["ml"][name]); mc["seed"] = 20261125
        node = fit(name, mc, Xn, Yn); beam = fit(name, mc, Xb, Yb)
        pn = predict(name, node, Vn); pb = predict(name, beam, Vb)
        pn_phys = pn * np.asarray(inv_d["scale"]) + np.asarray(inv_d["mean"])
        pb_phys = pb * np.asarray(inv_f["scale"]) + np.asarray(inv_f["mean"])
        path = out / f"ml_{name}.pkl"
        with path.open("wb") as fh:
            pickle.dump({"node": node, "beam": beam, "config": mc}, fh)
        with path.open("rb") as fh:
            loaded = pickle.load(fh)
        n2 = predict(name, loaded["node"], Vn[:2]); b2 = predict(name, loaded["beam"], Vb[:2])
        equal = bool(np.allclose(n2, pn[:2], rtol=1e-12, atol=1e-12)
                     and np.allclose(b2, pb[:2], rtol=1e-12, atol=1e-12))
        report["models"][name] = {
            "node_pred_shape": list(pn.shape), "beam_pred_shape": list(pb.shape),
            "pred_finite": bool(np.isfinite(pn_phys).all() and np.isfinite(pb_phys).all()),
            "serialization_roundtrip_equal": equal,
            "model_path": str(path),
        }
        print(name, report["models"][name], flush=True)
    report["all_ok"] = all(v["pred_finite"] and v["serialization_roundtrip_equal"]
                           and v["node_pred_shape"][1] == 6 and v["beam_pred_shape"][1] == 12
                           for v in report["models"].values())
    (out / "smoke_ml.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"all_ok": report["all_ok"]}, ensure_ascii=False))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
