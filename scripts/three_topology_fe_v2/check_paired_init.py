#!/usr/bin/env python
"""Paired-initialisation checks for the 36-task mechanism batch.

Checks:
  1. same run seed -> HGT and HGT-global shared tensors bit-identical;
  2. different run seeds -> shared tensors differ;
  3. construction order does not change shared or global tensors;
  4. repeated construction is reproducible;
  5. building a model does not advance the ambient torch RNG (shuffle/dropout);
  6. load_state_dict after construction restores the checkpoint exactly;
  7. capacity-matched HGT is deterministic per seed and differs across seeds.

No data, training, test or server access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402

from src.data.three_topology_fe_v2 import paths, schema  # noqa: E402
from src.train.three_topology_v2 import registry  # noqa: E402


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build(name, cfg, seed):
    mcfg = schema.task_model_cfg(name, cfg, "topology1", seed)
    return registry.build_gradient_model(name, mcfg)


def shared(state):
    return {k: v for k, v in state.items()
            if not k.startswith(("global_cell.", "global_bcast.", "fusion."))}


def global_only(state):
    return {k: v for k, v in state.items()
            if k.startswith(("global_cell.", "global_bcast.", "fusion."))}


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def differs(a, b, keys=None):
    keys = keys or list(a)
    return any(not torch.equal(a[k], b[k]) for k in keys if k in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(paths.CONFIG_ROOT / "strong_baselines.yaml"))
    ap.add_argument("--out", default=str(paths.OUTPUT_ROOT.parent / "strong_baseline_prep_20261126" / "checks"))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    s1, s2 = 20261117, 20261118
    rep = {"schema": "qt-fe-v2-paired-init-v1", "seeds": [s1, s2], "checks": {}}

    # 1/2. same/different seed
    torch.manual_seed(1); h1 = build("hgt", cfg, s1)
    torch.manual_seed(2); g1 = build("hgt_global", cfg, s1)
    torch.manual_seed(3); h2 = build("hgt", cfg, s2)
    torch.manual_seed(4); g2 = build("hgt_global", cfg, s2)
    rep["checks"]["same_seed_shared_equal"] = same(shared(h1.state_dict()), shared(g1.state_dict()))
    rep["checks"]["different_seed_shared_differ"] = differs(shared(h1.state_dict()), shared(h2.state_dict()))
    rep["checks"]["different_seed_global_differ"] = differs(global_only(g1.state_dict()), global_only(g2.state_dict()))

    # 3. construction order
    torch.manual_seed(77); g_first = build("hgt_global", cfg, s1)
    torch.manual_seed(77); h_after = build("hgt", cfg, s1)
    torch.manual_seed(77); h_first = build("hgt", cfg, s1)
    torch.manual_seed(77); g_after = build("hgt_global", cfg, s1)
    rep["checks"]["order_shared_equal"] = same(shared(g_first.state_dict()), shared(h_first.state_dict()))
    rep["checks"]["order_global_equal"] = same(global_only(g_first.state_dict()), global_only(g_after.state_dict()))
    rep["checks"]["same_seed_global_equal"] = same(global_only(g1.state_dict()), global_only(g_after.state_dict()))

    # 4. repeat
    torch.manual_seed(9); g_a = build("hgt_global", cfg, s1)
    torch.manual_seed(9); g_b = build("hgt_global", cfg, s1)
    rep["checks"]["repeat_reproducible"] = same(g_a.state_dict(), g_b.state_dict())

    # 5. RNG pollution
    torch.manual_seed(12345)
    st0 = torch.get_rng_state().clone()
    build("hgt", cfg, s1)
    build("hgt_global", cfg, s1)
    build("hgt_capmatch", cfg, s1)
    st1 = torch.get_rng_state().clone()
    rep["checks"]["ambient_rng_unchanged"] = torch.equal(st0, st1)
    torch.manual_seed(12345); r0 = torch.rand(8)
    torch.manual_seed(12345); build("hgt_global", cfg, s1); r1 = torch.rand(8)
    rep["checks"]["post_build_draws_equal"] = torch.equal(r0, r1)

    # 6. checkpoint load
    m = build("hgt_global", cfg, s1)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    m.load_state_dict(sd)
    rep["checks"]["checkpoint_load_restores"] = same(sd, m.state_dict())

    # 7. capmatch determinism
    torch.manual_seed(5); c1 = build("hgt_capmatch", cfg, s1)
    torch.manual_seed(6); c2 = build("hgt_capmatch", cfg, s1)
    torch.manual_seed(7); c3 = build("hgt_capmatch", cfg, s2)
    rep["checks"]["capmatch_deterministic"] = same(c1.state_dict(), c2.state_dict())
    rep["checks"]["capmatch_differs_by_seed"] = differs(c1.state_dict(), c3.state_dict())
    rep["init_seed_records"] = {
        "hgt": schema.task_model_cfg("hgt", cfg, "topology1", s1)["init_seed"],
        "hgt_global": schema.task_model_cfg("hgt_global", cfg, "topology1", s1)["init_seed"],
        "hgt_capmatch": schema.task_model_cfg("hgt_capmatch", cfg, "topology1", s1)["init_seed"],
        "global_init_seed": g1.global_init_seed if hasattr(g1, "global_init_seed") else None,
    }
    rep["all_ok"] = all(rep["checks"].values())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "paired_init_checks.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
