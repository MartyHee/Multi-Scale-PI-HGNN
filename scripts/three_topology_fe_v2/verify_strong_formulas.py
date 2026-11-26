#!/usr/bin/env python
"""Independent formula checks for Transolver++ / LinearNO cores.

These compare the local modules to an explicit reference written from the
official source files (saved under docs/.../official_core/), including the
normalisation axes, mass denominator, per-head layout, and gradients.  They
are not official-weight equality checks; the official model/encoder cannot be
instantiated on the FE task without a full adaptation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.models.strong_baselines.linearno import LinearNOAttention  # noqa: E402
from src.models.strong_baselines.transolver import PhysicsAttention  # noqa: E402


def check_transolver(dev):
    torch.manual_seed(7)
    dim, heads, slices, B, N = 16, 2, 4, 2, 11
    attn = PhysicsAttention(dim, heads, dim // heads, slices, 0.0,
                            gumbel_mode="off", ada_temp=True).to(dev)
    x = torch.randn(B, N, dim, device=dev, requires_grad=True)
    out = attn(x)
    # explicit reference
    dh = dim // heads
    xm = attn.in_project_x(x).reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    tau = torch.clamp(attn.proj_temperature(xm) + attn.bias, min=0.01)
    w = torch.softmax(attn.in_project_slice(xm) / tau, dim=-1)
    norm = w.sum(dim=2)
    st = torch.einsum("bhnc,bhng->bhgc", xm, w) / ((norm + 1e-5)[:, :, :, None])
    q, k, v = attn.to_q(st), attn.to_k(st), attn.to_v(st)
    os = F.scaled_dot_product_attention(q, k, v)
    ox = torch.einsum("bhgc,bhng->bhnc", os, w).permute(0, 2, 1, 3).reshape(B, N, dim)
    ref = attn.to_out(ox)
    fwd = torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
    g = torch.autograd.grad(out.sum(), x)[0]
    return {"forward_equal": bool(fwd), "grad_finite": bool(torch.isfinite(g).all()),
            "grad_nonzero": bool(g.abs().sum() > 0),
            "noise_always": None}


def check_transolver_modes(dev):
    torch.manual_seed(8)
    attn = PhysicsAttention(8, 2, 4, 4, gumbel_mode="always").to(dev).train()
    x = torch.randn(1, 9, 8, device=dev)
    a = attn(x); b = attn(x)
    noisy = not torch.allclose(a, b)
    attn.eval(); c = attn(x); d = attn(x)
    noisy_eval = not torch.allclose(c, d)
    attn2 = PhysicsAttention(8, 2, 4, 4, gumbel_mode="train_only").to(dev).eval()
    e = attn2(x); f = attn2(x)
    deterministic_eval = torch.allclose(e, f)
    return {"always_noisy_train": bool(noisy), "always_noisy_eval": bool(noisy_eval),
            "train_only_deterministic_eval": bool(deterministic_eval)}


def check_linearno(dev):
    torch.manual_seed(9)
    dim, heads, M, B, N = 16, 2, 6, 2, 13
    attn = LinearNOAttention(dim, heads, dim // heads, M, 0.0, False).to(dev)
    x = torch.randn(B, N, dim, device=dev, requires_grad=True)
    out = attn(x)
    dh = dim // heads
    xm = attn.in_project_x(x).reshape(B, N, heads, dh).permute(0, 2, 1, 3)
    q, k, v = attn.to_q(xm), attn.to_k(xm), attn.to_v(xm)
    q = torch.softmax(q, dim=-1)
    k = torch.softmax(k, dim=-2)
    kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
    ox = torch.einsum("bhnd,bhdc->bhnc", q, kv)
    ox = ox.permute(0, 2, 1, 3).reshape(B, N, dim)
    ref = attn.to_out(ox)
    fwd = torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
    g = torch.autograd.grad(out.sum(), x)[0]
    return {"forward_equal": bool(fwd), "grad_finite": bool(torch.isfinite(g).all()),
            "grad_nonzero": bool(g.abs().sum() > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path("outputs") / "strong_baseline_prep_20261126" / "checks"))
    a = ap.parse_args()
    dev = torch.device("cpu")
    rep = {"schema": "qt-fe-v2-strong-formula-check-v1",
           "transolver": check_transolver(dev),
           "transolver_modes": check_transolver_modes(dev),
           "linearno": check_linearno(dev)}
    rep["all_ok"] = all([rep["transolver"]["forward_equal"], rep["transolver"]["grad_finite"],
                         rep["transolver"]["grad_nonzero"], rep["linearno"]["forward_equal"],
                         rep["linearno"]["grad_finite"], rep["linearno"]["grad_nonzero"],
                         rep["transolver_modes"]["train_only_deterministic_eval"]])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "strong_formula_checks.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
