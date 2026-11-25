"""Candidate group-level protocol from the accepted G2 holdout groups.

The protocol is a cell-level candidate, not an approved formal training plan.
A/B/C share the common training core but use disjoint validation/test cells.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

TOPO = ["topology1", "topology2", "topology3"]


class ProtocolError(RuntimeError):
    pass


def build_protocol(plan: Dict, index) -> Dict:
    roles: Dict[str, Dict[str, str]] = {}
    groups: Dict[str, Dict[str, Dict[str, str]]] = {}
    for t in TOPO:
        dg, fg = plan["design_groups"][t], plan["field_groups"][t]
        Dtr = set(dg["seen_g1"] + dg["seen_old"] + dg["blind_3"])
        Dval = set(dg["blind_2"]); Dtest = set(dg["blind_1"])
        Ftr = set(fg["seen_g1"] + fg["train"])
        Fval = set(fg["blind_validation"]); Ftest = set(fg["blind_test"])
        if len(Dtr | Dval | Dtest) != 48 or len(Ftr | Fval | Ftest) != 64:
            raise ProtocolError(f"{t}: group sizes do not cover 48/64")
        dgrp = {d: g for g, ids in dg.items() for d in ids}
        fgrp = {f: g for g, ids in fg.items() for f in ids}
        groups[t] = {"design": dgrp, "field": fgrp}
        for r in index.rows:
            if r["topology_token"] != t:
                continue
            d, f = r["design_id"], r["field_id"]
            a = "train" if d in Dtr and f in Ftr else ("val" if d in Dval and f in Ftr else
                 ("test" if d in Dtest and f in Ftr else "blocked"))
            b = "train" if d in Dtr and f in Ftr else ("val" if d in Dtr and f in Fval else
                 ("test" if d in Dtr and f in Ftest else "blocked"))
            c = "train" if d in Dtr and f in Ftr else ("val" if d in Dval and f in Fval else
                 ("test" if d in Dtest and f in Ftest else "blocked"))
            roles[r["sample_id"]] = {"A": a, "B": b, "C": c}
    if len(roles) != 9216:
        raise ProtocolError(f"protocol roles cover {len(roles)} != 9216")
    return {"schema": "qt-fe-v2-protocol-v1", "not_approved": True,
            "source": "g2_holdout_plan.json seed 20261120; cell allocation revised",
            "seed": int(plan["seed"]),
            "groups": groups, "roles": roles}


def role_counts(protocol: Dict) -> Dict:
    out = {}
    for p in "ABC":
        out[p] = {r: sum(1 for v in protocol["roles"].values() if v[p] == r)
                  for r in ("train", "val", "test", "blocked")}
        if sum(out[p].values()) != 9216:
            raise ProtocolError(f"{p}: role sum != 9216")
        train = {sid for sid, v in protocol["roles"].items() if v[p] == "train"}
        val = {sid for sid, v in protocol["roles"].items() if v[p] == "val"}
        test = {sid for sid, v in protocol["roles"].items() if v[p] == "test"}
        if train & val or train & test or val & test:
            raise ProtocolError(f"{p}: train/val/test overlap")
    train_sets = {p: {sid for sid, v in protocol["roles"].items() if v[p] == "train"} for p in "ABC"}
    if not (train_sets["A"] == train_sets["B"] == train_sets["C"]):
        raise ProtocolError("A/B/C do not share a common training core")
    for r in ("val", "test"):
        sets = [ {sid for sid, v in protocol["roles"].items() if v[p] == r} for p in "ABC" ]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ProtocolError(f"{r} cells overlap across protocols")
    return out


def ids_for(index, protocol: Dict, topology: str, protocol_name: str, role: str) -> List[str]:
    return [r["sample_id"] for r in index.rows
            if r["topology_token"] == topology
            and protocol["roles"][r["sample_id"]][protocol_name] == role]


def write_ledger(protocol: Dict, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "A", "B", "C"])
        for sid, v in sorted(protocol["roles"].items()):
            w.writerow([sid, v["A"], v["B"], v["C"]])


def write_protocol_json(protocol: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
