"""
inspect_structural_link_edge_attr.py — Diagnostic script for structural_link edge_attr.

Reads training graphs from the dataset and computes per-dimension statistics
(mean, std, min, max) on the structural_link edge_attr to identify:
  - Which dimensions are near-constant (low variance)
  - Whether the data supports variance-based edge_attr conditioning

Usage:
    python scripts/inspect_structural_link_edge_attr.py \\
        --dataset processed/hetero_graph_dataset_v2 \\
        --max-graphs 10 \\
        --output outputs/diagnostics/structural_link_edge_attr_stats.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.hetero_graph_dataset import HeteroGraphDataset
from src.data.hetero_transforms import HeteroFeatureScaler

STRUCTURAL_LINK_KEY = ("mesh_node", "structural_link", "mesh_node")
FIELD_NAMES = [
    "Kx", "Ky", "Kz",
    "Krx", "Kry", "Krz",
    "BetaAngle", "DistanceRatio",
    "ElasticLinkType", "is_rigid",
]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect structural_link edge_attr statistics."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to processed dataset (e.g. processed/hetero_graph_dataset_v2)",
    )
    parser.add_argument(
        "--max-graphs", type=int, default=10,
        help="Number of training graphs to examine",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()

    # Load scaler and dataset
    scaler_path = os.path.join(args.dataset, "feature_stats.json")
    if os.path.exists(scaler_path):
        scaler = HeteroFeatureScaler.load(Path(scaler_path))
        print(f"[ok] Loaded feature stats from {scaler_path}")
    else:
        scaler = None
        print("[warn] No feature_stats.json found — data may not be standardized")

    dataset = HeteroGraphDataset(
        processed_dir=args.dataset,
        split="train",
        transform=scaler,
    )
    print(f"[ok] Dataset: {len(dataset)} training graphs")
    print(f"[ok] Using up to {args.max_graphs} graphs\n")

    # Collect all structural_link edge_attr tensors
    all_attr = []
    graphs_seen = 0
    total_edges = 0

    for idx in range(min(len(dataset), args.max_graphs)):
        data = dataset[idx]
        if (STRUCTURAL_LINK_KEY in data.edge_types
                and hasattr(data[STRUCTURAL_LINK_KEY], "edge_attr")):
            attr = data[STRUCTURAL_LINK_KEY].edge_attr
            all_attr.append(attr)
            graphs_seen += 1
            total_edges += attr.shape[0]

    if len(all_attr) == 0:
        print("[err] No structural_link edge_attr found in any graph!")
        sys.exit(1)

    stacked = torch.cat(all_attr, dim=0)
    print(f"Graphs with structural_link: {graphs_seen}")
    print(f"Total structural_link edges: {stacked.shape[0]}")
    print(f"Edge attr dimension: {stacked.shape[1]}\n")

    # Per-dimension statistics
    print(f"{'Dimension':<16} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Range':>12} {'Constant?':>10}")
    print("-" * 86)

    results = []
    for i, name in enumerate(FIELD_NAMES):
        col = stacked[:, i]
        mean = col.mean().item()
        std = col.std().item()
        min_v = col.min().item()
        max_v = col.max().item()
        range_v = max_v - min_v
        is_const = "YES" if std < 1e-8 or range_v < 1e-8 else "no"
        print(f"{name:<16} {mean:>12.6f} {std:>12.6f} {min_v:>12.6f} {max_v:>12.6f} {range_v:>12.6f} {is_const:>10}")
        results.append({
            "field": name,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "range": range_v,
            "constant": is_const,
        })

    # Summary
    const_dims = [r for r in results if r["constant"] == "YES"]
    var_dims = [r for r in results if r["constant"] == "no"]
    print(f"\n{'='*50}")
    print(f"Summary: {len(const_dims)} constant dims, {len(var_dims)} variable dims")
    if const_dims:
        print(f"  Constant: {', '.join(d['field'] for d in const_dims)}")
    if var_dims:
        print(f"  Variable: {', '.join(d['field'] for d in var_dims)}")
        min_std = min(d['std'] for d in var_dims)
        max_std = max(d['std'] for d in var_dims)
        print(f"  Std range among variable dims: [{min_std:.6f}, {max_std:.6f}]")

    # Note about standardization
    print(f"\nNote: Data loaded with HeteroFeatureScaler, so statistics are")
    print(f"      POST-STANDARDIZATION (Z-score normalized). Means ≈ 0, Std ≈ 1")
    print(f"      for normally distributed raw features.")
    print(f"      Edge counts per graph: {total_edges // max(1, graphs_seen)} (avg)")

    # Save CSV if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "field", "mean", "std", "min", "max", "range", "constant"
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()
