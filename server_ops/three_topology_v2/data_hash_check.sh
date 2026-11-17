#!/usr/bin/env bash
# Verify the unified index + dataset manifest hashes on the server.
# Uses THREE_TOPO_PACKAGE_ROOT (defaults to DATA_ROOT).
set -euo pipefail
DATA_ROOT="${THREE_TOPO_PACKAGE_ROOT:-${DATA_ROOT:?set DATA_ROOT or THREE_TOPO_PACKAGE_ROOT}}"
IDX="$DATA_ROOT/data/processed/three_topology_graph_v1/samples.jsonl"
MAN="$DATA_ROOT/data/processed/three_topology_graph_v1/dataset_manifest.json"
EXP_IDX="5d8bab5dd706c0f48e2310b69cbed877ce28b81e56b8eb18edef5c85801b99f4"
EXP_MAN="c5db8924c9fb21cf33ab771cf231a23cb0723ed8c97c1dc63494b8175c82a6ad"

got_idx=$(sha256sum "$IDX" | awk '{print $1}')
got_man=$(sha256sum "$MAN" | awk '{print $1}')
echo "samples.jsonl      $got_idx"
echo "dataset_manifest   $got_man"
if [[ "$got_idx" != "$EXP_IDX" || "$got_man" != "$EXP_MAN" ]]; then
  echo "HASH MISMATCH — stop. expected idx=$EXP_IDX man=$EXP_MAN"; exit 1
fi
echo "DATA HASH OK"
# optional: transfer_manifest per-file check
if [[ -f "$DATA_ROOT/transfer_manifest.json" ]]; then
  echo "transfer_manifest.json sha256: $(sha256sum "$DATA_ROOT/transfer_manifest.json" | awk '{print $1}')"
fi
